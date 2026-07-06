from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from .model import IFNetUNet, model_config_from_dict, soft_argmax_if
from .postprocess import apply_if_postprocess
from .predict_routed import DEFAULT_EXPERTS, DEFAULT_POSTPROCESS
from .quality_selector import (
    QUALITY_FEATURE_DIM,
    QualitySelector,
    candidate_quality_features,
    quality_selector_config_from_dict,
)
from .router import DEFAULT_SCENARIO_TO_ROUTE, ROUTE_NAMES, HardRouteClassifier, router_config_from_dict
from .simulation import ChirpSimulator, sim_config_from_dict
from .tf import feature_channels, log_spectrogram, sample_if_to_frames, stft_config_from_dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def best_align_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    q = pred.shape[0]
    best_cost = None
    for perm in itertools.permutations(range(q)):
        cost = (pred[list(perm)] - target).abs().mean()
        if best_cost is None or cost < best_cost:
            best_cost = cost
    return best_cost if best_cost is not None else pred.new_tensor(0.0)


@torch.no_grad()
def load_router(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    route_names = tuple(ckpt.get("route_names", cfg.get("route_names", ROUTE_NAMES)))
    scenario_to_route = dict(ckpt.get("scenario_to_route", cfg.get("scenario_to_route", DEFAULT_SCENARIO_TO_ROUTE)))
    sim_cfg = sim_config_from_dict(cfg["data"])
    stft_cfg = stft_config_from_dict(cfg["stft"])
    router_cfg = router_config_from_dict(cfg["router"])
    router = HardRouteClassifier(feature_channels(stft_cfg), len(route_names), router_cfg).to(device)
    router.load_state_dict(ckpt["model"])
    router.eval()
    return router, sim_cfg, stft_cfg, route_names, scenario_to_route


@torch.no_grad()
def load_expert(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    sim_cfg = sim_config_from_dict(cfg["data"])
    stft_cfg = stft_config_from_dict(cfg["stft"])
    model_cfg = model_config_from_dict(cfg["model"])
    model = IFNetUNet(feature_channels(stft_cfg), sim_cfg.num_components, model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, sim_cfg, stft_cfg, model_cfg


@torch.no_grad()
def build_quality_batch(
    batch: dict,
    router: HardRouteClassifier,
    router_stft_cfg,
    router_fs: float,
    route_names: tuple[str, ...],
    experts: dict[str, tuple],
    device: torch.device,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    signal = batch["signal"]
    router_feats, _ = log_spectrogram(signal, router_stft_cfg, router_fs)
    route_probs = torch.softmax(router(router_feats), dim=1)
    top_values, top_indices = torch.topk(route_probs, k=min(topk, route_probs.shape[1]), dim=1)
    del top_values

    bsz, k = top_indices.shape
    candidate_data: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for route_idx in sorted(set(int(x) for x in top_indices.reshape(-1).detach().cpu().tolist())):
        route_name = route_names[route_idx]
        model, sim_cfg, stft_cfg, model_cfg = experts[route_name]
        sample_positions = (top_indices == route_idx).nonzero(as_tuple=False)
        sample_indices = sample_positions[:, 0].unique()
        route_signal = signal.index_select(0, sample_indices)
        feats, freq_grid = log_spectrogram(route_signal, stft_cfg, sim_cfg.fs)
        logits = model(feats)
        pred_if, probs = soft_argmax_if(logits, freq_grid, model_cfg.temperature)
        pred_if = apply_if_postprocess(
            pred_if,
            mode=DEFAULT_POSTPROCESS[route_name],
            degree=3,
            robust_iters=4,
            huber_hz=10.0,
            probs=probs,
            freq_grid=freq_grid,
            topk=9,
        )
        target_if = sample_if_to_frames(batch["if_hz"].index_select(0, sample_indices), logits.shape[-1], stft_cfg.hop_length)
        for local_idx, sample_idx_tensor in enumerate(sample_indices):
            sample_idx = int(sample_idx_tensor.detach().cpu())
            candidate_data[(sample_idx, route_idx)] = (
                pred_if[local_idx : local_idx + 1],
                probs[local_idx : local_idx + 1],
                target_if[local_idx : local_idx + 1],
            )

    feature_rows = []
    mae_rows = []
    oracle_mae = []
    top1_mae = []
    for sample_idx in range(bsz):
        sample_features = []
        sample_maes = []
        for pos in range(k):
            route_idx = int(top_indices[sample_idx, pos].detach().cpu())
            pred_if, probs, target_if = candidate_data[(sample_idx, route_idx)]
            feat = candidate_quality_features(
                route_probs[sample_idx],
                route_idx,
                pred_if,
                probs,
                num_routes=len(route_names),
            )
            mae = best_align_mae(pred_if[0], target_if[0])
            sample_features.append(feat)
            sample_maes.append(mae)
        feature_rows.append(torch.stack(sample_features, dim=0))
        mae_tensor = torch.stack(sample_maes)
        mae_rows.append(mae_tensor)
        oracle_mae.append(mae_tensor.min())
        top1_mae.append(mae_tensor[0])

    features = torch.stack(feature_rows, dim=0).to(device)
    maes = torch.stack(mae_rows, dim=0).to(device)
    stats = {
        "oracle_mae": float(torch.stack(oracle_mae).mean().detach().cpu()),
        "top1_mae": float(torch.stack(top1_mae).mean().detach().cpu()),
        "oracle_gain": float((torch.stack(top1_mae).mean() - torch.stack(oracle_mae).mean()).detach().cpu()),
    }
    return features, maes, stats


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    router, sim_cfg, router_stft_cfg, route_names, scenario_to_route = load_router(args.router_checkpoint, device)
    del scenario_to_route
    experts = {name: load_expert(path, device) for name, path in DEFAULT_EXPERTS.items() if name in route_names}
    simulator = ChirpSimulator(sim_cfg, seed=args.seed)

    cfg = {
        "seed": args.seed,
        "router_checkpoint": args.router_checkpoint,
        "expert_paths": DEFAULT_EXPERTS,
        "route_names": route_names,
        "topk": args.topk,
        "quality_selector": {"hidden": args.hidden, "dropout": args.dropout},
        "train": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "val_batches": args.val_batches,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    selector = QualitySelector(quality_selector_config_from_dict(cfg["quality_selector"]), QUALITY_FEATURE_DIM).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_gain = float("-inf")
    pbar = trange(1, args.steps + 1, desc="quality-selector")
    for step in pbar:
        batch = simulator.generate_batch(args.batch_size, device=device)
        features, maes, stats = build_quality_batch(
            batch,
            router,
            router_stft_cfg,
            sim_cfg.fs,
            route_names,
            experts,
            device,
            args.topk,
        )
        selector.train()
        flat_scores = selector(features.reshape(-1, features.shape[-1]))
        scores = flat_scores.reshape(features.shape[0], features.shape[1])
        labels = maes.argmin(dim=1)
        rank_loss = F.cross_entropy(scores, labels)
        pred_mae = maes.gather(1, scores.argmax(dim=1, keepdim=True)).mean()
        oracle_mae = maes.min(dim=1).values.mean()
        top1_mae = maes[:, 0].mean()

        optimizer.zero_grad(set_to_none=True)
        rank_loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), args.grad_clip)
        optimizer.step()

        row = {
            "step": step,
            "loss": float(rank_loss.detach().cpu()),
            "train_select_mae": float(pred_mae.detach().cpu()),
            "train_top1_mae": float(top1_mae.detach().cpu()),
            "train_oracle_mae": float(oracle_mae.detach().cpu()),
            **stats,
        }
        if step % args.print_every == 0 or step == args.steps:
            val = evaluate_selector(
                selector,
                simulator,
                router,
                router_stft_cfg,
                sim_cfg.fs,
                route_names,
                experts,
                device,
                args.batch_size,
                args.val_batches,
                args.topk,
            )
            row.update({f"val_{key}": value for key, value in val.items()})
            if row["val_gain_vs_top1"] > best_gain:
                best_gain = row["val_gain_vs_top1"]
                save_selector(run_dir / "best.pt", selector, cfg)
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        history.append(row)
        pbar.set_postfix(loss=f"{row['loss']:.3f}", sel=f"{row['train_select_mae']:.2f}", top1=f"{row['train_top1_mae']:.2f}")

    save_selector(run_dir / "latest.pt", selector, cfg)
    return {"run_dir": str(run_dir), "last": history[-1]}


def save_selector(path: Path, selector: QualitySelector, cfg: dict) -> None:
    torch.save(
        {
            "config": cfg,
            "model": selector.state_dict(),
            "feature_dim": QUALITY_FEATURE_DIM,
        },
        path,
    )


@torch.no_grad()
def evaluate_selector(
    selector: QualitySelector,
    simulator: ChirpSimulator,
    router: HardRouteClassifier,
    router_stft_cfg,
    router_fs: float,
    route_names: tuple[str, ...],
    experts: dict[str, tuple],
    device: torch.device,
    batch_size: int,
    batches: int,
    topk: int,
) -> dict[str, float]:
    selector.eval()
    selected = []
    top1 = []
    oracle = []
    correct = []
    for _ in range(batches):
        batch = simulator.generate_batch(batch_size, device=device)
        features, maes, _stats = build_quality_batch(
            batch,
            router,
            router_stft_cfg,
            router_fs,
            route_names,
            experts,
            device,
            topk,
        )
        scores = selector(features.reshape(-1, features.shape[-1])).reshape(features.shape[0], features.shape[1])
        chosen = scores.argmax(dim=1)
        best = maes.argmin(dim=1)
        selected.append(maes.gather(1, chosen.view(-1, 1)).squeeze(1))
        top1.append(maes[:, 0])
        oracle.append(maes.min(dim=1).values)
        correct.append((chosen == best).float())
    selected_t = torch.cat(selected)
    top1_t = torch.cat(top1)
    oracle_t = torch.cat(oracle)
    correct_t = torch.cat(correct)
    return {
        "select_mae": float(selected_t.mean().cpu()),
        "top1_mae": float(top1_t.mean().cpu()),
        "oracle_mae": float(oracle_t.mean().cpu()),
        "rank_accuracy": float(correct_t.mean().cpu()),
        "gain_vs_top1": float((top1_t.mean() - selected_t.mean()).cpu()),
        "oracle_gap": float((selected_t.mean() - oracle_t.mean()).cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-checkpoint", default="ifnet_stage1/runs/router_hard_v3/latest.pt")
    parser.add_argument("--run-dir", default="ifnet_stage1/runs/quality_selector_v1")
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--val-batches", type=int, default=8)
    args = parser.parse_args()
    result = train(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
