from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .quality_selector import QUALITY_FEATURE_DIM, QualitySelector, quality_selector_config_from_dict
from .simulation import SCENARIOS, ChirpSimulator
from .train_quality_selector import build_quality_batch, load_expert, load_router


@torch.no_grad()
def evaluate_quality_selector(
    selector_checkpoint: str | Path,
    router_checkpoint: str | Path,
    output_dir: str | Path,
    scenarios: list[str],
    batch_size: int,
    batches: int,
    seed: int,
    device_name: str,
    topk: int,
    margins: list[float],
    protect_top_routes: set[str],
    protect_min_prob: float,
) -> dict:
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    ckpt = torch.load(selector_checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    selector = QualitySelector(quality_selector_config_from_dict(cfg.get("quality_selector")), ckpt.get("feature_dim", QUALITY_FEATURE_DIM)).to(device)
    selector.load_state_dict(ckpt["model"])
    selector.eval()

    router, sim_cfg, router_stft_cfg, route_names, _scenario_to_route = load_router(router_checkpoint, device)
    expert_paths = cfg.get("expert_paths")
    experts = {name: load_expert(path, device) for name, path in expert_paths.items() if name in route_names}
    simulator = ChirpSimulator(sim_cfg, seed=seed)

    results = {}
    for scenario in scenarios:
        top1_maes = []
        argmax_maes = []
        oracle_maes = []
        margin_maes = {str(margin): [] for margin in margins}
        argmax_changed = 0
        margin_changed = {str(margin): 0 for margin in margins}
        total = 0
        for _ in range(batches):
            batch = simulator.generate_batch(batch_size, device=device, scenarios=[scenario])
            features, maes, _stats = build_quality_batch(
                batch,
                router,
                router_stft_cfg,
                sim_cfg.fs,
                route_names,
                experts,
                device,
                topk,
            )
            scores = selector(features.reshape(-1, features.shape[-1])).reshape(features.shape[0], features.shape[1])
            top1_mae = maes[:, 0]
            oracle_mae = maes.min(dim=1).values
            argmax_choice = scores.argmax(dim=1)
            protected = protected_top1_mask(features, route_names, protect_top_routes, protect_min_prob)
            argmax_choice = torch.where(protected, torch.zeros_like(argmax_choice), argmax_choice)
            argmax_mae = maes.gather(1, argmax_choice.view(-1, 1)).squeeze(1)
            top1_score = scores[:, 0]

            top1_maes.extend(top1_mae.detach().cpu().tolist())
            oracle_maes.extend(oracle_mae.detach().cpu().tolist())
            argmax_maes.extend(argmax_mae.detach().cpu().tolist())
            argmax_changed += int((argmax_choice != 0).sum().detach().cpu())
            total += int(maes.shape[0])

            for margin in margins:
                choice = torch.zeros_like(argmax_choice)
                for candidate_idx in range(1, scores.shape[1]):
                    better = scores[:, candidate_idx] - top1_score > float(margin)
                    choice = torch.where(better, torch.full_like(choice, candidate_idx), choice)
                choice = torch.where(protected, torch.zeros_like(choice), choice)
                selected = maes.gather(1, choice.view(-1, 1)).squeeze(1)
                key = str(margin)
                margin_maes[key].extend(selected.detach().cpu().tolist())
                margin_changed[key] += int((choice != 0).sum().detach().cpu())

        scenario_result = {
            "top1_mae_hz": _summary(top1_maes),
            "argmax_mae_hz": _summary(argmax_maes),
            "oracle_mae_hz": _summary(oracle_maes),
            "argmax_gain_vs_top1": float(np.mean(top1_maes) - np.mean(argmax_maes)),
            "oracle_gain_vs_top1": float(np.mean(top1_maes) - np.mean(oracle_maes)),
            "argmax_change_fraction": argmax_changed / max(1, total),
            "num_examples": total,
            "gated": {},
        }
        for margin in margins:
            key = str(margin)
            scenario_result["gated"][key] = {
                "mae_hz": _summary(margin_maes[key]),
                "gain_vs_top1": float(np.mean(top1_maes) - np.mean(margin_maes[key])),
                "change_fraction": margin_changed[key] / max(1, total),
            }
        results[scenario] = scenario_result

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality_selector_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def protected_top1_mask(
    features: torch.Tensor,
    route_names: tuple[str, ...],
    protect_top_routes: set[str],
    protect_min_prob: float,
) -> torch.Tensor:
    if not protect_top_routes:
        return torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
    route_probs = features[:, 0, : len(route_names)]
    route_onehot_start = len(route_names)
    top1_route = features[:, 0, route_onehot_start : route_onehot_start + len(route_names)].argmax(dim=1)
    top1_prob = route_probs.gather(1, top1_route.view(-1, 1)).squeeze(1)
    protected_indices = [idx for idx, name in enumerate(route_names) if name in protect_top_routes]
    if not protected_indices:
        return torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
    mask = torch.zeros_like(top1_route, dtype=torch.bool)
    for idx in protected_indices:
        mask = mask | (top1_route == idx)
    return mask & (top1_prob >= float(protect_min_prob))


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)) if values else float("nan"),
        "median": float(np.median(values)) if values else float("nan"),
        "p95": float(np.percentile(values, 95)) if values else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--router-checkpoint", default="ifnet_stage1/runs/router_hard_v3/latest.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=67890)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.35, 0.5, 0.75])
    parser.add_argument("--protect-top-routes", nargs="*", default=[])
    parser.add_argument("--protect-min-prob", type=float, default=0.0)
    args = parser.parse_args()
    result = evaluate_quality_selector(
        selector_checkpoint=args.selector_checkpoint,
        router_checkpoint=args.router_checkpoint,
        output_dir=args.output_dir,
        scenarios=args.scenarios,
        batch_size=args.batch_size,
        batches=args.batches,
        seed=args.seed,
        device_name=args.device,
        topk=args.topk,
        margins=args.margins,
        protect_top_routes=set(args.protect_top_routes),
        protect_min_prob=args.protect_min_prob,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
