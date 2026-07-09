from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from ifnet_stage1.config import choose_device
from ifnet_stage1.simulation import SCENARIOS, ChirpSimulator, sim_config_from_dict

from stage2_iccd.eval_active_routed_stage2 import Stage2Bundle
from stage2_iccd.eval_scenarios import parse_noise_types
from stage2_iccd.quality_selector import (
    STAGE2_QUALITY_FEATURE_DIM,
    STAGE2_QUALITY_FEATURES,
    Stage2QualitySelector,
    stage2_quality_features,
    stage2_quality_selector_config_from_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-checkpoint", default="stage2_iccd/runs/simple_multicomponent_long/latest.pt")
    parser.add_argument("--specialist-checkpoint", default="stage2_iccd/runs/poly_multicomponent_refine/latest.pt")
    parser.add_argument("--run-dir", default="stage2_iccd/runs/stage2_quality_selector_poly")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenarios", nargs="*", default=["linear", "quadratic", "cubic", "near_parallel"])
    parser.add_argument("--active-components", type=int, default=2)
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--hidden", type=int, default=72)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--balance-classes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--margin-scale-hz", type=float, default=0.8)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--snr-db-min", type=float, default=-2.0)
    parser.add_argument("--snr-db-max", type=float, default=28.0)
    parser.add_argument("--noise-types-json", default="{white:0.60,colored:0.25,impulsive:0.07,trend:0.08}")
    args = parser.parse_args()

    result = train(args)
    print(json.dumps(result, indent=2))


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    default = Stage2Bundle(args.default_checkpoint, device)
    specialist = Stage2Bundle(args.specialist_checkpoint, device)
    selector_cfg = {"hidden": args.hidden, "dropout": args.dropout}
    selector = Stage2QualitySelector(stage2_quality_selector_config_from_dict(selector_cfg), STAGE2_QUALITY_FEATURE_DIM).to(device)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    sim_cfg = make_sim_cfg(default.cfg["data"], args)
    simulator = ChirpSimulator(sim_cfg, seed=args.seed)
    cfg = {
        "seed": args.seed,
        "default_checkpoint": args.default_checkpoint,
        "specialist_checkpoint": args.specialist_checkpoint,
        "feature_names": STAGE2_QUALITY_FEATURES,
        "feature_dim": STAGE2_QUALITY_FEATURE_DIM,
        "quality_selector": selector_cfg,
        "data": {
            "scenarios": args.scenarios,
            "active_components": args.active_components,
            "snr_db_min": args.snr_db_min,
            "snr_db_max": args.snr_db_max,
            "noise_types": parse_noise_types(args.noise_types_json),
        },
        "train": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "balance_classes": args.balance_classes,
            "margin_scale_hz": args.margin_scale_hz,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    history = []
    best_val_mae = float("inf")
    pbar = trange(1, args.steps + 1, desc="stage2-quality-selector")
    for step in pbar:
        batch = simulator.generate_batch(args.batch_size, device=device)
        features, labels, branch_mae = build_selector_batch(default, specialist, batch, device)
        selector.train()
        logits = selector(features)
        loss = weighted_selector_loss(logits, labels, branch_mae, balance_classes=args.balance_classes, margin_scale_hz=args.margin_scale_hz)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            selected_mae = branch_mae.gather(1, pred.view(-1, 1)).mean()
            default_mae = branch_mae[:, 0].mean()
            specialist_mae = branch_mae[:, 1].mean()
            oracle_mae = branch_mae.min(dim=1).values.mean()
            acc = (pred == labels).float().mean()
        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "accuracy": float(acc.detach().cpu()),
            "selected_if_mae_hz": float(selected_mae.detach().cpu()),
            "default_if_mae_hz": float(default_mae.detach().cpu()),
            "specialist_if_mae_hz": float(specialist_mae.detach().cpu()),
            "oracle_if_mae_hz": float(oracle_mae.detach().cpu()),
        }
        history.append(row)
        pbar.set_postfix(loss=f"{row['loss']:.3f}", acc=f"{row['accuracy']:.2f}", mae=f"{row['selected_if_mae_hz']:.2f}")

        if step % args.print_every == 0 or step == args.steps:
            val = evaluate_selector(selector, default, specialist, simulator, args.batch_size, args.val_batches, device)
            row.update({f"val_{key}": value for key, value in val.items()})
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if val["selected_if_mae_hz"] < best_val_mae:
                best_val_mae = val["selected_if_mae_hz"]
                save_selector(run_dir / "best.pt", selector, cfg, step)
        if step == args.steps:
            save_selector(run_dir / "latest.pt", selector, cfg, step)

    return {"run_dir": str(run_dir), "last": history[-1] if history else {}, "best_val_selected_if_mae_hz": best_val_mae}


@torch.no_grad()
def build_selector_batch(
    default: Stage2Bundle,
    specialist: Stage2Bundle,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_default = default.run(batch)
    out_specialist = specialist.run(batch)
    features = stage2_quality_features(out_default, out_specialist, batch["signal"]).to(device)
    default_mae = per_sample_if_mae(out_default["refined_if_hz"], batch["if_hz"], batch.get("active_mask"))
    specialist_mae = per_sample_if_mae(out_specialist["refined_if_hz"], batch["if_hz"], batch.get("active_mask"))
    branch_mae = torch.stack([default_mae, specialist_mae], dim=1).to(device)
    labels = branch_mae.argmin(dim=1)
    return features, labels, branch_mae


@torch.no_grad()
def evaluate_selector(
    selector: Stage2QualitySelector,
    default: Stage2Bundle,
    specialist: Stage2Bundle,
    simulator: ChirpSimulator,
    batch_size: int,
    batches: int,
    device: torch.device,
) -> dict[str, float]:
    selector.eval()
    rows = []
    for _ in range(batches):
        batch = simulator.generate_batch(batch_size, device=device)
        features, labels, branch_mae = build_selector_batch(default, specialist, batch, device)
        logits = selector(features)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        selected_mae = branch_mae.gather(1, pred.view(-1, 1)).squeeze(1)
        rows.append(
            {
                "accuracy": (pred == labels).float(),
                "selected_if_mae_hz": selected_mae,
                "default_if_mae_hz": branch_mae[:, 0],
                "specialist_if_mae_hz": branch_mae[:, 1],
                "oracle_if_mae_hz": branch_mae.min(dim=1).values,
                "uses_specialist": (pred == 1).float(),
                "oracle_uses_specialist": (labels == 1).float(),
                "confidence": probs.max(dim=1).values,
            }
        )
    out: dict[str, float] = {}
    for key in rows[0]:
        values = torch.cat([row[key].reshape(-1) for row in rows], dim=0)
        out[key] = float(values.mean().detach().cpu())
        if key.endswith("if_mae_hz"):
            out[f"{key}_p90"] = float(torch.quantile(values, 0.90).detach().cpu())
            out[f"{key}_p95"] = float(torch.quantile(values, 0.95).detach().cpu())
    return out


def make_sim_cfg(base_data: dict[str, Any], args: argparse.Namespace):
    sim_data = dict(base_data)
    sim_data["active_components"] = int(args.active_components)
    sim_data["snr_db_min"] = float(args.snr_db_min)
    sim_data["snr_db_max"] = float(args.snr_db_max)
    sim_data["noise_types"] = parse_noise_types(args.noise_types_json)
    selected = set(args.scenarios)
    sim_data["scenario_weights"] = {name: (1.0 if name in selected else 0.0) for name in SCENARIOS}
    return sim_config_from_dict(sim_data)


def weighted_selector_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    branch_mae: torch.Tensor,
    *,
    balance_classes: bool = True,
    margin_scale_hz: float = 0.8,
) -> torch.Tensor:
    per_sample = F.cross_entropy(logits, labels, reduction="none")
    margin = (branch_mae[:, 0] - branch_mae[:, 1]).abs()
    sample_weight = 0.25 + (margin / float(max(margin_scale_hz, 1.0e-6))).clamp(max=1.0)
    if balance_classes:
        counts = torch.bincount(labels, minlength=2).to(dtype=logits.dtype, device=logits.device).clamp_min(1.0)
        class_weight = counts.sum() / (2.0 * counts)
        sample_weight = sample_weight * class_weight[labels]
    return (per_sample * sample_weight.detach()).sum() / sample_weight.detach().sum().clamp_min(1.0)


def per_sample_if_mae(pred_if: torch.Tensor, target_if: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
    if pred_if.shape[:2] != target_if.shape[:2]:
        raise ValueError("Predicted and target IF tensors must share [B,Q].")
    bsz, q, _ = pred_if.shape
    if active_mask is None:
        active_mask = torch.ones((bsz, q), device=pred_if.device, dtype=pred_if.dtype)
    else:
        active_mask = active_mask.to(device=pred_if.device, dtype=pred_if.dtype)
    perms = list(itertools.permutations(range(q)))
    rows = torch.arange(q, device=pred_if.device)
    component_cost = torch.empty((bsz, q, q), device=pred_if.device, dtype=pred_if.dtype)
    for pred_idx in range(q):
        component_cost[:, pred_idx, :] = (pred_if[:, pred_idx : pred_idx + 1] - target_if).abs().mean(dim=-1)
    costs = []
    for perm in perms:
        perm_tensor = torch.tensor(perm, device=pred_if.device)
        matched = component_cost[:, rows, perm_tensor]
        matched_mask = active_mask[:, perm_tensor]
        costs.append((matched * matched_mask).sum(dim=1) / matched_mask.sum(dim=1).clamp_min(1.0))
    return torch.stack(costs, dim=1).min(dim=1).values


def save_selector(path: Path, selector: Stage2QualitySelector, cfg: dict[str, Any], step: int) -> None:
    torch.save(
        {
            "step": step,
            "config": cfg,
            "feature_dim": STAGE2_QUALITY_FEATURE_DIM,
            "feature_names": STAGE2_QUALITY_FEATURES,
            "model": selector.state_dict(),
        },
        path,
    )


if __name__ == "__main__":
    main()
