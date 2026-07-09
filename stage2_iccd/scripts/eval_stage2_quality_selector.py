from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ifnet_stage1.config import choose_device
from ifnet_stage1.simulation import SCENARIOS, ChirpSimulator, sim_config_from_dict

from stage2_iccd.eval_active_routed_stage2 import Stage2Bundle
from stage2_iccd.eval_scenarios import parse_noise_types
from stage2_iccd.quality_context import QualityContextProvider
from stage2_iccd.quality_selector import (
    Stage2QualitySelector,
    stage2_quality_selector_config_from_dict,
)
from train_stage2_quality_selector import build_selector_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-checkpoint", default="stage2_iccd/runs/stage2_quality_selector_poly/latest.pt")
    parser.add_argument("--default-checkpoint", default=None)
    parser.add_argument("--specialist-checkpoint", default=None)
    parser.add_argument("--stage1-router-checkpoint", default=None)
    parser.add_argument("--active-count-checkpoint", default=None)
    parser.add_argument("--output-dir", default="stage2_iccd/runs/stage2_quality_selector_poly/eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenarios", nargs="*", default=["linear", "quadratic", "cubic", "near_parallel"])
    parser.add_argument("--active-components", type=int, default=2)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--snr-db-min", type=float, default=-2.0)
    parser.add_argument("--snr-db-max", type=float, default=28.0)
    parser.add_argument("--noise-types-json", default="{white:0.60,colored:0.25,impulsive:0.07,trend:0.08}")
    args = parser.parse_args()

    result = evaluate(args)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = choose_device(args.device)
    ckpt = torch.load(args.selector_checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    default_checkpoint = args.default_checkpoint or cfg["default_checkpoint"]
    specialist_checkpoint = args.specialist_checkpoint or cfg["specialist_checkpoint"]
    feature_names = tuple(ckpt.get("feature_names", cfg.get("feature_names", [])))
    if not feature_names:
        feature_names = tuple(cfg.get("feature_names", []))
    selector = Stage2QualitySelector(
        stage2_quality_selector_config_from_dict(cfg.get("quality_selector")),
        int(ckpt.get("feature_dim", cfg.get("feature_dim", 24))),
    ).to(device)
    selector.load_state_dict(ckpt["model"])
    selector.eval()
    default = Stage2Bundle(default_checkpoint, device)
    specialist = Stage2Bundle(specialist_checkpoint, device)
    context_provider = QualityContextProvider(
        device=device,
        stage1_router_checkpoint=args.stage1_router_checkpoint or cfg.get("stage1_router_checkpoint") or None,
        active_count_checkpoint=args.active_count_checkpoint or cfg.get("active_count_checkpoint") or None,
    )

    rows = []
    summaries = {}
    for scenario_idx, scenario in enumerate(args.scenarios):
        sim_data = dict(default.cfg["data"])
        sim_data["active_components"] = int(args.active_components)
        sim_data["snr_db_min"] = float(args.snr_db_min)
        sim_data["snr_db_max"] = float(args.snr_db_max)
        sim_data["noise_types"] = parse_noise_types(args.noise_types_json)
        sim_data["scenario_weights"] = {name: (1.0 if name == scenario else 0.0) for name in SCENARIOS}
        sim_cfg = sim_config_from_dict(sim_data)
        simulator = ChirpSimulator(sim_cfg, seed=args.seed + scenario_idx * 1009)
        scenario_rows = []
        for batch_idx in range(args.batches):
            batch = simulator.generate_batch(args.batch_size, device=device)
            features, labels, branch_mae = build_selector_batch(
                default,
                specialist,
                batch,
                device,
                context_provider=context_provider,
                feature_names=feature_names,
            )
            logits = selector(features)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1)
            for sample_idx in range(args.batch_size):
                chosen = int(pred[sample_idx].detach().cpu())
                label = int(labels[sample_idx].detach().cpu())
                row = {
                    "scenario": scenario,
                    "batch_idx": batch_idx,
                    "sample_idx": sample_idx,
                    "default_if_mae_hz": float(branch_mae[sample_idx, 0].detach().cpu()),
                    "specialist_if_mae_hz": float(branch_mae[sample_idx, 1].detach().cpu()),
                    "selected_if_mae_hz": float(branch_mae[sample_idx, chosen].detach().cpu()),
                    "oracle_if_mae_hz": float(branch_mae[sample_idx].min().detach().cpu()),
                    "selected_branch": chosen,
                    "oracle_branch": label,
                    "selector_correct": float(chosen == label),
                    "selector_confidence": float(probs[sample_idx, chosen].detach().cpu()),
                }
                rows.append(row)
                scenario_rows.append(row)
        summaries[scenario] = summarize_rows(scenario_rows)
    summaries["aggregate"] = summarize_rows(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "selector_checkpoint": args.selector_checkpoint,
        "default_checkpoint": str(default_checkpoint),
        "specialist_checkpoint": str(specialist_checkpoint),
        "batches": args.batches,
        "batch_size": args.batch_size,
        "summary": summaries,
        "rows": rows,
    }
    (out_dir / "stage2_quality_selector_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "stage2_quality_selector_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ("default_if_mae_hz", "specialist_if_mae_hz", "selected_if_mae_hz", "oracle_if_mae_hz"):
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_p90"] = float(np.percentile(values, 90))
        out[f"{key}_p95"] = float(np.percentile(values, 95))
    out["selector_accuracy"] = float(np.mean([float(row["selector_correct"]) for row in rows]))
    out["uses_specialist_rate"] = float(np.mean([float(row["selected_branch"] == 1) for row in rows]))
    out["oracle_uses_specialist_rate"] = float(np.mean([float(row["oracle_branch"] == 1) for row in rows]))
    out["selector_confidence"] = float(np.mean([float(row["selector_confidence"]) for row in rows]))
    return out


if __name__ == "__main__":
    main()
