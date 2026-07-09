from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from ifnet_stage1.config import choose_device
from ifnet_stage1.simulation import SCENARIOS, ChirpSimulator, sim_config_from_dict

from stage2_iccd.eval_scenarios import parse_noise_types

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_old_new_stage2_comparison import Stage2Bundle, metrics_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-checkpoint", default="stage2_iccd/runs/simple_multicomponent_long/latest.pt")
    parser.add_argument("--b-checkpoint", default="stage2_iccd/runs/poly_multicomponent_refine/latest.pt")
    parser.add_argument("--output-dir", default="stage2_iccd/runs/poly_multicomponent_refine/compare_to_simple")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenario", default="quadratic")
    parser.add_argument("--active-components", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--snr-db-min", type=float, default=4.0)
    parser.add_argument("--snr-db-max", type=float, default=28.0)
    parser.add_argument("--noise-types-json", default="{white:0.75,colored:0.25,impulsive:0.0,trend:0.0}")
    args = parser.parse_args()

    device = choose_device(args.device)
    a = Stage2Bundle(args.a_checkpoint, device)
    b = Stage2Bundle(args.b_checkpoint, device)
    sim_data = dict(a.cfg["data"])
    sim_data["active_components"] = int(args.active_components)
    sim_data["snr_db_min"] = float(args.snr_db_min)
    sim_data["snr_db_max"] = float(args.snr_db_max)
    sim_data["noise_types"] = parse_noise_types(args.noise_types_json)
    sim_data["scenario_weights"] = {name: (1.0 if name == args.scenario else 0.0) for name in SCENARIOS}
    sim_cfg = sim_config_from_dict(sim_data)
    simulator = ChirpSimulator(sim_cfg, seed=args.seed)

    rows: list[dict[str, Any]] = []
    for idx in range(args.num_samples):
        batch = simulator.generate_batch(1, device=device)
        out_a = a.run(batch)
        out_b = b.run(batch)
        met_a = metrics_for(out_a, batch, sim_cfg.fs, a.weights)
        met_b = metrics_for(out_b, batch, sim_cfg.fs, b.weights)
        rows.append(
            {
                "idx": idx,
                "scenario": args.scenario,
                "active_components": args.active_components,
                "a_if_mae_hz": met_a["if_mae_hz"],
                "b_if_mae_hz": met_b["if_mae_hz"],
                "delta_if_mae_hz": met_b["if_mae_hz"] - met_a["if_mae_hz"],
                "a_rec_snr_db": met_a["rec_snr_db"],
                "b_rec_snr_db": met_b["rec_snr_db"],
                "delta_rec_snr_db": met_b["rec_snr_db"] - met_a["rec_snr_db"],
            }
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    payload = {
        "a_checkpoint": args.a_checkpoint,
        "b_checkpoint": args.b_checkpoint,
        "scenario": args.scenario,
        "active_components": args.active_components,
        "num_samples": args.num_samples,
        "data": {
            "snr_db_min": args.snr_db_min,
            "snr_db_max": args.snr_db_max,
            "noise_types": parse_noise_types(args.noise_types_json),
        },
        "summary": summary,
        "rows": rows,
    }
    (out_dir / "comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, indent=2))


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    a = np.array([row["a_if_mae_hz"] for row in rows], dtype=np.float64)
    b = np.array([row["b_if_mae_hz"] for row in rows], dtype=np.float64)
    snr_a = np.array([row["a_rec_snr_db"] for row in rows], dtype=np.float64)
    snr_b = np.array([row["b_rec_snr_db"] for row in rows], dtype=np.float64)
    return {
        "a_if_mae_mean": float(a.mean()),
        "b_if_mae_mean": float(b.mean()),
        "a_if_mae_p50": float(np.percentile(a, 50)),
        "b_if_mae_p50": float(np.percentile(b, 50)),
        "a_if_mae_p90": float(np.percentile(a, 90)),
        "b_if_mae_p90": float(np.percentile(b, 90)),
        "a_if_mae_p95": float(np.percentile(a, 95)),
        "b_if_mae_p95": float(np.percentile(b, 95)),
        "b_better_rate": float((b < a).mean()),
        "a_rec_snr_mean": float(snr_a.mean()),
        "b_rec_snr_mean": float(snr_b.mean()),
    }


if __name__ == "__main__":
    main()
