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
    parser.add_argument("--default-checkpoint", default="stage2_iccd/runs/simple_multicomponent_long/latest.pt")
    parser.add_argument("--specialist-checkpoint", default="stage2_iccd/runs/poly_multicomponent_refine/latest.pt")
    parser.add_argument("--output-dir", default="stage2_iccd/runs/poly_multicomponent_refine/quality_gate")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenarios", nargs="*", default=["linear", "quadratic", "cubic", "near_parallel"])
    parser.add_argument("--active-components", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--snr-db-min", type=float, default=4.0)
    parser.add_argument("--snr-db-max", type=float, default=28.0)
    parser.add_argument("--noise-types-json", default="{white:0.75,colored:0.25,impulsive:0.0,trend:0.0}")
    parser.add_argument("--delta-penalty", type=float, default=0.015)
    parser.add_argument("--smooth-penalty", type=float, default=0.000002)
    parser.add_argument("--score-margin", type=float, default=0.05)
    parser.add_argument("--print-rows", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    default = Stage2Bundle(args.default_checkpoint, device)
    specialist = Stage2Bundle(args.specialist_checkpoint, device)
    rows: list[dict[str, Any]] = []

    for scenario_idx, scenario in enumerate(args.scenarios):
        sim_data = dict(default.cfg["data"])
        sim_data["active_components"] = int(args.active_components)
        sim_data["snr_db_min"] = float(args.snr_db_min)
        sim_data["snr_db_max"] = float(args.snr_db_max)
        sim_data["noise_types"] = parse_noise_types(args.noise_types_json)
        sim_data["scenario_weights"] = {name: (1.0 if name == scenario else 0.0) for name in SCENARIOS}
        sim_cfg = sim_config_from_dict(sim_data)
        simulator = ChirpSimulator(sim_cfg, seed=args.seed + scenario_idx * 1009)

        for idx in range(args.num_samples):
            batch = simulator.generate_batch(1, device=device)
            out_default = default.run(batch)
            out_specialist = specialist.run(batch)
            met_default = metrics_for(out_default, batch, sim_cfg.fs, default.weights)
            met_specialist = metrics_for(out_specialist, batch, sim_cfg.fs, specialist.weights)
            q_default = unsupervised_quality(out_default, batch, args.delta_penalty, args.smooth_penalty)
            q_specialist = unsupervised_quality(out_specialist, batch, args.delta_penalty, args.smooth_penalty)

            gate_uses_specialist = q_specialist["quality_score"] > q_default["quality_score"] + args.score_margin
            gate_metrics = met_specialist if gate_uses_specialist else met_default
            oracle_uses_specialist = met_specialist["if_mae_hz"] < met_default["if_mae_hz"]
            oracle_metrics = met_specialist if oracle_uses_specialist else met_default

            rows.append(
                {
                    "scenario": scenario,
                    "idx": idx,
                    "default_if_mae_hz": met_default["if_mae_hz"],
                    "specialist_if_mae_hz": met_specialist["if_mae_hz"],
                    "gated_if_mae_hz": gate_metrics["if_mae_hz"],
                    "oracle_if_mae_hz": oracle_metrics["if_mae_hz"],
                    "default_rec_snr_db": met_default["rec_snr_db"],
                    "specialist_rec_snr_db": met_specialist["rec_snr_db"],
                    "gated_rec_snr_db": gate_metrics["rec_snr_db"],
                    "oracle_rec_snr_db": oracle_metrics["rec_snr_db"],
                    "default_observed_snr_db": q_default["observed_snr_db"],
                    "specialist_observed_snr_db": q_specialist["observed_snr_db"],
                    "default_quality_score": q_default["quality_score"],
                    "specialist_quality_score": q_specialist["quality_score"],
                    "default_delta_rms_hz": q_default["delta_rms_hz"],
                    "specialist_delta_rms_hz": q_specialist["delta_rms_hz"],
                    "default_smooth_hz2": q_default["smooth_hz2"],
                    "specialist_smooth_hz2": q_specialist["smooth_hz2"],
                    "gate_uses_specialist": float(gate_uses_specialist),
                    "oracle_uses_specialist": float(oracle_uses_specialist),
                    "gate_matches_oracle": float(gate_uses_specialist == oracle_uses_specialist),
                }
            )

    summary = summarize(rows, args.scenarios)
    payload = {
        "default_checkpoint": args.default_checkpoint,
        "specialist_checkpoint": args.specialist_checkpoint,
        "active_components": args.active_components,
        "num_samples_per_scenario": args.num_samples,
        "data": {
            "snr_db_min": args.snr_db_min,
            "snr_db_max": args.snr_db_max,
            "noise_types": parse_noise_types(args.noise_types_json),
        },
        "gate": {
            "delta_penalty": args.delta_penalty,
            "smooth_penalty": args.smooth_penalty,
            "score_margin": args.score_margin,
        },
        "summary": summary,
        "rows": rows,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality_gate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "quality_gate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    printed_payload = payload if args.print_rows else {key: value for key, value in payload.items() if key != "rows"}
    print(json.dumps(printed_payload, indent=2))


def unsupervised_quality(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    delta_penalty: float,
    smooth_penalty: float,
) -> dict[str, float]:
    signal = batch["signal"]
    rec = out["reconstruction"]
    residual = torch.mean((signal - rec).pow(2), dim=-1)
    energy = torch.mean(signal.pow(2), dim=-1).clamp_min(1.0e-12)
    observed_snr = 10.0 * torch.log10(energy / residual.clamp_min(1.0e-12))
    delta_rms = torch.sqrt(out["delta_if_hz"].pow(2).mean(dim=(1, 2)))
    diff = out["refined_if_hz"][..., 1:] - out["refined_if_hz"][..., :-1]
    smooth = diff.pow(2).mean(dim=(1, 2))
    score = observed_snr - float(delta_penalty) * delta_rms - float(smooth_penalty) * smooth
    return {
        "observed_snr_db": float(observed_snr.mean().detach().cpu()),
        "delta_rms_hz": float(delta_rms.mean().detach().cpu()),
        "smooth_hz2": float(smooth.mean().detach().cpu()),
        "quality_score": float(score.mean().detach().cpu()),
    }


def summarize(rows: list[dict[str, Any]], scenarios: list[str]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for scenario in [*scenarios, "aggregate"]:
        selected = rows if scenario == "aggregate" else [row for row in rows if row["scenario"] == scenario]
        if not selected:
            continue
        summary[scenario] = summarize_group(selected)
    return summary


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in (
        "default_if_mae_hz",
        "specialist_if_mae_hz",
        "gated_if_mae_hz",
        "oracle_if_mae_hz",
        "default_rec_snr_db",
        "specialist_rec_snr_db",
        "gated_rec_snr_db",
        "oracle_rec_snr_db",
        "gate_uses_specialist",
        "oracle_uses_specialist",
        "gate_matches_oracle",
    ):
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean())
    for prefix in ("default", "specialist", "gated", "oracle"):
        values = np.array([float(row[f"{prefix}_if_mae_hz"]) for row in rows], dtype=np.float64)
        out[f"{prefix}_if_mae_p90"] = float(np.percentile(values, 90))
        out[f"{prefix}_if_mae_p95"] = float(np.percentile(values, 95))
    return out


if __name__ == "__main__":
    main()
