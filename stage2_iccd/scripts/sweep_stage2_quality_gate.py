from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--delta-penalties", nargs="*", type=float, default=[0.0, 0.005, 0.01, 0.015, 0.02, 0.03])
    parser.add_argument("--smooth-penalties", nargs="*", type=float, default=[0.0, 0.000001, 0.000002, 0.000005])
    parser.add_argument("--margins", nargs="*", type=float, default=[-0.08, -0.04, 0.0, 0.03, 0.05, 0.08, 0.12])
    args = parser.parse_args()

    rows = read_rows([Path(path) for path in args.csv])
    default_summary = summarize_choice(rows, np.zeros(len(rows), dtype=bool))
    specialist_summary = summarize_choice(rows, np.ones(len(rows), dtype=bool))
    oracle_mask = np.array([row["specialist_if_mae_hz"] < row["default_if_mae_hz"] for row in rows], dtype=bool)
    oracle_summary = summarize_choice(rows, oracle_mask)

    candidates: list[dict[str, Any]] = []
    for delta_penalty in args.delta_penalties:
        for smooth_penalty in args.smooth_penalties:
            default_score = score(rows, "default", delta_penalty, smooth_penalty)
            specialist_score = score(rows, "specialist", delta_penalty, smooth_penalty)
            for margin in args.margins:
                use_specialist = specialist_score > default_score + margin
                item = summarize_choice(rows, use_specialist)
                item.update(
                    {
                        "delta_penalty": float(delta_penalty),
                        "smooth_penalty": float(smooth_penalty),
                        "margin": float(margin),
                    }
                )
                candidates.append(item)

    top_by_mean = sorted(candidates, key=lambda item: item["if_mae_mean"])[: args.top_k]
    top_by_p95 = sorted(candidates, key=lambda item: item["if_mae_p95"])[: args.top_k]
    top_by_oracle_match = sorted(candidates, key=lambda item: item["oracle_match_rate"], reverse=True)[: args.top_k]
    payload = {
        "inputs": args.csv,
        "num_rows": len(rows),
        "baselines": {
            "default": default_summary,
            "specialist": specialist_summary,
            "oracle": oracle_summary,
        },
        "top_by_mean": top_by_mean,
        "top_by_p95": top_by_p95,
        "top_by_oracle_match": top_by_oracle_match,
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def read_rows(paths: list[Path]) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                row: dict[str, float | str] = {"source": str(path), "scenario": raw["scenario"]}
                for key, value in raw.items():
                    if key in {"scenario"}:
                        continue
                    row[key] = float(value)
                rows.append(row)
    return rows


def score(rows: list[dict[str, float | str]], prefix: str, delta_penalty: float, smooth_penalty: float) -> np.ndarray:
    obs = np.array([float(row[f"{prefix}_observed_snr_db"]) for row in rows], dtype=np.float64)
    delta = np.array([float(row[f"{prefix}_delta_rms_hz"]) for row in rows], dtype=np.float64)
    smooth = np.array([float(row[f"{prefix}_smooth_hz2"]) for row in rows], dtype=np.float64)
    return obs - float(delta_penalty) * delta - float(smooth_penalty) * smooth


def summarize_choice(rows: list[dict[str, float | str]], use_specialist: np.ndarray) -> dict[str, float]:
    default_mae = np.array([float(row["default_if_mae_hz"]) for row in rows], dtype=np.float64)
    specialist_mae = np.array([float(row["specialist_if_mae_hz"]) for row in rows], dtype=np.float64)
    chosen_mae = np.where(use_specialist, specialist_mae, default_mae)
    oracle_mask = specialist_mae < default_mae
    return {
        "if_mae_mean": float(chosen_mae.mean()),
        "if_mae_p90": float(np.percentile(chosen_mae, 90)),
        "if_mae_p95": float(np.percentile(chosen_mae, 95)),
        "uses_specialist_rate": float(use_specialist.mean()),
        "oracle_specialist_rate": float(oracle_mask.mean()),
        "oracle_match_rate": float((use_specialist == oracle_mask).mean()),
    }


if __name__ == "__main__":
    main()
