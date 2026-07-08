from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ifnet_stage1.config import choose_device
from ifnet_stage1.simulation import SCENARIOS, ChirpSimulator, sim_config_from_dict
from ifnet_stage1.tf import feature_channels, log_spectrogram, stft_config_from_dict

from .active_count import (
    ACTIVE_COUNT_NAMES,
    ActiveCountClassifier,
    active_count_config_from_dict,
    active_count_labels,
    active_count_metrics,
)
from .eval_scenarios import parse_noise_types


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    scenarios: list[str] | None = None,
    active_components: list[int] | None = None,
    batches: int = 16,
    batch_size: int = 8,
    data_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device = choose_device(device_name)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    stft_cfg = stft_config_from_dict(cfg["stft"])
    model_cfg = active_count_config_from_dict(ckpt.get("model_cfg", cfg.get("active_count")))
    model = ActiveCountClassifier(feature_channels(stft_cfg), model_cfg, num_classes=len(ACTIVE_COUNT_NAMES)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    selected_scenarios = scenarios or ["linear", "quadratic", "cubic"]
    selected_active = active_components or [1, 2]
    rows = []
    for active in tqdm(selected_active, desc="active-count-eval"):
        for scenario in selected_scenarios:
            sim_data = dict(cfg["data"])
            if data_overrides:
                sim_data.update(data_overrides)
            sim_data["active_components"] = int(active)
            sim_data["scenario_weights"] = {name: (1.0 if name == scenario else 0.0) for name in SCENARIOS}
            sim_cfg = sim_config_from_dict(sim_data)
            simulator = ChirpSimulator(sim_cfg, seed=int(cfg.get("seed", 0)) + 7000 + active * 97 + len(rows))
            metric_rows = []
            losses = []
            for _ in range(batches):
                batch = simulator.generate_batch(batch_size, device=device)
                feats, _ = log_spectrogram(batch["signal"], stft_cfg, sim_cfg.fs)
                labels = active_count_labels(batch["active_mask"])
                logits = model(feats)
                losses.append(F.cross_entropy(logits, labels).detach())
                metric_rows.append(active_count_metrics(logits, labels))
            row = {
                "scenario": scenario,
                "active_components": int(active),
                "loss": float(torch.stack(losses).mean().cpu()),
            }
            for key in metric_rows[0]:
                values = [item[key] for item in metric_rows]
                row[key] = float(np.nanmean(values)) if not all(np.isnan(value) for value in values) else float("nan")
            rows.append(row)

    aggregate = {"scenario": "aggregate", "active_components": -1}
    for key in rows[0]:
        if key not in {"scenario", "active_components"}:
            aggregate[key] = float(np.nanmean([row[key] for row in rows]))
    rows.append(aggregate)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint),
        "batches": batches,
        "batch_size": batch_size,
        "data_overrides": data_overrides or {},
        "rows": rows,
    }
    (out_dir / "active_count_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "active_count_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--active-components", nargs="*", type=int, default=None)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--snr-db-min", type=float, default=None)
    parser.add_argument("--snr-db-max", type=float, default=None)
    parser.add_argument("--noise-types-json", default=None)
    args = parser.parse_args()

    data_overrides: dict[str, Any] = {}
    if args.snr_db_min is not None:
        data_overrides["snr_db_min"] = args.snr_db_min
    if args.snr_db_max is not None:
        data_overrides["snr_db_max"] = args.snr_db_max
    if args.noise_types_json is not None:
        data_overrides["noise_types"] = parse_noise_types(args.noise_types_json)

    result = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        scenarios=args.scenarios,
        active_components=args.active_components,
        batches=args.batches,
        batch_size=args.batch_size,
        data_overrides=data_overrides,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
