from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from ifnet_stage1.config import choose_device
from ifnet_stage1.simulation import SCENARIOS, ChirpSimulator, sim_config_from_dict

from .differentiable_iccd import iccd_config_from_dict
from .model import Stage2ICCDModel, stage2_model_config_from_dict
from .train_stage2 import compute_loss, get_candidates, load_stage2_model_state, make_candidate_provider


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    scenarios: list[str] | None = None,
    batches: int = 16,
    batch_size: int = 4,
    data_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device = choose_device(device_name)
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    base_sim_cfg = sim_config_from_dict(cfg["data"])
    iccd_cfg = iccd_config_from_dict({**cfg["iccd"], "fs": base_sim_cfg.fs, "n_samples": base_sim_cfg.n_samples})
    model_cfg = stage2_model_config_from_dict(
        {
            **cfg["model"],
            "num_candidates": int(cfg.get("init", {}).get("num_candidates", cfg["model"].get("num_candidates", 2))),
            "freq_min": base_sim_cfg.freq_min,
            "freq_max": base_sim_cfg.freq_max,
        }
    )
    model = Stage2ICCDModel(iccd_cfg, model_cfg, num_components=base_sim_cfg.num_components).to(device)
    load_stage2_model_state(model, ckpt["model"])
    model.eval()

    init_cfg = cfg.get("init", {})
    provider = make_candidate_provider(init_cfg, device=device, seed=int(cfg.get("seed", 0)) + 991)
    weights = cfg["train"].get("loss", {})
    selected_scenarios = scenarios or list(SCENARIOS)

    rows = []
    for scenario in tqdm(selected_scenarios, desc="stage2-eval"):
        sim_data = dict(cfg["data"])
        if data_overrides:
            sim_data.update(data_overrides)
        sim_data["scenario_weights"] = {name: (1.0 if name == scenario else 0.0) for name in SCENARIOS}
        sim_cfg = sim_config_from_dict(sim_data)
        simulator = ChirpSimulator(sim_cfg, seed=int(cfg.get("seed", 0)) + 1009 + len(rows))
        metric_rows = []
        for _ in range(batches):
            batch = simulator.generate_batch(batch_size, device=device)
            candidate_if = get_candidates(provider, init_cfg, batch["signal"], batch["if_hz"], sim_cfg.n_samples)
            candidate_if = candidate_if.clamp(sim_cfg.freq_min, sim_cfg.freq_max).detach()
            out = model(batch["signal"], candidate_if)
            _, metrics = compute_loss(
                out,
                batch["clean"],
                batch["components"],
                batch["if_hz"],
                sim_cfg.fs,
                weights,
                active_mask=batch.get("active_mask"),
            )
            metric_rows.append(metrics)
        row = {"scenario": scenario}
        for key in metric_rows[0]:
            row[key] = float(sum(item[key] for item in metric_rows) / len(metric_rows))
        rows.append(row)

    aggregate = {"scenario": "aggregate"}
    for key in rows[0]:
        if key != "scenario":
            aggregate[key] = float(sum(row[key] for row in rows) / len(rows))
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
    (out_dir / "scenario_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "scenario_metrics.csv").open("w", newline="", encoding="utf-8") as f:
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
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--snr-db-min", type=float, default=None)
    parser.add_argument("--snr-db-max", type=float, default=None)
    parser.add_argument("--active-components", type=int, default=None)
    parser.add_argument(
        "--noise-types-json",
        default=None,
        help='Optional JSON object, for example {"white":0.75,"colored":0.25,"impulsive":0.0,"trend":0.0}.',
    )
    args = parser.parse_args()
    data_overrides: dict[str, Any] = {}
    if args.snr_db_min is not None:
        data_overrides["snr_db_min"] = args.snr_db_min
    if args.snr_db_max is not None:
        data_overrides["snr_db_max"] = args.snr_db_max
    if args.active_components is not None:
        data_overrides["active_components"] = args.active_components
    if args.noise_types_json is not None:
        data_overrides["noise_types"] = parse_noise_types(args.noise_types_json)

    result = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        scenarios=args.scenarios,
        batches=args.batches,
        batch_size=args.batch_size,
        data_overrides=data_overrides,
    )
    print(json.dumps(result, indent=2))


def parse_noise_types(value: str) -> dict[str, float]:
    try:
        parsed = json.loads(value)
        return {str(key): float(item) for key, item in parsed.items()}
    except json.JSONDecodeError:
        items = value.strip().strip("{}")
        result: dict[str, float] = {}
        for item in items.split(","):
            if not item.strip():
                continue
            key, raw = item.split(":", 1)
            result[key.strip().strip("'\"")] = float(raw)
        return result


if __name__ == "__main__":
    main()
