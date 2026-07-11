from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ifnet_stage1.config import choose_device
from ifnet_stage1.simulation import SCENARIOS, ChirpSimulator, sim_config_from_dict
from ifnet_stage1.tf import feature_channels, log_spectrogram, stft_config_from_dict

from .active_count import ActiveCountClassifier, active_count_config_from_dict, active_count_labels
from .differentiable_iccd import iccd_config_from_dict
from .eval_scenarios import parse_noise_types
from .model import Stage2ICCDModel, stage2_model_config_from_dict
from .train_stage2 import (
    build_refinement_extra,
    compute_loss,
    get_candidates,
    load_stage2_model_state,
    make_candidate_provider,
)


class Stage2Bundle:
    def __init__(self, checkpoint: str | Path, device: torch.device):
        self.checkpoint = str(checkpoint)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.cfg = ckpt["config"]
        self.sim_cfg = sim_config_from_dict(self.cfg["data"])
        iccd_cfg = iccd_config_from_dict({**self.cfg["iccd"], "fs": self.sim_cfg.fs, "n_samples": self.sim_cfg.n_samples})
        model_cfg = stage2_model_config_from_dict(
            {
                **self.cfg["model"],
                "num_candidates": int(self.cfg.get("init", {}).get("num_candidates", self.cfg["model"].get("num_candidates", 2))),
                "freq_min": self.sim_cfg.freq_min,
                "freq_max": self.sim_cfg.freq_max,
            }
        )
        self.model = Stage2ICCDModel(iccd_cfg, model_cfg, num_components=self.sim_cfg.num_components).to(device)
        load_stage2_model_state(self.model, ckpt["model"])
        self.model.eval()
        self.init_cfg = self.cfg.get("init", {})
        self.provider = make_candidate_provider(self.init_cfg, device=device, seed=int(self.cfg.get("seed", 0)) + 131)
        self.weights = self.cfg["train"].get("loss", {})
        self.refinement_extra_cfg = self.cfg.get("train", {}).get("refinement_extra", {})

    @torch.no_grad()
    def run(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        candidate_if = get_candidates(self.provider, self.init_cfg, batch["signal"], batch["if_hz"], batch["if_hz"].shape[-1])
        candidate_if = candidate_if.clamp(self.sim_cfg.freq_min, self.sim_cfg.freq_max).detach()
        refinement_extra = build_refinement_extra(
            batch,
            self.model.model_cfg,
            batch["if_hz"].shape[-1],
            self.refinement_extra_cfg,
        )
        return self.model(batch["signal"], candidate_if, refinement_extra=refinement_extra)


@torch.no_grad()
def evaluate_routed(
    active_checkpoint: str | Path,
    single_checkpoint: str | Path,
    multi_checkpoint: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    scenarios: list[str] | None = None,
    active_components: list[int] | None = None,
    batches: int = 16,
    batch_size: int = 8,
    data_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    device = choose_device(device_name)
    active_ckpt = torch.load(active_checkpoint, map_location=device, weights_only=False)
    active_cfg = active_ckpt["config"]
    stft_cfg = stft_config_from_dict(active_cfg["stft"])
    active_model_cfg = active_count_config_from_dict(active_ckpt.get("model_cfg", active_cfg.get("active_count")))
    active_model = ActiveCountClassifier(feature_channels(stft_cfg), active_model_cfg).to(device)
    active_model.load_state_dict(active_ckpt["model"])
    active_model.eval()

    single = Stage2Bundle(single_checkpoint, device)
    multi = Stage2Bundle(multi_checkpoint, device)
    selected_scenarios = scenarios or ["linear", "quadratic", "cubic"]
    selected_active = active_components or [1, 2]

    rows = []
    for active in tqdm(selected_active, desc="active-routed-stage2"):
        for scenario in selected_scenarios:
            sim_data = dict(multi.cfg["data"])
            if data_overrides:
                sim_data.update(data_overrides)
            sim_data["active_components"] = int(active)
            sim_data["scenario_weights"] = {name: (1.0 if name == scenario else 0.0) for name in SCENARIOS}
            sim_cfg = sim_config_from_dict(sim_data)
            simulator = ChirpSimulator(sim_cfg, seed=int(active_cfg.get("seed", 0)) + 9000 + active * 101 + len(rows))
            metric_rows = []
            route_acc = []
            route_conf = []
            route_to_single = []
            for _ in range(batches):
                batch = simulator.generate_batch(batch_size, device=device)
                feats, _ = log_spectrogram(batch["signal"], stft_cfg, sim_cfg.fs)
                logits = active_model(feats)
                probs = torch.softmax(logits, dim=1)
                pred = probs.argmax(dim=1)
                labels = active_count_labels(batch["active_mask"])
                route_acc.append((pred == labels).float().mean().detach())
                route_conf.append(probs.max(dim=1).values.mean().detach())
                route_to_single.append((pred == 0).float().mean().detach())

                for pred_label, bundle in ((0, single), (1, multi)):
                    mask = pred == pred_label
                    if not bool(mask.any()):
                        continue
                    sub_batch = _slice_batch(batch, mask)
                    out = bundle.run(sub_batch)
                    _, metrics = compute_loss(
                        out,
                        sub_batch["clean"],
                        sub_batch["components"],
                        sub_batch["if_hz"],
                        sim_cfg.fs,
                        bundle.weights,
                        active_mask=sub_batch.get("active_mask"),
                    )
                    metrics["num_samples"] = float(mask.float().sum().detach().cpu())
                    metric_rows.append(metrics)
            row = {
                "scenario": scenario,
                "active_components": int(active),
                "route_accuracy": float(torch.stack(route_acc).mean().cpu()),
                "route_confidence": float(torch.stack(route_conf).mean().cpu()),
                "route_to_single_rate": float(torch.stack(route_to_single).mean().cpu()),
            }
            row.update(_weighted_average_metrics(metric_rows))
            rows.append(row)

    aggregate = {"scenario": "aggregate", "active_components": -1}
    for key in rows[0]:
        if key not in {"scenario", "active_components"}:
            aggregate[key] = float(np.nanmean([row[key] for row in rows]))
    rows.append(aggregate)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "active_checkpoint": str(active_checkpoint),
        "single_checkpoint": str(single_checkpoint),
        "multi_checkpoint": str(multi_checkpoint),
        "batches": batches,
        "batch_size": batch_size,
        "data_overrides": data_overrides or {},
        "rows": rows,
    }
    (out_dir / "routed_stage2_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "routed_stage2_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def _slice_batch(batch: dict[str, Any], mask: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == mask.shape[:1]:
            out[key] = value[mask]
        elif isinstance(value, list) and len(value) == mask.numel():
            keep = mask.detach().cpu().tolist()
            out[key] = [item for item, use in zip(value, keep, strict=False) if use]
        else:
            out[key] = value
    return out


def _weighted_average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    total = sum(row.get("num_samples", 1.0) for row in rows)
    keys = [key for key in rows[0] if key != "num_samples"]
    return {
        key: float(sum(row[key] * row.get("num_samples", 1.0) for row in rows) / max(total, 1.0))
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-checkpoint", required=True)
    parser.add_argument("--single-checkpoint", required=True)
    parser.add_argument("--multi-checkpoint", required=True)
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

    result = evaluate_routed(
        active_checkpoint=args.active_checkpoint,
        single_checkpoint=args.single_checkpoint,
        multi_checkpoint=args.multi_checkpoint,
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
