from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ifnet_stage1.config import choose_device
from ifnet_stage1.simulation import SCENARIOS, ChirpSimulator, sim_config_from_dict

from .active_count import active_count_labels
from .eval_scenarios import parse_noise_types
from .pipeline import P15PipelineConfig, P15Stage2Pipeline
from .train_stage2 import compute_loss, masked_permutation_l1


DEFAULT_BRANCH_SCENARIOS = ("linear", "quadratic", "cubic", "sinusoidal_fm", "crossing", "near_parallel", "local_jump", "tangent_or_overlap")


@torch.no_grad()
def evaluate_p15_pipeline(
    output_dir: str | Path,
    device_name: str = "auto",
    scenarios: list[str] | None = None,
    active_components: list[int] | None = None,
    batches: int = 12,
    batch_size: int = 4,
    use_scenario_hints: bool = True,
    data_overrides: dict[str, Any] | None = None,
    plots_per_case: int = 0,
    crossing_checkpoint: str | None = None,
) -> dict[str, Any]:
    device = choose_device(device_name)
    cfg = P15PipelineConfig(use_scenario_hints=use_scenario_hints, crossing_checkpoint=crossing_checkpoint)
    pipeline = P15Stage2Pipeline(cfg, device=device)
    selected_scenarios = scenarios or list(DEFAULT_BRANCH_SCENARIOS)
    selected_active = active_components or [1, 2]
    base_data = dict(pipeline.branches["all_expert"].cfg["data"])

    out_dir = Path(output_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if plots_per_case > 0:
        plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    overview_items = []
    for active in tqdm(selected_active, desc="p15-active"):
        for scenario in selected_scenarios:
            sim_data = dict(base_data)
            if data_overrides:
                sim_data.update(data_overrides)
            sim_data["active_components"] = int(active)
            sim_data["scenario_weights"] = {name: (1.0 if name == scenario else 0.0) for name in SCENARIOS}
            sim_cfg = sim_config_from_dict(sim_data)
            simulator = ChirpSimulator(sim_cfg, seed=20260731 + active * 1009 + len(rows) * 17)

            metric_rows = []
            route_correct = []
            route_conf = []
            second_peak_rates = []
            second_peak_ratios = []
            top2_coverage = []
            branch_counts: dict[str, int] = {}
            for batch_idx in range(batches):
                batch = simulator.generate_batch(batch_size, device=device)
                out, route = pipeline.run(
                    batch,
                    sim_cfg.fs,
                    scenario_hints=batch["scenario"] if use_scenario_hints else None,
                )
                _, metrics = compute_loss(
                    out,
                    batch["clean"],
                    batch["components"],
                    batch["if_hz"],
                    sim_cfg.fs,
                    {},
                    active_mask=batch.get("active_mask"),
                )
                if "identity_stable_if_hz" in out:
                    metrics["post_identity_if_mae_hz"] = float(
                        masked_permutation_l1(
                            out["identity_stable_if_hz"],
                            batch["if_hz"],
                            batch.get("active_mask"),
                        )
                        .detach()
                        .cpu()
                    )
                metric_rows.append(metrics)
                labels = active_count_labels(batch["active_mask"], num_classes=len(pipeline.active_names)) + 1
                route_correct.append((route["active_pred"] == labels).float().mean().detach())
                route_conf.append(route["active_confidence"].mean().detach())
                second_peak_rates.append(route["second_peak_rate"].mean().detach())
                second_peak_ratios.append(route["second_peak_ratio"].mean().detach())
                if route["candidate_top2_weights"] is not None:
                    top2_coverage.append(route["candidate_top2_weights"].sum(dim=1).mean().detach())
                for branch_name in route["branch"]:
                    branch_counts[branch_name] = branch_counts.get(branch_name, 0) + 1
                if plots_per_case > 0 and batch_idx < plots_per_case:
                    image_path = plot_dir / f"{scenario}_active{active}_sample{batch_idx}.png"
                    save_p15_plot(image_path, batch, out, route, sim_cfg.fs, sim_cfg.freq_max)
                    overview_items.append((scenario, active, image_path, metrics, dict(branch_counts)))

            row = {"scenario": scenario, "active_components": int(active)}
            for key in metric_rows[0]:
                row[key] = float(np.mean([item[key] for item in metric_rows]))
            total_routes = max(sum(branch_counts.values()), 1)
            row.update(
                {
                    "active_route_accuracy": float(torch.stack(route_correct).mean().cpu()),
                    "active_route_confidence": float(torch.stack(route_conf).mean().cpu()),
                    "second_peak_rate": float(torch.stack(second_peak_rates).mean().cpu()),
                    "second_peak_ratio": float(torch.stack(second_peak_ratios).mean().cpu()),
                    "top2_candidate_coverage": float(torch.stack(top2_coverage).mean().cpu()) if top2_coverage else float("nan"),
                    "single_rate": branch_counts.get("single", 0) / total_routes,
                    "multi_rate": branch_counts.get("multi", 0) / total_routes,
                    "local_jump_rate": branch_counts.get("local_jump", 0) / total_routes,
                    "crossing_rate": branch_counts.get("crossing", 0) / total_routes,
                    "all_expert_rate": branch_counts.get("all_expert", 0) / total_routes,
                }
            )
            rows.append(row)

    aggregate = {"scenario": "aggregate", "active_components": -1}
    for key in rows[0]:
        if key not in {"scenario", "active_components"}:
            aggregate[key] = float(np.nanmean([row[key] for row in rows]))
    rows.append(aggregate)

    if overview_items:
        save_overview(out_dir / "overview.png", overview_items)

    payload = {
        "pipeline": {
            "active_checkpoint": cfg.active_checkpoint,
            "single_checkpoint": cfg.single_checkpoint,
            "multi_checkpoint": cfg.multi_checkpoint,
            "local_jump_checkpoint": cfg.local_jump_checkpoint,
            "all_expert_checkpoint": cfg.all_expert_checkpoint,
            "crossing_checkpoint": cfg.crossing_checkpoint,
            "use_scenario_hints": use_scenario_hints,
        },
        "batches": batches,
        "batch_size": batch_size,
        "data_overrides": data_overrides or {},
        "rows": rows,
    }
    (out_dir / "p15_pipeline_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "p15_pipeline_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return payload


def save_p15_plot(
    image_path: Path,
    batch: dict[str, Any],
    out: dict[str, torch.Tensor],
    route: dict[str, Any],
    fs: float,
    freq_max: float,
) -> None:
    signal = batch["signal"][0].detach().cpu()
    target = batch["if_hz"][0].detach().cpu()
    scenario = str(batch["scenario"][0])
    pred_key = "refined_if_hz" if scenario == "crossing" and "refined_if_hz" in out else "identity_stable_if_hz"
    pred_label = "refined" if pred_key == "refined_if_hz" else "stable"
    pred = out[pred_key][0].detach().cpu()
    active_mask = batch["active_mask"][0].detach().cpu()
    spec, freq, frame_times = spectrogram_for_plot(signal, fs)
    sample_times = np.arange(signal.shape[-1]) / fs
    fig, ax = plt.subplots(figsize=(7.0, 4.3), constrained_layout=True)
    ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[frame_times[0], frame_times[-1], float(freq[0]), float(freq[-1])],
        cmap="magma",
        alpha=0.84,
    )
    for idx in torch.where(active_mask > 0.5)[0].tolist():
        ax.plot(sample_times, target[idx], color="#35e56a", linewidth=1.5, label="true IF" if idx == 0 else None)
        ax.plot(sample_times, pred[idx], color="#ffcc33", linestyle="--", linewidth=1.35, label="P2.5 IF" if idx == 0 else None)
    ax.set_ylim(0.0, freq_max + 75.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(
        f"{scenario} | active={int(active_mask.sum().item())} | "
        f"branch={route['branch'][0]} | conf={float(route['active_confidence'][0]):.3f} | {pred_label}",
        fontsize=10,
    )
    ax.grid(alpha=0.16, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(image_path, dpi=165)
    plt.close(fig)


def spectrogram_for_plot(signal: torch.Tensor, fs: float):
    n_fft = 256
    hop = 4
    win = 128
    window = torch.hann_window(win)
    spec = torch.stft(signal, n_fft=n_fft, hop_length=hop, win_length=win, window=window, center=True, return_complex=True)
    mag = torch.log(spec.abs() + 1.0e-6)
    mag = (mag - mag.mean()) / mag.std().clamp_min(1.0e-5)
    freq = torch.linspace(0.0, fs / 2.0, n_fft // 2 + 1)
    frame_times = torch.arange(mag.shape[-1]) * hop / fs
    return mag.numpy(), freq.numpy(), frame_times.numpy()


def save_overview(path: Path, items: list[tuple[str, int, Path, dict[str, float], dict[str, int]]]) -> None:
    cols = 2
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13.2, 4.0 * rows), constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    for ax, (scenario, active, image_path, metrics, branch_counts) in zip(axes_flat, items, strict=False):
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"{scenario} active={active} | IF {metrics['if_mae_hz']:.2f} Hz | branches={branch_counts}",
            fontsize=9,
        )
    for ax in axes_flat[len(items) :]:
        ax.axis("off")
    fig.savefig(path, dpi=145)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="stage2_iccd/runs/p15_pipeline/eval_default")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--active-components", nargs="*", type=int, default=None)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-scenario-hints", action="store_true")
    parser.add_argument("--plots-per-case", type=int, default=0)
    parser.add_argument("--crossing-checkpoint", default=None)
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

    result = evaluate_p15_pipeline(
        output_dir=args.output_dir,
        device_name=args.device,
        scenarios=args.scenarios,
        active_components=args.active_components,
        batches=args.batches,
        batch_size=args.batch_size,
        use_scenario_hints=not args.no_scenario_hints,
        data_overrides=data_overrides,
        plots_per_case=args.plots_per_case,
        crossing_checkpoint=args.crossing_checkpoint,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
