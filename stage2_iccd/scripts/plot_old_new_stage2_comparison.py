from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ifnet_stage1.config import choose_device
from ifnet_stage1.simulation import SCENARIOS, ChirpSimulator, sim_config_from_dict
from ifnet_stage1.tf import feature_channels, log_spectrogram, stft_config_from_dict

from stage2_iccd.active_count import ActiveCountClassifier, active_count_config_from_dict, active_count_labels
from stage2_iccd.differentiable_iccd import iccd_config_from_dict
from stage2_iccd.eval_scenarios import parse_noise_types
from stage2_iccd.model import Stage2ICCDModel, stage2_model_config_from_dict
from stage2_iccd.train_stage2 import compute_loss, get_candidates, load_stage2_model_state, make_candidate_provider


DEFAULT_SCENARIOS = ("linear", "quadratic", "cubic", "near_parallel")


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
        self.provider = make_candidate_provider(self.init_cfg, device=device, seed=int(self.cfg.get("seed", 0)) + 1777)
        self.weights = self.cfg["train"].get("loss", {})

    @torch.no_grad()
    def run(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        candidate_if = get_candidates(self.provider, self.init_cfg, batch["signal"], batch["if_hz"], batch["if_hz"].shape[-1])
        candidate_if = candidate_if.clamp(self.sim_cfg.freq_min, self.sim_cfg.freq_max).detach()
        return self.model(batch["signal"], candidate_if)


class RoutedStage2:
    def __init__(self, active_checkpoint: str | Path, single_checkpoint: str | Path, multi_checkpoint: str | Path, device: torch.device):
        active_ckpt = torch.load(active_checkpoint, map_location=device, weights_only=False)
        self.active_cfg = active_ckpt["config"]
        self.stft_cfg = stft_config_from_dict(self.active_cfg["stft"])
        model_cfg = active_count_config_from_dict(active_ckpt.get("model_cfg", self.active_cfg.get("active_count")))
        self.active_model = ActiveCountClassifier(feature_channels(self.stft_cfg), model_cfg).to(device)
        self.active_model.load_state_dict(active_ckpt["model"])
        self.active_model.eval()
        self.single = Stage2Bundle(single_checkpoint, device)
        self.multi = Stage2Bundle(multi_checkpoint, device)

    @torch.no_grad()
    def run(self, batch: dict[str, Any], fs: float) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        feats, _ = log_spectrogram(batch["signal"], self.stft_cfg, fs)
        logits = self.active_model(feats)
        probs = torch.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
        label = int(active_count_labels(batch["active_mask"]).item())
        bundle = self.single if pred == 0 else self.multi
        out = bundle.run(batch)
        route = {
            "route_pred_active": float(pred + 1),
            "route_true_active": float(label + 1),
            "route_confidence": float(probs.max(dim=1).values.item()),
            "route_correct": float(pred == label),
        }
        return out, route


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-checkpoint", default="stage2_iccd/runs/all_from_separated/latest.pt")
    parser.add_argument("--active-checkpoint", default="stage2_iccd/runs/active_count_simple_near_parallel/latest.pt")
    parser.add_argument("--single-checkpoint", default="stage2_iccd/runs/simple_single_component/latest.pt")
    parser.add_argument("--multi-checkpoint", default="stage2_iccd/runs/simple_multicomponent_long/latest.pt")
    parser.add_argument("--output-dir", default="output/figures/stage2_old_new_compare")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--scenarios", nargs="*", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--active-components", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--snr-db-min", type=float, default=4.0)
    parser.add_argument("--snr-db-max", type=float, default=28.0)
    parser.add_argument("--noise-types-json", default="{white:0.75,colored:0.25,impulsive:0.0,trend:0.0}")
    args = parser.parse_args()

    device = choose_device(args.device)
    out_dir = Path(args.output_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    old = Stage2Bundle(args.old_checkpoint, device)
    new = RoutedStage2(args.active_checkpoint, args.single_checkpoint, args.multi_checkpoint, device)

    rows = []
    overview = []
    for active in args.active_components:
        for scenario in args.scenarios:
            batch, sim_cfg = make_one_sample(
                old.cfg["data"],
                scenario=scenario,
                active_components=active,
                seed=args.seed + active * 1000 + len(rows),
                device=device,
                snr_db_min=args.snr_db_min,
                snr_db_max=args.snr_db_max,
                noise_types=parse_noise_types(args.noise_types_json),
            )
            old_out = old.run(batch)
            new_out, route = new.run(batch, sim_cfg.fs)
            old_metrics = metrics_for(old_out, batch, sim_cfg.fs, old.weights)
            routed_weights = new.single.weights if int(route["route_pred_active"]) == 1 else new.multi.weights
            new_metrics = metrics_for(new_out, batch, sim_cfg.fs, routed_weights)

            name = f"{scenario}_active{active}"
            image = plot_dir / f"{name}.png"
            save_comparison_plot(
                batch=batch,
                old_if=old_out["refined_if_hz"],
                new_if=new_out["refined_if_hz"],
                image_path=image,
                title=f"{scenario} | active={active}",
                old_mae=old_metrics["if_mae_hz"],
                new_mae=new_metrics["if_mae_hz"],
                route=route,
                fs=sim_cfg.fs,
                freq_max=sim_cfg.freq_max,
            )
            row = {
                "scenario": scenario,
                "active_components": active,
                "image": str(image),
                "old_if_mae_hz": old_metrics["if_mae_hz"],
                "new_if_mae_hz": new_metrics["if_mae_hz"],
                "old_rec_snr_db": old_metrics["rec_snr_db"],
                "new_rec_snr_db": new_metrics["rec_snr_db"],
                **route,
            }
            rows.append(row)
            overview.append((scenario, active, image, old_metrics, new_metrics, route))

    save_overview(out_dir / "overview.png", overview)
    payload = {
        "old_checkpoint": args.old_checkpoint,
        "active_checkpoint": args.active_checkpoint,
        "single_checkpoint": args.single_checkpoint,
        "multi_checkpoint": args.multi_checkpoint,
        "rows": rows,
    }
    (out_dir / "comparison_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def make_one_sample(
    base_data: dict[str, Any],
    scenario: str,
    active_components: int,
    seed: int,
    device: torch.device,
    snr_db_min: float,
    snr_db_max: float,
    noise_types: dict[str, float],
):
    sim_data = dict(base_data)
    sim_data["active_components"] = int(active_components)
    sim_data["snr_db_min"] = float(snr_db_min)
    sim_data["snr_db_max"] = float(snr_db_max)
    sim_data["noise_types"] = noise_types
    sim_data["scenario_weights"] = {name: (1.0 if name == scenario else 0.0) for name in SCENARIOS}
    sim_cfg = sim_config_from_dict(sim_data)
    simulator = ChirpSimulator(sim_cfg, seed=seed)
    return simulator.generate_batch(1, device=device), sim_cfg


def metrics_for(out: dict[str, torch.Tensor], batch: dict[str, Any], fs: float, weights: dict[str, Any]) -> dict[str, float]:
    _, metrics = compute_loss(
        out,
        batch["clean"],
        batch["components"],
        batch["if_hz"],
        fs,
        weights,
        active_mask=batch.get("active_mask"),
    )
    return metrics


def save_comparison_plot(
    batch: dict[str, Any],
    old_if: torch.Tensor,
    new_if: torch.Tensor,
    image_path: Path,
    title: str,
    old_mae: float,
    new_mae: float,
    route: dict[str, float],
    fs: float,
    freq_max: float,
) -> None:
    signal = batch["signal"][0].detach().cpu()
    target = batch["if_hz"][0].detach().cpu()
    active_mask = batch["active_mask"][0].detach().cpu()
    old_aligned = align_prediction(old_if[0].detach().cpu(), target, active_mask)
    new_aligned = align_prediction(new_if[0].detach().cpu(), target, active_mask)
    spec, freq, frame_times = spectrogram_for_plot(signal, fs)
    sample_times = np.arange(target.shape[-1]) / fs

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, pred, label, mae, color in (
        (axes[0], old_aligned, "old generic Stage2", old_mae, "#4da3ff"),
        (axes[1], new_aligned, "new active-routed Stage2", new_mae, "#ffcc33"),
    ):
        ax.imshow(
            spec,
            origin="lower",
            aspect="auto",
            extent=[frame_times[0], frame_times[-1], float(freq[0]), float(freq[-1])],
            cmap="magma",
            alpha=0.82,
        )
        for idx in active_indices(active_mask):
            ax.plot(sample_times, target[idx], color="#33e06f", linewidth=1.8, label="true IF" if idx == 0 else None)
            ax.plot(sample_times, pred[idx], color=color, linewidth=1.35, linestyle="--", label="pred IF" if idx == 0 else None)
        ax.set_title(f"{label}\nIF MAE={mae:.2f} Hz")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.16, linewidth=0.6)
        ax.set_ylim(0.0, freq_max + 80.0)
    axes[0].set_ylabel("Frequency (Hz)")
    axes[1].legend(loc="upper right", fontsize=8)
    route_text = (
        f"route: pred={int(route['route_pred_active'])}, true={int(route['route_true_active'])}, "
        f"conf={route['route_confidence']:.3f}"
    )
    fig.suptitle(f"{title} | {route_text}", fontsize=12)
    fig.savefig(image_path, dpi=170)
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


def align_prediction(pred: torch.Tensor, target: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    indices = active_indices(active_mask)
    aligned = pred.clone()
    best_perm = tuple(range(pred.shape[0]))
    best_cost = float("inf")
    for perm in itertools.permutations(range(pred.shape[0])):
        cost = 0.0
        for target_idx in indices:
            cost += float((pred[perm[target_idx]] - target[target_idx]).abs().mean())
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    for target_idx in indices:
        aligned[target_idx] = pred[best_perm[target_idx]]
    return aligned


def active_indices(active_mask: torch.Tensor) -> list[int]:
    indices = torch.where(active_mask > 0.5)[0].tolist()
    return indices or [0]


def save_overview(path: Path, overview: list[tuple[str, int, Path, dict[str, float], dict[str, float], dict[str, float]]]) -> None:
    cols = 2
    rows = int(np.ceil(len(overview) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13.0, 4.0 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, item in zip(axes, overview, strict=False):
        scenario, active, image, old_metrics, new_metrics, route = item
        img = plt.imread(image)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"{scenario} active={active}: old {old_metrics['if_mae_hz']:.2f} Hz -> "
            f"new {new_metrics['if_mae_hz']:.2f} Hz | route conf {route['route_confidence']:.2f}",
            fontsize=9,
        )
    for ax in axes[len(overview) :]:
        ax.axis("off")
    fig.savefig(path, dpi=145)
    plt.close(fig)


if __name__ == "__main__":
    main()
