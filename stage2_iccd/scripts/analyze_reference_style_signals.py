from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ifnet_stage1.config import choose_device

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_old_new_stage2_comparison import RoutedStage2


TEMPLATE_NAMES = (
    "image1_local_jump_like",
    "image2_four_component_wavy",
    "image3_cross_tangent_three",
    "image4_shutdown_decay",
    "image5_two_component_crossing",
    "image6_multiband_transient",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/figures/reference_style_stage2")
    parser.add_argument("--active-checkpoint", default="stage2_iccd/runs/active_count_simple_near_parallel/latest.pt")
    parser.add_argument("--single-checkpoint", default="stage2_iccd/runs/simple_single_component/latest.pt")
    parser.add_argument("--multi-checkpoint", default="stage2_iccd/runs/simple_multicomponent_long/latest.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--templates", nargs="*", default=list(TEMPLATE_NAMES))
    parser.add_argument("--fs", type=float, default=1024.0)
    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    signal_dir = out_dir / "signals"
    plot_dir = out_dir / "plots"
    signal_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    model = RoutedStage2(args.active_checkpoint, args.single_checkpoint, args.multi_checkpoint, device)
    rng = np.random.default_rng(args.seed)

    rows = []
    overview_items = []
    for template_name in args.templates:
        if template_name not in TEMPLATE_NAMES:
            raise ValueError(f"Unknown template: {template_name}. Available: {TEMPLATE_NAMES}")
        sample = make_template_signal(template_name, args.fs, args.n_samples, args.snr_db, rng)
        signal_np = sample["signal"].astype(np.float32)
        np.save(signal_dir / f"{template_name}.npy", signal_np)
        np.savez(
            signal_dir / f"{template_name}_truth.npz",
            signal=signal_np,
            clean=sample["clean"].astype(np.float32),
            components=sample["components"].astype(np.float32),
            if_hz=sample["if_hz"].astype(np.float32),
            amplitude=sample["amplitude"].astype(np.float32),
            active_mask=sample["active_mask"].astype(np.float32),
            fs=np.array([args.fs], dtype=np.float32),
        )
        batch = make_model_batch(sample, device)
        out, route = model.run(batch, args.fs)
        pred = out["refined_if_hz"][0].detach().cpu().numpy()
        initial = out["initial_if_hz"][0].detach().cpu().numpy()
        image_path = plot_dir / f"{template_name}.png"
        save_analysis_plot(
            image_path=image_path,
            signal=signal_np,
            true_if=sample["if_hz"],
            pred_if=pred,
            initial_if=initial,
            fs=args.fs,
            freq_max=430.0,
            title=template_name,
            route=route,
        )
        row = {
            "template": template_name,
            "components_in_signal": int(sample["if_hz"].shape[0]),
            "model_output_components": int(pred.shape[0]),
            "signal_path": str(signal_dir / f"{template_name}.npy"),
            "truth_path": str(signal_dir / f"{template_name}_truth.npz"),
            "plot_path": str(image_path),
            **route,
        }
        rows.append(row)
        overview_items.append((template_name, image_path, row))

    save_overview(out_dir / "overview.png", overview_items)
    payload = {
        "note": (
            "Templates are time-domain synthetic signals shaped like the provided STFT examples. "
            "Frequencies are scaled into the current trained model band, roughly 35-430 Hz. "
            "The current routed Stage2 model outputs at most two components."
        ),
        "active_checkpoint": args.active_checkpoint,
        "single_checkpoint": args.single_checkpoint,
        "multi_checkpoint": args.multi_checkpoint,
        "rows": rows,
    }
    (out_dir / "reference_style_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def make_model_batch(sample: dict[str, np.ndarray], device: torch.device) -> dict[str, Any]:
    return {
        "signal": torch.from_numpy(sample["signal"][None].astype(np.float32)).to(device),
        "clean": torch.from_numpy(sample["clean"][None].astype(np.float32)).to(device),
        "components": torch.from_numpy(sample["components"][:2][None].astype(np.float32)).to(device),
        "if_hz": torch.from_numpy(sample["if_hz"][:2][None].astype(np.float32)).to(device),
        "active_mask": torch.ones((1, min(2, sample["if_hz"].shape[0])), dtype=torch.float32, device=device),
    }


def make_template_signal(name: str, fs: float, n_samples: int, snr_db: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    u = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)
    if name == "image1_local_jump_like":
        curves = [
            265 - 115 * sigmoid((u - 0.30) / 0.035) + 210 * sigmoid((u - 0.43) / 0.012)
            - 105 * sigmoid((u - 0.49) / 0.035)
            + 115 * np.clip((u - 0.68) / 0.32, 0.0, 1.0)
        ]
        amps = [0.9 + 0.7 * gaussian(u, 0.43, 0.045)]
    elif name == "image2_four_component_wavy":
        base = 45 + 35 * u + 16 * np.sin(2 * np.pi * (1.55 * u - 0.18))
        curves = [
            38 + 18 * u + 4 * np.sin(2 * np.pi * (1.0 * u + 0.15)),
            base + 45,
            base + 118 + 8 * np.sin(2 * np.pi * (0.7 * u + 0.30)),
            base + 195 + 12 * np.sin(2 * np.pi * (0.9 * u + 0.55)),
        ]
        amps = [0.45, 0.85, 0.72, 0.58]
    elif name == "image3_cross_tangent_three":
        c1 = 70 + 110 * u + 60 * sigmoid((u - 0.43) / 0.04) - 55 * sigmoid((u - 0.55) / 0.04)
        c2 = 310 - 185 * u + 15 * np.sin(2 * np.pi * 0.8 * u)
        c3 = 145 + 250 * u**2 - 75 * gaussian(u, 0.58, 0.10)
        curves = [c1, c2, c3]
        amps = [0.88 + 0.55 * gaussian(u, 0.48, 0.05), 0.75, 0.68 + 0.30 * gaussian(u, 0.70, 0.06)]
    elif name == "image4_shutdown_decay":
        curves = [
            46 + 235 * np.exp(-2.3 * u),
            72 + 190 * np.exp(-2.1 * u),
            96 + 165 * np.exp(-2.0 * u),
            120 + 140 * np.exp(-1.9 * u),
        ]
        decay = np.exp(-0.85 * u)
        amps = [0.95 * decay, 0.55 * decay, 0.43 * decay, 0.35 * decay]
    elif name == "image5_two_component_crossing":
        curves = [
            380 - 260 * u + 140 * (u - 0.5) ** 2,
            72 + 360 * u - 150 * u * (1 - u),
        ]
        amps = [0.90, 0.85]
    elif name == "image6_multiband_transient":
        burst = gaussian(u, 0.42, 0.06)
        curves = [
            62 + 10 * np.sin(2 * np.pi * 0.55 * u),
            140 + 8 * np.sin(2 * np.pi * 0.45 * u) + 45 * burst,
            230 + 7 * np.sin(2 * np.pi * 0.50 * u) + 80 * burst,
            325 + 10 * np.sin(2 * np.pi * 0.50 * u) + 65 * burst,
        ]
        amps = [0.45, 0.60 + 0.25 * burst, 0.65 + 0.55 * burst, 0.55 + 0.45 * burst]
    else:
        raise ValueError(name)

    if_hz = np.stack([np.asarray(curve, dtype=np.float32) for curve in curves], axis=0)
    if_hz = np.clip(if_hz, 35.0, 430.0)
    amp_arr = np.stack([np.broadcast_to(np.asarray(amp, dtype=np.float32), (n_samples,)) for amp in amps], axis=0)
    phase0 = rng.uniform(0.0, 2 * np.pi, size=(if_hz.shape[0], 1)).astype(np.float32)
    phase = 2 * np.pi * np.cumsum(if_hz / float(fs), axis=-1) + phase0
    components = amp_arr * np.cos(phase)
    clean = components.sum(axis=0)
    noise = rng.normal(size=clean.shape).astype(np.float32)
    sig_power = float(np.mean(clean**2) + 1.0e-12)
    noise_power = float(np.mean(noise**2) + 1.0e-12)
    noise = noise * np.sqrt(sig_power / (10 ** (snr_db / 10.0)) / noise_power)
    signal = clean + noise
    scale = float(np.std(signal) + 1.0e-6)
    return {
        "signal": (signal / scale).astype(np.float32),
        "clean": (clean / scale).astype(np.float32),
        "components": (components / scale).astype(np.float32),
        "if_hz": if_hz.astype(np.float32),
        "amplitude": (amp_arr / scale).astype(np.float32),
        "active_mask": np.ones(if_hz.shape[0], dtype=np.float32),
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gaussian(u: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((u - center) / width) ** 2)


def save_analysis_plot(
    image_path: Path,
    signal: np.ndarray,
    true_if: np.ndarray,
    pred_if: np.ndarray,
    initial_if: np.ndarray,
    fs: float,
    freq_max: float,
    title: str,
    route: dict[str, float],
) -> None:
    spec, freq, frame_times = spectrogram_for_plot(signal, fs)
    t = np.arange(signal.shape[-1]) / fs
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharex=True, sharey=True, constrained_layout=True)
    for ax, show_pred in ((axes[0], False), (axes[1], True)):
        ax.imshow(
            spec,
            origin="lower",
            aspect="auto",
            extent=[frame_times[0], frame_times[-1], float(freq[0]), float(freq[-1])],
            cmap="magma",
            alpha=0.84,
        )
        for idx, curve in enumerate(true_if):
            ax.plot(t, curve, color="#39e66f", linewidth=1.3, alpha=0.80, label="template IF" if idx == 0 else None)
        if show_pred:
            for idx, curve in enumerate(initial_if):
                ax.plot(t, curve, color="#80bfff", linewidth=1.0, linestyle=":", alpha=0.90, label="initial IF" if idx == 0 else None)
            for idx, curve in enumerate(pred_if):
                ax.plot(t, curve, color="#ffcc33", linewidth=1.55, linestyle="--", label="Stage2 IF" if idx == 0 else None)
        ax.set_ylim(0.0, freq_max + 55.0)
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.16, linewidth=0.6)
    axes[0].set_title("synthesized signal STFT + template IF")
    axes[1].set_title("model output on synthesized signal")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"{title} | route pred={int(route['route_pred_active'])}, conf={route['route_confidence']:.3f}",
        fontsize=11,
    )
    fig.savefig(image_path, dpi=170)
    plt.close(fig)


def spectrogram_for_plot(signal: np.ndarray, fs: float):
    tensor = torch.from_numpy(signal.astype(np.float32))
    n_fft = 256
    hop = 4
    win = 128
    window = torch.hann_window(win)
    spec = torch.stft(tensor, n_fft=n_fft, hop_length=hop, win_length=win, window=window, center=True, return_complex=True)
    mag = spec.abs().numpy()
    log_mag = np.log(mag + 1.0e-5)
    lo, hi = np.percentile(log_mag, [4, 99.6])
    image = np.clip((log_mag - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)
    freqs = np.linspace(0.0, fs / 2.0, n_fft // 2 + 1)
    times = np.arange(image.shape[-1]) * hop / fs
    return image, freqs, times


def save_overview(path: Path, items: list[tuple[str, Path, dict[str, Any]]]) -> None:
    cols = 2
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13.2, 4.9 * rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).reshape(-1)
    for ax, (name, image_path, row) in zip(axes_flat, items, strict=False):
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.set_title(f"{name} | comps={row['components_in_signal']} -> model {row['model_output_components']}")
        ax.axis("off")
    for ax in axes_flat[len(items) :]:
        ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
