from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ifnet_stage1.config import choose_device

from .pipeline import P15PipelineConfig, P15Stage2Pipeline


def infer_signal(
    input_npy: str | Path,
    output_dir: str | Path,
    fs: float = 1024.0,
    device_name: str = "auto",
    scenario_hint: str | None = None,
    crossing_checkpoint: str | None = None,
    target_samples: int | None = 1024,
) -> dict:
    device = choose_device(device_name)
    signal_np = np.load(input_npy).astype(np.float32)
    if signal_np.ndim != 1:
        raise ValueError(f"Expected a 1D signal array, got shape {signal_np.shape}.")
    original_samples = int(signal_np.shape[0])
    signal_np = prepare_signal(signal_np, target_samples=target_samples)
    signal = torch.from_numpy(signal_np[None]).to(device)
    n_samples = signal.shape[-1]
    dummy_if = torch.zeros((1, 2, n_samples), dtype=signal.dtype, device=device)
    dummy_components = torch.zeros((1, 2, n_samples), dtype=signal.dtype, device=device)
    batch = {
        "signal": signal,
        "if_hz": dummy_if,
        "clean": signal,
        "components": dummy_components,
        "active_mask": torch.ones((1, 2), dtype=signal.dtype, device=device),
        "scenario": [scenario_hint or "unknown"],
    }
    pipeline = P15Stage2Pipeline(
        P15PipelineConfig(
            use_scenario_hints=scenario_hint is not None,
            crossing_checkpoint=crossing_checkpoint,
        ),
        device=device,
    )
    out, route = pipeline.run(batch, fs, scenario_hints=[scenario_hint] if scenario_hint else None)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    refined_if = out["refined_if_hz"][0].detach().cpu().numpy()
    identity_if = out["identity_stable_if_hz"][0].detach().cpu().numpy()
    plot_if_source = "refined_if_hz" if scenario_hint == "crossing" else "identity_stable_if_hz"
    if_hz = refined_if if plot_if_source == "refined_if_hz" else identity_if
    active_pred = int(route["active_pred"][0].detach().cpu())
    np.save(out_dir / "p15_if_hz.npy", if_hz.astype(np.float32))
    np.save(out_dir / "p15_active_if_hz.npy", if_hz[:active_pred].astype(np.float32))
    np.save(out_dir / "p15_refined_if_hz.npy", refined_if.astype(np.float32))
    np.save(out_dir / "p15_identity_stable_if_hz.npy", identity_if.astype(np.float32))
    plot_path = out_dir / "p15_inference.png"
    save_inference_plot(plot_path, signal_np, if_hz, fs, active_count=active_pred, title=f"P2.5 inference ({plot_if_source})")
    payload = {
        "input_npy": str(input_npy),
        "fs": float(fs),
        "original_samples": original_samples,
        "model_samples": int(signal_np.shape[0]),
        "resampled": original_samples != int(signal_np.shape[0]),
        "target_samples": int(target_samples) if target_samples is not None else None,
        "crossing_checkpoint": str(crossing_checkpoint) if crossing_checkpoint else None,
        "plot_if_source": plot_if_source,
        "branch": route["branch"][0],
        "active_pred": active_pred,
        "active_confidence": float(route["active_confidence"][0].detach().cpu()),
        "active_probs": [float(v) for v in route["active_probs"][0].detach().cpu()],
        "candidate_top2_weights": [float(v) for v in route["candidate_top2_weights"][0].detach().cpu()],
        "candidate_top2_indices": [int(v) for v in route["candidate_top2_indices"][0].detach().cpu()],
        "if_path": str(out_dir / "p15_if_hz.npy"),
        "active_if_path": str(out_dir / "p15_active_if_hz.npy"),
        "refined_if_path": str(out_dir / "p15_refined_if_hz.npy"),
        "identity_stable_if_path": str(out_dir / "p15_identity_stable_if_hz.npy"),
        "plot_path": str(plot_path),
    }
    (out_dir / "p15_inference.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def prepare_signal(signal_np: np.ndarray, target_samples: int | None = 1024) -> np.ndarray:
    signal_np = np.asarray(signal_np, dtype=np.float32)
    if target_samples is not None and int(target_samples) > 0 and signal_np.shape[0] != int(target_samples):
        src = np.linspace(0.0, 1.0, signal_np.shape[0], dtype=np.float32)
        dst = np.linspace(0.0, 1.0, int(target_samples), dtype=np.float32)
        signal_np = np.interp(dst, src, signal_np).astype(np.float32)
    return ((signal_np - signal_np.mean()) / max(float(signal_np.std()), 1.0e-6)).astype(np.float32)


def save_inference_plot(
    path: Path,
    signal: np.ndarray,
    if_hz: np.ndarray,
    fs: float,
    active_count: int,
    title: str = "P2.5 pipeline inference",
) -> None:
    spec, freq, frame_times = spectrogram_for_plot(signal, fs)
    t = np.arange(signal.shape[-1]) / fs
    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[frame_times[0], frame_times[-1], float(freq[0]), float(freq[-1])],
        cmap="magma",
        alpha=0.86,
    )
    for idx, curve in enumerate(if_hz[:active_count]):
        ax.plot(t, curve, linewidth=1.45, linestyle="--", label=f"P1.5 IF {idx + 1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    ax.grid(alpha=0.16, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=165)
    plt.close(fig)


def spectrogram_for_plot(signal: np.ndarray, fs: float):
    tensor = torch.from_numpy(signal.astype(np.float32))
    n_fft = 256
    hop = 4
    win = 128
    window = torch.hann_window(win)
    spec = torch.stft(tensor, n_fft=n_fft, hop_length=hop, win_length=win, window=window, center=True, return_complex=True)
    mag = torch.log(spec.abs() + 1.0e-6)
    mag = (mag - mag.mean()) / mag.std().clamp_min(1.0e-5)
    freq = torch.linspace(0.0, fs / 2.0, n_fft // 2 + 1)
    frame_times = torch.arange(mag.shape[-1]) * hop / fs
    return mag.numpy(), freq.numpy(), frame_times.numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npy", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fs", type=float, default=1024.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenario-hint", default=None)
    parser.add_argument("--crossing-checkpoint", default="stage2_iccd/runs/crossing_first_candidate_p25/latest.pt")
    parser.add_argument("--target-samples", type=int, default=1024)
    args = parser.parse_args()
    result = infer_signal(
        input_npy=args.input_npy,
        output_dir=args.output_dir,
        fs=args.fs,
        device_name=args.device,
        scenario_hint=args.scenario_hint,
        crossing_checkpoint=args.crossing_checkpoint,
        target_samples=args.target_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
