from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ifnet_stage1.config import choose_device
from ifnet_stage1.simulation import ChirpSimulator, sim_config_from_dict
from ifnet_stage1.tf import log_spectrogram, stft_config_from_dict


def stft_domain_summary(
    npy_dir: str | Path,
    reference_config: dict[str, Any],
    output_json: str | Path,
    device_name: str = "auto",
    max_files: int = 64,
    reference_batches: int = 8,
    batch_size: int = 8,
) -> dict[str, Any]:
    device = choose_device(device_name)
    stft_cfg = stft_config_from_dict(reference_config.get("stft", _default_stft_config()))
    sim_cfg = sim_config_from_dict(reference_config["data"])
    files = sorted(Path(npy_dir).glob("*.npy"))[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No .npy files found in {npy_dir}.")
    target_features = []
    for path in files:
        signal = np.load(path).astype(np.float32)
        if signal.ndim != 1:
            continue
        signal = (signal - signal.mean()) / max(float(signal.std()), 1.0e-6)
        tensor = torch.from_numpy(signal[None]).to(device)
        feats, _ = log_spectrogram(tensor, stft_cfg, sim_cfg.fs)
        target_features.append(_stats(feats))
    if not target_features:
        raise ValueError("No valid 1D .npy signals were found.")
    target = torch.cat(target_features, dim=0).mean(dim=0)

    simulator = ChirpSimulator(sim_cfg, seed=int(reference_config.get("seed", 0)) + 707)
    ref_features = []
    for _ in range(reference_batches):
        batch = simulator.generate_batch(batch_size, device=device)
        feats, _ = log_spectrogram(batch["signal"], stft_cfg, sim_cfg.fs)
        ref_features.append(_stats(feats))
    reference = torch.cat(ref_features, dim=0).mean(dim=0)
    diff = target - reference
    payload = {
        "npy_dir": str(npy_dir),
        "num_files": len(files),
        "feature_l2": float(diff.pow(2).sum().sqrt().detach().cpu()),
        "feature_l1": float(diff.abs().mean().detach().cpu()),
        "reference_mean": [float(v) for v in reference.flatten().detach().cpu()],
        "target_mean": [float(v) for v in target.flatten().detach().cpu()],
        "note": "This is a P2 domain-gap diagnostic. It does not update Stage1; use it before Stage2-only fine-tuning.",
    }
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _stats(features: torch.Tensor) -> torch.Tensor:
    flat = features.flatten(2)
    mean = flat.mean(dim=-1)
    std = flat.std(dim=-1)
    q90 = flat.quantile(0.90, dim=-1)
    return torch.cat([mean, std, q90], dim=1)


def _default_stft_config() -> dict[str, Any]:
    return {
        "hop_length": 4,
        "target_n_fft": 256,
        "log_eps": 1.0e-6,
        "scales": [
            {"n_fft": 128, "win_length": 64},
            {"n_fft": 256, "win_length": 128},
            {"n_fft": 512, "win_length": 256},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy-dir", required=True)
    parser.add_argument("--reference-config", default="stage2_iccd/runs/active_count_simple_near_parallel/config.json")
    parser.add_argument("--output-json", default="stage2_iccd/runs/p2_domain/domain_summary.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-files", type=int, default=64)
    args = parser.parse_args()
    cfg = json.loads(Path(args.reference_config).read_text(encoding="utf-8"))
    result = stft_domain_summary(
        args.npy_dir,
        cfg,
        args.output_json,
        device_name=args.device,
        max_files=args.max_files,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
