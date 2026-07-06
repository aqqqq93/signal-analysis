from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import load_config
from .losses import pairwise_ridge_nll, permutation_l1, second_difference_smoothness
from .model import IFNetUNet, model_config_from_dict, soft_argmax_if
from .simulation import SCENARIOS, ChirpSimulator, sim_config_from_dict
from .tf import feature_channels, log_spectrogram, sample_if_to_frames, stft_config_from_dict


def main() -> None:
    cfg_path = Path("ifnet_stage1/configs/default.yaml")
    cfg = load_config(cfg_path)
    cfg["data"]["n_samples"] = 512
    cfg["stft"]["target_n_fft"] = 128
    cfg["stft"]["scales"] = [
        {"n_fft": 64, "win_length": 32},
        {"n_fft": 128, "win_length": 64},
    ]
    cfg["stft"]["hop_length"] = 4
    cfg["model"]["base_channels"] = 8
    cfg["model"]["depth"] = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim_cfg = sim_config_from_dict(cfg["data"])
    stft_cfg = stft_config_from_dict(cfg["stft"])
    model_cfg = model_config_from_dict(cfg["model"])
    simulator = ChirpSimulator(sim_cfg, seed=123)
    model = IFNetUNet(feature_channels(stft_cfg), sim_cfg.num_components, model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    batch = simulator.generate_batch(len(SCENARIOS), device=device, scenarios=SCENARIOS)
    feats, freq_grid = log_spectrogram(batch["signal"], stft_cfg, sim_cfg.fs)
    logits = model(feats)
    target_if = sample_if_to_frames(batch["if_hz"], logits.shape[-1], stft_cfg.hop_length)
    pred_if, _ = soft_argmax_if(logits, freq_grid, model_cfg.temperature)
    loss = (
        pairwise_ridge_nll(logits, target_if, freq_grid, sigma_hz=8.0)
        + 0.25 * permutation_l1(pred_if, target_if) / sim_cfg.fs
        + 0.01 * second_difference_smoothness(pred_if) / (sim_cfg.fs**2)
    )
    if not torch.isfinite(loss):
        raise RuntimeError("Smoke loss is not finite.")

    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        logits = model(feats)
        pred_if, _ = soft_argmax_if(logits, freq_grid, model_cfg.temperature)
        loss = (
            pairwise_ridge_nll(logits, target_if, freq_grid, sigma_hz=8.0)
            + 0.25 * permutation_l1(pred_if, target_if) / sim_cfg.fs
            + 0.01 * second_difference_smoothness(pred_if) / (sim_cfg.fs**2)
        )

    print(
        json.dumps(
            {
                "ok": True,
                "device": str(device),
                "scenarios": batch["scenario"],
                "features": list(feats.shape),
                "logits": list(logits.shape),
                "if_mae_hz": float(permutation_l1(pred_if, target_if).detach().cpu()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
