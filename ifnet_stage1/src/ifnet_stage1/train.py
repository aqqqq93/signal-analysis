from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from tqdm import trange

from .config import choose_device, load_config
from .losses import (
    pairwise_ridge_nll,
    permutation_l1,
    permutation_slope_l1,
    polynomial_residual,
    polynomial_residual_per_sample,
    second_difference_smoothness,
)
from .model import IFNetUNet, model_config_from_dict, soft_argmax_if
from .simulation import ChirpSimulator, sim_config_from_dict
from .tf import feature_channels, log_spectrogram, sample_if_to_frames, stft_config_from_dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = choose_device(str(cfg.get("device", "auto")))
    run_dir = Path(cfg.get("run_dir", "ifnet_stage1/runs/default"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    sim_cfg = sim_config_from_dict(cfg["data"])
    stft_cfg = stft_config_from_dict(cfg["stft"])
    model_cfg = model_config_from_dict(cfg["model"])
    train_cfg = cfg["train"]

    simulator = ChirpSimulator(sim_cfg, seed=seed)
    model = IFNetUNet(feature_channels(stft_cfg), sim_cfg.num_components, model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-6)),
    )

    loss_weights = train_cfg.get("loss", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    steps = int(train_cfg.get("steps", 3000))
    print_every = int(train_cfg.get("print_every", 50))
    save_every = int(train_cfg.get("save_every", 500))
    ridge_sigma = float(train_cfg.get("ridge_sigma_hz", 6.0))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    start_step = 0

    resume_path = train_cfg.get("resume")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (KeyError, ValueError, RuntimeError):
            pass
        start_step = int(ckpt.get("step", 0))

    history = []
    pbar = trange(1, steps + 1, desc="ifnet-train")
    for local_step in pbar:
        step = start_step + local_step
        model.train()
        batch = simulator.generate_batch(batch_size, device=device)
        feats, freq_grid = log_spectrogram(batch["signal"], stft_cfg, sim_cfg.fs)
        logits = model(feats)
        target_if = sample_if_to_frames(batch["if_hz"], logits.shape[-1], stft_cfg.hop_length)
        pred_if, _ = soft_argmax_if(logits, freq_grid, model_cfg.temperature)

        loss_nll = pairwise_ridge_nll(logits, target_if, freq_grid, ridge_sigma)
        loss_l1 = permutation_l1(pred_if, target_if)
        loss_slope = permutation_slope_l1(pred_if, target_if)
        loss_smooth = second_difference_smoothness(pred_if)
        loss_poly = make_polynomial_loss(pred_if, batch["scenario"], loss_weights)
        loss = (
            float(loss_weights.get("nll", 1.0)) * loss_nll
            + float(loss_weights.get("if_l1", 0.25)) * (loss_l1 / sim_cfg.fs)
            + float(loss_weights.get("identity_slope", 0.0)) * (loss_slope / ridge_sigma)
            + float(loss_weights.get("smooth", 0.01)) * (loss_smooth / (sim_cfg.fs**2))
            + float(loss_weights.get("poly_residual", 0.0)) * (loss_poly / (sim_cfg.fs**2))
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        metrics = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "nll": float(loss_nll.detach().cpu()),
            "if_mae_hz": float(loss_l1.detach().cpu()),
            "slope_mae_hz_per_frame": float(loss_slope.detach().cpu()),
            "smooth": float(loss_smooth.detach().cpu()),
            "poly_residual": float(loss_poly.detach().cpu()),
        }
        history.append(metrics)
        pbar.set_postfix(loss=f"{metrics['loss']:.4f}", mae=f"{metrics['if_mae_hz']:.2f}Hz")

        if step % print_every == 0:
            val = evaluate(model, simulator, stft_cfg, model_cfg, sim_cfg.fs, batch_size, int(train_cfg.get("val_batches", 8)), device)
            metrics.update({f"val_{k}": v for k, v in val.items()})
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")

        if step % save_every == 0 or step == steps:
            save_checkpoint(run_dir / "latest.pt", model, optimizer, cfg, step)
            save_checkpoint(run_dir / f"step_{step:06d}.pt", model, optimizer, cfg, step)

    return {"run_dir": str(run_dir), "last": history[-1] if history else {}}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    simulator: ChirpSimulator,
    stft_cfg,
    model_cfg,
    fs: float,
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    maes = []
    for _ in range(num_batches):
        batch = simulator.generate_batch(batch_size, device=device)
        feats, freq_grid = log_spectrogram(batch["signal"], stft_cfg, fs)
        logits = model(feats)
        target_if = sample_if_to_frames(batch["if_hz"], logits.shape[-1], stft_cfg.hop_length)
        pred_if, _ = soft_argmax_if(logits, freq_grid, model_cfg.temperature)
        maes.append(permutation_l1(pred_if, target_if).detach())
    return {"if_mae_hz": float(torch.stack(maes).mean().cpu())}


def save_checkpoint(path: Path, model, optimizer, cfg: dict[str, Any], step: int) -> None:
    torch.save(
        {
            "step": step,
            "config": cfg,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def make_polynomial_loss(pred_if: torch.Tensor, scenarios: list[str], loss_weights: dict[str, Any]) -> torch.Tensor:
    if float(loss_weights.get("poly_residual", 0.0)) <= 0.0:
        return pred_if.new_tensor(0.0)
    degree = int(loss_weights.get("poly_degree", 3))
    selected = loss_weights.get("poly_residual_scenarios")
    if not selected:
        return polynomial_residual(pred_if, degree=degree)

    selected_set = set(str(item) for item in selected)
    mask = torch.tensor([name in selected_set for name in scenarios], device=pred_if.device, dtype=pred_if.dtype)
    if mask.sum() <= 0:
        return pred_if.new_tensor(0.0)
    per_sample = polynomial_residual_per_sample(pred_if, degree=degree)
    return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ifnet_stage1/configs/default.yaml")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.run_dir is not None:
        cfg["run_dir"] = args.run_dir
    if args.resume is not None:
        cfg["train"]["resume"] = args.resume

    result = train_from_config(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
