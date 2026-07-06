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
from .jump_aux import (
    IFNetJumpAux,
    jump_aux_config_from_dict,
    jump_location_from_centers,
    make_jump_center_targets,
    make_jump_targets,
    masked_jump_nll_loss,
)
from .losses import pairwise_ridge_nll, permutation_l1, permutation_slope_l1, second_difference_smoothness
from .model import model_config_from_dict, soft_argmax_if
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
    run_dir = Path(cfg.get("run_dir", "ifnet_stage1/runs/local_jump_aux"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    sim_cfg = sim_config_from_dict(cfg["data"])
    stft_cfg = stft_config_from_dict(cfg["stft"])
    model_cfg = model_config_from_dict(cfg["model"])
    jump_cfg = jump_aux_config_from_dict(cfg.get("jump_aux"))
    train_cfg = cfg["train"]

    simulator = ChirpSimulator(sim_cfg, seed=seed)
    model = IFNetJumpAux(feature_channels(stft_cfg), sim_cfg.num_components, model_cfg).to(device)
    resume_path = train_cfg.get("resume")
    if resume_path:
        load_resume(model, resume_path, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-6)),
    )

    loss_weights = train_cfg.get("loss", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    steps = int(train_cfg.get("steps", 1200))
    print_every = int(train_cfg.get("print_every", 50))
    save_every = int(train_cfg.get("save_every", 300))
    ridge_sigma = float(train_cfg.get("ridge_sigma_hz", 6.0))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    history = []
    pbar = trange(1, steps + 1, desc="jump-aux-train")
    for step in pbar:
        model.train()
        batch = simulator.generate_batch(batch_size, device=device)
        feats, freq_grid = log_spectrogram(batch["signal"], stft_cfg, sim_cfg.fs)
        ridge_logits, jump_logits = model(feats)
        target_if = sample_if_to_frames(batch["if_hz"], ridge_logits.shape[-1], stft_cfg.hop_length)
        pred_if, _ = soft_argmax_if(ridge_logits, freq_grid, model_cfg.temperature)
        if "jump_center" in batch and "jump_valid" in batch:
            jump_target, jump_valid = make_jump_center_targets(
                batch["jump_center"],
                batch["jump_valid"],
                frames=jump_logits.shape[-1],
                sigma_frames=jump_cfg.jump_sigma_frames,
            )
        else:
            jump_target = make_jump_targets(
                target_if,
                sigma_frames=jump_cfg.jump_sigma_frames,
                min_jump_hz=jump_cfg.min_jump_hz,
            )
            jump_valid = torch.ones(jump_logits.shape[:2], device=jump_logits.device, dtype=torch.bool)

        loss_nll = pairwise_ridge_nll(ridge_logits, target_if, freq_grid, ridge_sigma)
        loss_l1 = permutation_l1(pred_if, target_if)
        loss_slope = permutation_slope_l1(pred_if, target_if)
        loss_smooth = second_difference_smoothness(pred_if)
        loss_jump = masked_jump_nll_loss(jump_logits, jump_target, jump_valid)
        loss = (
            jump_cfg.ridge_weight * float(loss_weights.get("nll", 1.0)) * loss_nll
            + jump_cfg.ridge_weight * float(loss_weights.get("if_l1", 0.30)) * (loss_l1 / sim_cfg.fs)
            + jump_cfg.ridge_weight * float(loss_weights.get("identity_slope", 0.12)) * (loss_slope / ridge_sigma)
            + jump_cfg.ridge_weight * float(loss_weights.get("smooth", 0.004)) * (loss_smooth / (sim_cfg.fs**2))
            + jump_cfg.jump_weight * loss_jump
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "nll": float(loss_nll.detach().cpu()),
            "if_mae_hz": float(loss_l1.detach().cpu()),
            "jump_nll": float(loss_jump.detach().cpu()),
        }
        if step % print_every == 0 or step == steps:
            row.update(
                {
                    f"val_{k}": v
                    for k, v in evaluate(model, simulator, stft_cfg, model_cfg, jump_cfg, sim_cfg.fs, batch_size, int(train_cfg.get("val_batches", 8)), device).items()
                }
            )
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        if step % save_every == 0 or step == steps:
            save_checkpoint(run_dir / "latest.pt", model, optimizer, cfg, step)
            save_checkpoint(run_dir / f"step_{step:06d}.pt", model, optimizer, cfg, step)
        history.append(row)
        pbar.set_postfix(mae=f"{row['if_mae_hz']:.2f}", jump=f"{row['jump_nll']:.3f}")

    return {"run_dir": str(run_dir), "last": history[-1] if history else {}}


@torch.no_grad()
def evaluate(model, simulator, stft_cfg, model_cfg, jump_cfg, fs: float, batch_size: int, num_batches: int, device: torch.device) -> dict[str, float]:
    model.eval()
    maes = []
    jump_frame_err = []
    for _ in range(num_batches):
        batch = simulator.generate_batch(batch_size, device=device)
        feats, freq_grid = log_spectrogram(batch["signal"], stft_cfg, fs)
        ridge_logits, jump_logits = model(feats)
        target_if = sample_if_to_frames(batch["if_hz"], ridge_logits.shape[-1], stft_cfg.hop_length)
        pred_if, _ = soft_argmax_if(ridge_logits, freq_grid, model_cfg.temperature)
        maes.append(permutation_l1(pred_if, target_if).detach())
        if "jump_center" in batch and "jump_valid" in batch:
            target_jump = jump_location_from_centers(batch["jump_center"], jump_logits.shape[-1])
            jump_valid = batch["jump_valid"].to(dtype=torch.bool)
        else:
            target_jump = make_jump_targets(target_if, jump_cfg.jump_sigma_frames, jump_cfg.min_jump_hz).argmax(dim=-1)
            jump_valid = torch.ones(jump_logits.shape[:2], device=jump_logits.device, dtype=torch.bool)
        pred_jump = jump_logits.argmax(dim=-1)
        err = (pred_jump - target_jump).abs().float()
        weights = jump_valid.to(dtype=err.dtype)
        jump_frame_err.append((err * weights).sum() / weights.sum().clamp_min(1.0))
    return {
        "if_mae_hz": float(torch.stack(maes).mean().cpu()),
        "jump_frame_mae": float(torch.stack(jump_frame_err).mean().cpu()),
    }


def load_resume(model: IFNetJumpAux, path: str | Path, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"]
    mapped = {}
    for key, value in state.items():
        if key.startswith("head."):
            mapped["ridge_head." + key[len("head.") :]] = value
        else:
            mapped[key] = value
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    allowed_missing = {name for name in missing if name.startswith("jump_head.")}
    if set(missing) != allowed_missing or unexpected:
        print({"missing": missing, "unexpected": unexpected})


def save_checkpoint(path: Path, model, optimizer, cfg: dict[str, Any], step: int) -> None:
    torch.save(
        {
            "step": step,
            "config": cfg,
            "model_type": "IFNetJumpAux",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ifnet_stage1/configs/local_jump_aux.yaml")
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
    print(json.dumps(train_from_config(cfg), indent=2))


if __name__ == "__main__":
    main()
