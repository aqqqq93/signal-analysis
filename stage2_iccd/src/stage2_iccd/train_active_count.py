from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from ifnet_stage1.config import choose_device, load_config
from ifnet_stage1.simulation import ChirpSimulator, sim_config_from_dict
from ifnet_stage1.tf import feature_channels, log_spectrogram, stft_config_from_dict

from .active_count import (
    ACTIVE_COUNT_NAMES,
    ActiveCountClassifier,
    active_count_config_from_dict,
    active_count_labels,
    active_count_metrics,
)


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
    run_dir = Path(cfg.get("run_dir", "stage2_iccd/runs/active_count"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    sim_cfg = sim_config_from_dict(cfg["data"])
    stft_cfg = stft_config_from_dict(cfg["stft"])
    model_cfg = active_count_config_from_dict(cfg.get("active_count"))
    train_cfg = cfg["train"]

    simulator = ChirpSimulator(sim_cfg, seed=seed)
    model = ActiveCountClassifier(feature_channels(stft_cfg), model_cfg, num_classes=len(ACTIVE_COUNT_NAMES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 4.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-5)),
    )

    start_step = 0
    resume_path = train_cfg.get("resume")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (KeyError, ValueError, RuntimeError):
            print("Optimizer state is incompatible with the current model; restarting optimizer.")
        start_step = int(ckpt.get("step", 0))

    batch_size = int(train_cfg.get("batch_size", 32))
    steps = int(train_cfg.get("steps", 1200))
    print_every = int(train_cfg.get("print_every", 50))
    save_every = int(train_cfg.get("save_every", 300))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    class_weights = _class_weights(train_cfg, device)
    history = []

    pbar = trange(1, steps + 1, desc="active-count")
    for local_step in pbar:
        step = start_step + local_step
        model.train()
        batch = simulator.generate_batch(batch_size, device=device)
        feats, _ = log_spectrogram(batch["signal"], stft_cfg, sim_cfg.fs)
        labels = active_count_labels(batch["active_mask"])

        logits = model(feats)
        loss = F.cross_entropy(logits, labels, weight=class_weights)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        metrics = active_count_metrics(logits.detach(), labels)
        metrics.update({"step": step, "loss": float(loss.detach().cpu())})
        history.append(metrics)
        pbar.set_postfix(loss=f"{metrics['loss']:.4f}", acc=f"{metrics['accuracy']:.3f}", conf=f"{metrics['confidence']:.3f}")

        if step % print_every == 0:
            val = evaluate(model, simulator, stft_cfg, sim_cfg.fs, batch_size, int(train_cfg.get("val_batches", 8)), device)
            metrics.update({f"val_{key}": value for key, value in val.items()})
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")

        if step % save_every == 0 or local_step == steps:
            save_checkpoint(run_dir / "latest.pt", model, optimizer, cfg, model_cfg, step)
            save_checkpoint(run_dir / f"step_{step:06d}.pt", model, optimizer, cfg, model_cfg, step)

    return {"run_dir": str(run_dir), "last": history[-1] if history else {}}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    simulator: ChirpSimulator,
    stft_cfg,
    fs: float,
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    rows = []
    for _ in range(num_batches):
        batch = simulator.generate_batch(batch_size, device=device)
        feats, _ = log_spectrogram(batch["signal"], stft_cfg, fs)
        labels = active_count_labels(batch["active_mask"])
        logits = model(feats)
        losses.append(F.cross_entropy(logits, labels).detach())
        rows.append(active_count_metrics(logits, labels))
    merged = {key: float(np.nanmean([row[key] for row in rows])) for key in rows[0]}
    merged["loss"] = float(torch.stack(losses).mean().cpu())
    return merged


def save_checkpoint(path: Path, model, optimizer, cfg: dict[str, Any], model_cfg, step: int) -> None:
    torch.save(
        {
            "step": step,
            "config": cfg,
            "active_count_names": ACTIVE_COUNT_NAMES,
            "model_cfg": model_cfg.__dict__,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def _class_weights(train_cfg: dict[str, Any], device: torch.device) -> torch.Tensor | None:
    raw = train_cfg.get("class_weights")
    if not raw:
        return None
    values = [float(raw.get(name, 1.0)) for name in ACTIVE_COUNT_NAMES]
    return torch.tensor(values, dtype=torch.float32, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage2_iccd/configs/active_count_simple.yaml")
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
