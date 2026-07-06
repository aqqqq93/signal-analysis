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

from .config import choose_device, load_config
from .router import (
    DEFAULT_SCENARIO_TO_ROUTE,
    ROUTE_NAMES,
    HardRouteClassifier,
    router_config_from_dict,
    scenario_to_route_labels,
)
from .simulation import ChirpSimulator, sim_config_from_dict
from .tf import feature_channels, log_spectrogram, stft_config_from_dict


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
    run_dir = Path(cfg.get("run_dir", "ifnet_stage1/runs/router"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    sim_cfg = sim_config_from_dict(cfg["data"])
    stft_cfg = stft_config_from_dict(cfg["stft"])
    router_cfg = router_config_from_dict(cfg["router"])
    train_cfg = cfg["train"]
    route_names = tuple(cfg.get("route_names", ROUTE_NAMES))
    scenario_to_route = dict(cfg.get("scenario_to_route", DEFAULT_SCENARIO_TO_ROUTE))

    simulator = ChirpSimulator(sim_cfg, seed=seed)
    model = HardRouteClassifier(feature_channels(stft_cfg), len(route_names), router_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 5.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-5)),
    )

    batch_size = int(train_cfg.get("batch_size", 32))
    steps = int(train_cfg.get("steps", 1200))
    print_every = int(train_cfg.get("print_every", 50))
    save_every = int(train_cfg.get("save_every", 300))
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
    pbar = trange(1, steps + 1, desc="router-train")
    for local_step in pbar:
        step = start_step + local_step
        model.train()
        batch = simulator.generate_batch(batch_size, device=device)
        feats, _ = log_spectrogram(batch["signal"], stft_cfg, sim_cfg.fs)
        labels = scenario_to_route_labels(
            batch["scenario"],
            route_names=route_names,
            scenario_to_route=scenario_to_route,
            device=device,
        )

        logits = model(feats)
        class_weights = route_loss_weights(train_cfg, route_names, device)
        loss = F.cross_entropy(logits, labels, weight=class_weights)
        acc = (logits.argmax(dim=1) == labels).float().mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        metrics = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "accuracy": float(acc.detach().cpu()),
        }
        history.append(metrics)
        pbar.set_postfix(loss=f"{metrics['loss']:.4f}", acc=f"{metrics['accuracy']:.3f}")

        if step % print_every == 0:
            val = evaluate(
                model,
                simulator,
                stft_cfg,
                sim_cfg.fs,
                batch_size,
                int(train_cfg.get("val_batches", 8)),
                route_names,
                scenario_to_route,
                device,
            )
            metrics.update({f"val_{key}": value for key, value in val.items()})
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")

        if step % save_every == 0 or local_step == steps:
            save_checkpoint(run_dir / "latest.pt", model, optimizer, cfg, route_names, scenario_to_route, step)
            save_checkpoint(run_dir / f"step_{step:06d}.pt", model, optimizer, cfg, route_names, scenario_to_route, step)

    return {"run_dir": str(run_dir), "last": history[-1] if history else {}}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    simulator: ChirpSimulator,
    stft_cfg,
    fs: float,
    batch_size: int,
    num_batches: int,
    route_names: tuple[str, ...],
    scenario_to_route: dict[str, str],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    accuracies = []
    for _ in range(num_batches):
        batch = simulator.generate_batch(batch_size, device=device)
        feats, _ = log_spectrogram(batch["signal"], stft_cfg, fs)
        labels = scenario_to_route_labels(
            batch["scenario"],
            route_names=route_names,
            scenario_to_route=scenario_to_route,
            device=device,
        )
        logits = model(feats)
        losses.append(F.cross_entropy(logits, labels).detach())
        accuracies.append((logits.argmax(dim=1) == labels).float().mean().detach())
    return {
        "loss": float(torch.stack(losses).mean().cpu()),
        "accuracy": float(torch.stack(accuracies).mean().cpu()),
    }


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    cfg: dict[str, Any],
    route_names: tuple[str, ...],
    scenario_to_route: dict[str, str],
    step: int,
) -> None:
    torch.save(
        {
            "step": step,
            "config": cfg,
            "route_names": route_names,
            "scenario_to_route": scenario_to_route,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def route_loss_weights(train_cfg: dict[str, Any], route_names: tuple[str, ...], device: torch.device) -> torch.Tensor | None:
    raw = train_cfg.get("route_loss_weights")
    if not raw:
        return None
    weights = [float(raw.get(name, 1.0)) for name in route_names]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ifnet_stage1/configs/router_hard.yaml")
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
