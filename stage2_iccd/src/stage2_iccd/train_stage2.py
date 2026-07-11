from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from tqdm import trange

from ifnet_stage1.config import choose_device, load_config
from ifnet_stage1.simulation import ChirpSimulator, sim_config_from_dict

from .candidates import FrozenIFNetCandidateProvider, OraclePerturbedCandidateProvider
from .differentiable_iccd import iccd_config_from_dict
from .losses import (
    candidate_entropy,
    active_component_permutation_mse,
    active_component_permutation_l1,
    component_permutation_mse,
    component_permutation_l1,
    crossing_identity_loss,
    if_smoothness,
    if_third_derivative,
    min_gap_barrier,
    reconstruction_snr_db,
    sinusoidal_curvature_consistency,
)
from .model import Stage2ICCDModel, stage2_model_config_from_dict


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
    run_dir = Path(cfg.get("run_dir", "stage2_iccd/runs/default"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    sim_cfg = sim_config_from_dict(cfg["data"])
    iccd_cfg = iccd_config_from_dict({**cfg["iccd"], "fs": sim_cfg.fs, "n_samples": sim_cfg.n_samples})
    model_cfg = stage2_model_config_from_dict(
        {
            **cfg["model"],
            "num_candidates": int(cfg.get("init", {}).get("num_candidates", cfg["model"].get("num_candidates", 2))),
            "freq_min": sim_cfg.freq_min,
            "freq_max": sim_cfg.freq_max,
        }
    )
    simulator = ChirpSimulator(sim_cfg, seed=seed)
    model = Stage2ICCDModel(iccd_cfg, model_cfg, num_components=sim_cfg.num_components).to(device)

    init_cfg = cfg.get("init", {})
    provider = make_candidate_provider(init_cfg, device=device, seed=seed)
    train_cfg = cfg["train"]
    optimizer = make_optimizer(model, provider, train_cfg)

    start_step = 0
    resume_path = train_cfg.get("resume")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        load_stage2_model_state(model, ckpt["model"])
        if hasattr(provider, "load_state_dict") and "provider" in ckpt:
            provider.load_state_dict(ckpt["provider"])
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (KeyError, ValueError, RuntimeError):
            print("Optimizer state is incompatible with the current model; restarting optimizer.")
        start_step = int(ckpt.get("step", 0))

    batch_size = int(train_cfg.get("batch_size", 8))
    steps = int(train_cfg.get("steps", 500))
    print_every = int(train_cfg.get("print_every", 25))
    save_every = int(train_cfg.get("save_every", 250))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    weights = train_cfg.get("loss", {})
    ohem_cfg = train_cfg.get("ohem", {})

    history = []
    pbar = trange(1, steps + 1, desc="stage2-iccd")
    for local_step in pbar:
        step = start_step + local_step
        model.train()
        batch = simulator.generate_batch(batch_size, device=device)
        signal = batch["signal"]
        clean = batch["clean"]
        target_if = batch["if_hz"]
        target_components = batch["components"]
        active_mask = batch.get("active_mask")
        candidate_if = get_candidates(provider, init_cfg, signal, target_if, sim_cfg.n_samples)
        candidate_if = candidate_if.clamp(sim_cfg.freq_min, sim_cfg.freq_max)
        if not bool(getattr(provider, "trainable", False)):
            candidate_if = candidate_if.detach()
        refinement_extra = build_refinement_extra(batch, model_cfg, sim_cfg.n_samples, train_cfg.get("refinement_extra", {}))

        out = model(signal, candidate_if, refinement_extra=refinement_extra)
        loss, metrics = compute_loss(out, clean, target_components, target_if, sim_cfg.fs, weights, active_mask=active_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        metrics.update(
            {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "alpha": float(out["alpha"].detach().cpu()),
                "candidate_temperature": float(out["candidate_temperature"].detach().cpu()),
                "candidate_weights": [float(v) for v in out["candidate_weights"].detach().mean(dim=0).cpu()],
            }
        )
        history.append(metrics)
        pbar.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            snr=f"{metrics['rec_snr_db']:.2f}dB",
            ifmae=f"{metrics['if_mae_hz']:.2f}Hz",
        )

        if step % print_every == 0:
            val = evaluate(
                model,
                simulator,
                provider,
                init_cfg,
                sim_cfg,
                weights,
                batch_size,
                int(train_cfg.get("val_batches", 4)),
                device,
                train_cfg.get("refinement_extra", {}),
            )
            metrics.update({f"val_{k}": v for k, v in val.items()})
            maybe_update_ohem_sampling(simulator, val, ohem_cfg)
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")

        if step % save_every == 0 or step == steps:
            save_checkpoint(run_dir / "latest.pt", model, optimizer, cfg, step, provider=provider)
            save_checkpoint(run_dir / f"step_{step:06d}.pt", model, optimizer, cfg, step, provider=provider)

    return {"run_dir": str(run_dir), "last": history[-1] if history else {}}


def make_candidate_provider(init_cfg: dict[str, Any], device: torch.device, seed: int):
    mode = str(init_cfg.get("mode", "oracle_perturbed"))
    if mode == "frozen_ifnet_checkpoint":
        checkpoints = init_cfg.get("checkpoints")
        if checkpoints is not None:
            checkpoints = [str(item) for item in checkpoints]
        return FrozenIFNetCandidateProvider(
            checkpoint=init_cfg.get("checkpoint"),
            checkpoints=checkpoints,
            device=device,
            num_candidates=int(init_cfg.get("num_candidates", 2)),
            smooth_kernel=int(init_cfg.get("smooth_kernel", 31)),
            trainable=bool(init_cfg.get("trainable", False)),
            unfreeze_last_decoders=int(init_cfg.get("unfreeze_last_decoders", 0)),
            unfreeze_head=bool(init_cfg.get("unfreeze_head", True)),
        )
    if mode == "oracle_perturbed":
        return OraclePerturbedCandidateProvider(
            num_candidates=int(init_cfg.get("num_candidates", 2)),
            noise_hz=float(init_cfg.get("noise_hz", 10.0)),
            alt_noise_hz=float(init_cfg.get("alt_noise_hz", 24.0)),
            smooth_kernel=int(init_cfg.get("smooth_kernel", 31)),
            seed=seed,
        )
    raise ValueError(f"Unknown init.mode: {mode}")


def make_optimizer(model: Stage2ICCDModel, provider, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-6))
    groups: list[dict[str, Any]] = [
        {
            "params": list(model.parameters()),
            "lr": float(train_cfg.get("lr", 2.0e-4)),
            "weight_decay": weight_decay,
        }
    ]
    if bool(getattr(provider, "trainable", False)) and hasattr(provider, "trainable_parameters"):
        params = provider.trainable_parameters()
        if params:
            groups.append(
                {
                    "params": params,
                    "lr": float(train_cfg.get("stage1_lr", 1.0e-5)),
                    "weight_decay": float(train_cfg.get("stage1_weight_decay", weight_decay)),
                }
            )
    return torch.optim.AdamW(groups)


def get_candidates(provider, init_cfg: dict[str, Any], signal: torch.Tensor, target_if: torch.Tensor, n_samples: int) -> torch.Tensor:
    if str(init_cfg.get("mode", "oracle_perturbed")) == "frozen_ifnet_checkpoint":
        return provider(signal, n_samples)
    return provider(signal, target_if)


def build_refinement_extra(
    batch: dict[str, Any],
    model_cfg,
    n_samples: int,
    cfg: dict[str, Any] | None = None,
) -> torch.Tensor | None:
    channels = int(getattr(model_cfg, "refine_extra_channels", 0))
    if channels <= 0:
        return None
    signal = batch["signal"]
    device = signal.device
    dtype = signal.dtype
    cfg = cfg or {}
    mode = str(cfg.get("mode", "jump_mask"))
    if mode != "jump_mask":
        return signal.new_zeros((signal.shape[0], channels, n_samples))
    centers = batch.get("jump_center")
    valid = batch.get("jump_valid")
    if centers is None or valid is None:
        return signal.new_zeros((signal.shape[0], channels, n_samples))
    centers = centers.to(device=device, dtype=dtype)
    valid = valid.to(device=device, dtype=torch.bool)
    time = torch.linspace(0.0, 1.0, n_samples, device=device, dtype=dtype)
    sigma = float(cfg.get("sigma", 0.035))
    sigma = max(sigma, 1.0 / max(float(n_samples), 1.0))
    masks = torch.exp(-0.5 * ((time.view(1, 1, -1) - centers.unsqueeze(-1)) / sigma) ** 2)
    masks = masks * valid.to(dtype=dtype).unsqueeze(-1)
    if masks.shape[1] < channels:
        pad = masks.new_zeros((masks.shape[0], channels - masks.shape[1], n_samples))
        masks = torch.cat([masks, pad], dim=1)
    return masks[:, :channels].clamp(0.0, 1.0)


def compute_loss(
    out: dict[str, torch.Tensor],
    clean: torch.Tensor,
    target_components: torch.Tensor,
    target_if: torch.Tensor,
    fs: float,
    weights: dict[str, Any],
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    rec = out["reconstruction"]
    comps = out["components"]
    refined_if = out["refined_if_hz"]
    loss_rec = torch.mean((rec - clean).pow(2))
    if active_mask is not None:
        loss_comp, loss_inactive = active_component_permutation_mse(
            comps,
            target_components,
            active_mask=active_mask,
            inactive_weight=float(weights.get("inactive_component", 0.15)),
        )
        loss_comp_l1 = active_component_permutation_l1(comps, target_components, active_mask=active_mask)
    else:
        loss_comp = component_permutation_mse(comps, target_components)
        loss_inactive = comps.new_tensor(0.0)
        loss_comp_l1 = component_permutation_l1(comps, target_components)
    loss_if = masked_permutation_l1(refined_if, target_if, active_mask)
    loss_smooth = if_smoothness(refined_if)
    loss_third = if_third_derivative(refined_if)
    loss_crossing_identity = crossing_identity_loss(
        refined_if,
        gap_sigma_hz=float(weights.get("crossing_gap_sigma_hz", 24.0)),
    )
    loss_min_gap = min_gap_barrier(refined_if, min_gap_hz=float(weights.get("min_gap_hz", 8.0)))
    loss_sinusoidal = sinusoidal_curvature_consistency(refined_if)
    loss_entropy = candidate_entropy(out["candidate_weights"])
    loss_delta = out["delta_if_hz"].pow(2).mean()
    loss = (
        float(weights.get("reconstruction", 1.0)) * loss_rec
        + float(weights.get("component", 0.25)) * loss_comp
        + float(weights.get("if_l1", 0.05)) * (loss_if / float(fs))
        + float(weights.get("smooth", 0.001)) * (loss_smooth / (float(fs) ** 2))
        + float(weights.get("third_derivative", 0.0)) * (loss_third / (float(fs) ** 2))
        + float(weights.get("crossing_identity", 0.0)) * (loss_crossing_identity / (float(fs) ** 2))
        + float(weights.get("min_gap", 0.0)) * (loss_min_gap / (float(fs) ** 2))
        + float(weights.get("sinusoidal_curvature", 0.0)) * (loss_sinusoidal / (float(fs) ** 2))
        + float(weights.get("delta", 0.0005)) * (loss_delta / (float(fs) ** 2))
        - float(weights.get("candidate_entropy", 0.0)) * loss_entropy
    )
    metrics = {
        "rec_mse": float(loss_rec.detach().cpu()),
        "component_mse": float(loss_comp.detach().cpu()),
        "inactive_component_mse": float(loss_inactive.detach().cpu()),
        "component_l1": float(loss_comp_l1.detach().cpu()),
        "if_mae_hz": float(loss_if.detach().cpu()),
        "smooth": float(loss_smooth.detach().cpu()),
        "third_derivative": float(loss_third.detach().cpu()),
        "crossing_identity": float(loss_crossing_identity.detach().cpu()),
        "min_gap": float(loss_min_gap.detach().cpu()),
        "sinusoidal_curvature": float(loss_sinusoidal.detach().cpu()),
        "delta_rms_hz": float(torch.sqrt(loss_delta.detach()).cpu()),
        "rec_snr_db": float(reconstruction_snr_db(clean, rec).mean().detach().cpu()),
        "active_components": float(active_mask.sum(dim=1).mean().detach().cpu()) if active_mask is not None else float(target_if.shape[1]),
    }
    return loss, metrics


def masked_permutation_l1(pred_if: torch.Tensor, target_if: torch.Tensor, active_mask: torch.Tensor | None = None) -> torch.Tensor:
    if pred_if.shape[:2] != target_if.shape[:2]:
        raise ValueError("Predicted and target IF tensors must share [B, Q].")
    bsz, q, _ = pred_if.shape
    if active_mask is None:
        active_mask = torch.ones((bsz, q), device=pred_if.device, dtype=pred_if.dtype)
    else:
        active_mask = active_mask.to(device=pred_if.device, dtype=pred_if.dtype)
    perms = list(itertools.permutations(range(q)))
    rows = torch.arange(q, device=pred_if.device)
    costs = []
    component_cost = torch.empty((bsz, q, q), device=pred_if.device, dtype=pred_if.dtype)
    for pred_idx in range(q):
        component_cost[:, pred_idx, :] = (pred_if[:, pred_idx : pred_idx + 1] - target_if).abs().mean(dim=-1)
    for perm in perms:
        perm_tensor = torch.tensor(perm, device=pred_if.device)
        matched = component_cost[:, rows, perm_tensor]
        matched_mask = active_mask[:, perm_tensor]
        costs.append((matched * matched_mask).sum(dim=1) / matched_mask.sum(dim=1).clamp_min(1.0))
    return torch.stack(costs, dim=1).min(dim=1).values.mean()


@torch.no_grad()
def evaluate(
    model: Stage2ICCDModel,
    simulator: ChirpSimulator,
    provider,
    init_cfg: dict[str, Any],
    sim_cfg,
    weights: dict[str, Any],
    batch_size: int,
    num_batches: int,
    device: torch.device,
    refinement_extra_cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    model.eval()
    rows = []
    scenario_rows: dict[str, list[dict[str, float]]] = {}
    for _ in range(num_batches):
        batch = simulator.generate_batch(batch_size, device=device)
        candidate_if = get_candidates(provider, init_cfg, batch["signal"], batch["if_hz"], sim_cfg.n_samples)
        candidate_if = candidate_if.clamp(sim_cfg.freq_min, sim_cfg.freq_max).detach()
        refinement_extra = build_refinement_extra(batch, model.model_cfg, sim_cfg.n_samples, refinement_extra_cfg)
        out = model(batch["signal"], candidate_if, refinement_extra=refinement_extra)
        _, metrics = compute_loss(
            out,
            batch["clean"],
            batch["components"],
            batch["if_hz"],
            sim_cfg.fs,
            weights,
            active_mask=batch.get("active_mask"),
        )
        rows.append(metrics)
        for scenario in sorted(set(batch.get("scenario", []))):
            indices = [idx for idx, name in enumerate(batch.get("scenario", [])) if name == scenario]
            if not indices:
                continue
            idx_tensor = torch.tensor(indices, device=device)
            sub_out = {key: value.index_select(0, idx_tensor) if isinstance(value, torch.Tensor) and value.shape[:1] == (len(batch["scenario"]),) else value for key, value in out.items()}
            _loss, sub_metrics = compute_loss(
                sub_out,
                batch["clean"].index_select(0, idx_tensor),
                batch["components"].index_select(0, idx_tensor),
                batch["if_hz"].index_select(0, idx_tensor),
                sim_cfg.fs,
                weights,
                active_mask=batch.get("active_mask").index_select(0, idx_tensor) if batch.get("active_mask") is not None else None,
            )
            scenario_rows.setdefault(scenario, []).append(sub_metrics)
    merged = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    for scenario, items in scenario_rows.items():
        merged[f"scenario_{scenario}_if_mae_hz"] = float(np.mean([row["if_mae_hz"] for row in items]))
    return merged


def save_checkpoint(path: Path, model: Stage2ICCDModel, optimizer, cfg: dict[str, Any], step: int, provider=None) -> None:
    payload = {
        "step": step,
        "config": cfg,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if provider is not None and bool(getattr(provider, "trainable", False)) and hasattr(provider, "state_dict"):
        payload["provider"] = provider.state_dict()
    torch.save(
        payload,
        path,
    )


def maybe_update_ohem_sampling(simulator: ChirpSimulator, val: dict[str, float], ohem_cfg: dict[str, Any]) -> None:
    if not ohem_cfg or not bool(ohem_cfg.get("enabled", False)):
        return
    scenario_mae = {
        key.removeprefix("scenario_").removesuffix("_if_mae_hz"): value
        for key, value in val.items()
        if key.startswith("scenario_") and key.endswith("_if_mae_hz")
    }
    if not scenario_mae:
        return
    threshold = float(np.percentile(list(scenario_mae.values()), float(ohem_cfg.get("percentile", 80.0))))
    boost = float(ohem_cfg.get("boost", 3.0))
    base = dict(simulator.cfg.scenario_weights or {name: 1.0 for name in simulator.scenario_names})
    updated = {}
    for name in simulator.scenario_names:
        value = float(base.get(name, 0.0))
        if name in scenario_mae and scenario_mae[name] >= threshold:
            value *= boost
        updated[name] = value
    simulator.scenario_probs = simulator._normalize_probs([updated[name] for name in simulator.scenario_names])


def load_stage2_model_state(model: Stage2ICCDModel, state: dict[str, torch.Tensor]) -> None:
    """Load current or legacy stage-2 checkpoints.

    Early checkpoints used a single global `mixer.logits` parameter. The current
    mixer uses `mixer.bias` plus a learnable temperature. Mapping logits to bias
    keeps old runs usable while leaving new parameters at their initialized
    values.
    """

    migrated = dict(state)
    if "mixer.logits" in migrated and "mixer.bias" not in migrated:
        migrated["mixer.bias"] = migrated.pop("mixer.logits")
    current = model.state_dict()
    for key, value in list(migrated.items()):
        if key not in current or tuple(value.shape) == tuple(current[key].shape):
            continue
        if key == "refine_head.net.0.weight" and value.ndim == 3 and current[key].ndim == 3:
            fixed = current[key].clone()
            fixed.zero_()
            out_ch = min(fixed.shape[0], value.shape[0])
            in_ch = min(fixed.shape[1], value.shape[1])
            width = min(fixed.shape[2], value.shape[2])
            fixed[:out_ch, :in_ch, :width] = value[:out_ch, :in_ch, :width]
            migrated[key] = fixed
            continue
        migrated.pop(key)
    missing, unexpected = model.load_state_dict(migrated, strict=False)
    allowed_missing = {"mixer.raw_temperature"}
    real_missing = [key for key in missing if key not in allowed_missing]
    real_missing = [key for key in real_missing if not key.startswith("refine_head.jump_net.")]
    real_missing = [key for key in real_missing if not key.startswith("mixer.quality_net.")]
    if real_missing or unexpected:
        raise RuntimeError(f"Could not load stage-2 checkpoint. missing={real_missing}, unexpected={unexpected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="stage2_iccd/configs/default.yaml")
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
