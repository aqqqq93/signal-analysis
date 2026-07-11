from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from ifnet_stage1.tf import feature_channels, log_spectrogram, stft_config_from_dict

from .active_count import (
    ActiveCountClassifier,
    active_count_config_from_dict,
    active_count_names,
    compute_peak_count_features,
)
from .differentiable_iccd import iccd_config_from_dict
from .model import Stage2ICCDModel, stage2_model_config_from_dict
from .train_stage2 import (
    build_refinement_extra,
    get_candidates,
    load_stage2_model_state,
    make_candidate_provider,
)
from ifnet_stage1.simulation import sim_config_from_dict


DIFFICULT_SCENARIOS = {"crossing", "sinusoidal_fm", "tangent_or_overlap"}


@dataclass
class P15PipelineConfig:
    active_checkpoint: str = "stage2_iccd/runs/active_count_simple_near_parallel/latest.pt"
    single_checkpoint: str = "stage2_iccd/runs/simple_single_component/latest.pt"
    multi_checkpoint: str = "stage2_iccd/runs/simple_multicomponent_long/latest.pt"
    local_jump_checkpoint: str = "stage2_iccd/runs/local_jump_segmented_p1/latest.pt"
    all_expert_checkpoint: str = "stage2_iccd/runs/all_multiexpert_ohem_p1/latest.pt"
    low_confidence_all_expert_threshold: float = 0.58
    near_parallel_second_peak_rate: float = 0.18
    near_parallel_second_peak_ratio: float = 0.32
    use_scenario_hints: bool = True


class Stage2Branch:
    def __init__(self, name: str, checkpoint: str | Path, device: torch.device):
        self.name = str(name)
        self.checkpoint = str(checkpoint)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.cfg = ckpt["config"]
        self.sim_cfg = sim_config_from_dict(self.cfg["data"])
        iccd_cfg = iccd_config_from_dict({**self.cfg["iccd"], "fs": self.sim_cfg.fs, "n_samples": self.sim_cfg.n_samples})
        model_cfg = stage2_model_config_from_dict(
            {
                **self.cfg["model"],
                "num_candidates": int(
                    self.cfg.get("init", {}).get("num_candidates", self.cfg["model"].get("num_candidates", 2))
                ),
                "freq_min": self.sim_cfg.freq_min,
                "freq_max": self.sim_cfg.freq_max,
            }
        )
        self.model = Stage2ICCDModel(iccd_cfg, model_cfg, num_components=self.sim_cfg.num_components).to(device)
        load_stage2_model_state(self.model, ckpt["model"])
        self.model.eval()
        self.init_cfg = self.cfg.get("init", {})
        self.provider = make_candidate_provider(self.init_cfg, device=device, seed=int(self.cfg.get("seed", 0)) + 5051)
        if "provider" in ckpt and hasattr(self.provider, "load_state_dict"):
            self.provider.load_state_dict(ckpt["provider"])
        self.weights = self.cfg["train"].get("loss", {})
        self.refinement_extra_cfg = self.cfg.get("train", {}).get("refinement_extra", {})

    @torch.no_grad()
    def run(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        candidate_if = get_candidates(self.provider, self.init_cfg, batch["signal"], batch["if_hz"], batch["if_hz"].shape[-1])
        candidate_if = candidate_if.clamp(self.sim_cfg.freq_min, self.sim_cfg.freq_max).detach()
        refinement_extra = build_refinement_extra(
            batch,
            self.model.model_cfg,
            batch["if_hz"].shape[-1],
            self.refinement_extra_cfg,
        )
        return self.model(batch["signal"], candidate_if, refinement_extra=refinement_extra)


class P15Stage2Pipeline:
    """Stable P1.5 inference wrapper around the best current Stage2 branches."""

    def __init__(self, cfg: P15PipelineConfig | None = None, device: torch.device | str = "cpu"):
        self.cfg = cfg or P15PipelineConfig()
        self.device = torch.device(device)
        active_ckpt = torch.load(self.cfg.active_checkpoint, map_location=self.device, weights_only=False)
        self.active_cfg = active_ckpt["config"]
        self.stft_cfg = stft_config_from_dict(self.active_cfg["stft"])
        active_model_cfg = active_count_config_from_dict(active_ckpt.get("model_cfg", self.active_cfg.get("active_count")))
        self.active_names = tuple(active_ckpt.get("active_count_names", active_count_names(active_model_cfg.num_classes)))
        self.active_model = ActiveCountClassifier(
            feature_channels(self.stft_cfg),
            active_model_cfg,
            num_classes=len(self.active_names),
        ).to(self.device)
        self.active_model.load_state_dict(active_ckpt["model"])
        self.active_model.eval()
        self.branches = {
            "single": Stage2Branch("single", self.cfg.single_checkpoint, self.device),
            "multi": Stage2Branch("multi", self.cfg.multi_checkpoint, self.device),
            "local_jump": Stage2Branch("local_jump", self.cfg.local_jump_checkpoint, self.device),
            "all_expert": Stage2Branch("all_expert", self.cfg.all_expert_checkpoint, self.device),
        }

    @torch.no_grad()
    def run(
        self,
        batch: dict[str, Any],
        fs: float,
        scenario_hints: Sequence[str] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        route = self.route(batch["signal"], fs, scenario_hints=scenario_hints)
        outputs: dict[int, dict[str, torch.Tensor]] = {}
        for branch_name in sorted(set(route["branch"])):
            mask = torch.tensor([name == branch_name for name in route["branch"]], device=self.device, dtype=torch.bool)
            sub_batch = slice_batch(batch, mask)
            sub_out = self.branches[branch_name].run(sub_batch)
            keep = torch.where(mask)[0].detach().cpu().tolist()
            for local_idx, original_idx in enumerate(keep):
                outputs[int(original_idx)] = {key: _select_sample(value, local_idx) for key, value in sub_out.items()}
        merged = merge_sample_outputs(outputs, batch_size=batch["signal"].shape[0], device=self.device)
        route["candidate_top2_weights"], route["candidate_top2_indices"] = topk_candidate_weights(
            merged.get("candidate_weights"),
            k=2,
        )
        if "refined_if_hz" in merged:
            merged["identity_stable_if_hz"] = identity_stable_if(merged["refined_if_hz"])
        return merged, route

    @torch.no_grad()
    def route(
        self,
        signal: torch.Tensor,
        fs: float,
        scenario_hints: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        feats, _ = log_spectrogram(signal, self.stft_cfg, fs)
        logits = self.active_model(feats)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        confidence = probs.max(dim=1).values
        peak_features = compute_peak_count_features(feats, 8)
        second_peak_rate = peak_features[:, 0]
        second_peak_ratio = peak_features[:, 5]
        hints = list(scenario_hints) if scenario_hints is not None else [None] * signal.shape[0]
        if len(hints) != signal.shape[0]:
            raise ValueError("scenario_hints must have one item per sample.")
        branches: list[str] = []
        for idx, hint in enumerate(hints):
            branch = "single" if int(pred[idx]) == 0 else "multi"
            near_parallel_has_two_ridges = (
                hint == "near_parallel"
                and int(pred[idx]) > 0
                and float(second_peak_rate[idx]) >= self.cfg.near_parallel_second_peak_rate
                and float(second_peak_ratio[idx]) >= self.cfg.near_parallel_second_peak_ratio
            )
            if self.cfg.use_scenario_hints and hint == "local_jump":
                branch = "local_jump"
            elif self.cfg.use_scenario_hints and hint in DIFFICULT_SCENARIOS:
                branch = "all_expert"
            elif self.cfg.use_scenario_hints and near_parallel_has_two_ridges:
                branch = "multi"
            elif int(pred[idx]) > 0 and float(confidence[idx]) < self.cfg.low_confidence_all_expert_threshold:
                branch = "all_expert"
            branches.append(branch)
        return {
            "branch": branches,
            "active_pred": pred + 1,
            "active_confidence": confidence,
            "active_probs": probs,
            "second_peak_rate": second_peak_rate,
            "second_peak_ratio": second_peak_ratio,
        }


def slice_batch(batch: dict[str, Any], mask: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.shape[:1] == mask.shape[:1]:
            out[key] = value[mask]
        elif isinstance(value, list) and len(value) == mask.numel():
            keep = mask.detach().cpu().tolist()
            out[key] = [item for item, use in zip(value, keep, strict=False) if use]
        else:
            out[key] = value
    return out


def merge_sample_outputs(
    outputs: dict[int, dict[str, torch.Tensor]],
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if len(outputs) != batch_size:
        raise ValueError(f"Expected {batch_size} sample outputs, got {len(outputs)}.")
    keys = sorted(set().union(*(item.keys() for item in outputs.values())))
    merged: dict[str, torch.Tensor] = {}
    for key in keys:
        values = [outputs[idx][key] for idx in range(batch_size) if key in outputs[idx]]
        if not values:
            continue
        if values[0].ndim == 0:
            merged[key] = torch.stack([value.to(device) for value in values], dim=0)
            continue
        target_shape = list(values[0].shape)
        if key == "candidate_weights":
            target_shape[0] = max(value.shape[0] for value in values)
            padded = []
            for value in values:
                if value.shape[0] < target_shape[0]:
                    pad = value.new_zeros((target_shape[0] - value.shape[0],))
                    value = torch.cat([value, pad], dim=0)
                padded.append(value.to(device))
            merged[key] = torch.stack(padded, dim=0)
            continue
        if not all(tuple(value.shape) == tuple(values[0].shape) for value in values):
            continue
        merged[key] = torch.stack([value.to(device) for value in values], dim=0)
    return merged


def topk_candidate_weights(weights: torch.Tensor | None, k: int = 2) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if weights is None:
        return None, None
    masked = torch.nan_to_num(weights, nan=-1.0)
    count = min(int(k), masked.shape[1])
    values, indices = masked.topk(k=count, dim=1)
    values = values.clamp_min(0.0)
    if count < k:
        pad_v = values.new_zeros((values.shape[0], k - count))
        pad_i = indices.new_full((indices.shape[0], k - count), -1)
        values = torch.cat([values, pad_v], dim=1)
        indices = torch.cat([indices, pad_i], dim=1)
    return values, indices


def identity_stable_if(if_hz: torch.Tensor) -> torch.Tensor:
    if if_hz.ndim != 3 or if_hz.shape[1] < 2:
        return if_hz
    stable = if_hz.clone()
    for bidx in range(if_hz.shape[0]):
        prev = stable[bidx, :, 0]
        for tidx in range(1, if_hz.shape[-1]):
            current = stable[bidx, :, tidx].clone()
            keep_cost = (current - prev).abs().sum()
            swap = torch.flip(current, dims=(0,))
            swap_cost = (swap - prev).abs().sum()
            if swap_cost < keep_cost:
                stable[bidx, :, tidx] = swap
                current = swap
            prev = current
    return stable


def _select_sample(value: torch.Tensor, index: int) -> torch.Tensor:
    if value.ndim == 0:
        return value
    return value[index]
