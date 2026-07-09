from __future__ import annotations

from pathlib import Path

import torch

from ifnet_stage1.router import HardRouteClassifier, ROUTE_NAMES, router_config_from_dict
from ifnet_stage1.simulation import sim_config_from_dict
from ifnet_stage1.tf import feature_channels, log_spectrogram, stft_config_from_dict

from .active_count import ActiveCountClassifier, active_count_config_from_dict


class QualityContextProvider:
    """Optional context features for the Stage2 branch quality selector."""

    def __init__(
        self,
        device: torch.device,
        stage1_router_checkpoint: str | Path | None = None,
        active_count_checkpoint: str | Path | None = None,
    ):
        self.device = device
        self.router = _LoadedRouter(stage1_router_checkpoint, device) if stage1_router_checkpoint else None
        self.active_count = _LoadedActiveCount(active_count_checkpoint, device) if active_count_checkpoint else None

    @torch.no_grad()
    def __call__(self, signal: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if self.router is not None:
            out.update(self.router(signal))
        if self.active_count is not None:
            out.update(self.active_count(signal))
        return out


class _LoadedRouter:
    def __init__(self, checkpoint: str | Path, device: torch.device):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        self.sim_cfg = sim_config_from_dict(cfg["data"])
        self.stft_cfg = stft_config_from_dict(cfg["stft"])
        route_names = tuple(ckpt.get("route_names", cfg.get("route_names", ROUTE_NAMES)))
        self.route_names = route_names
        model_cfg = router_config_from_dict(cfg["router"])
        self.model = HardRouteClassifier(feature_channels(self.stft_cfg), len(route_names), model_cfg).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, signal: torch.Tensor) -> dict[str, torch.Tensor]:
        feats, _ = log_spectrogram(signal, self.stft_cfg, self.sim_cfg.fs)
        probs = torch.softmax(self.model(feats), dim=1)
        top2 = probs.topk(k=min(2, probs.shape[1]), dim=1).values
        margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else torch.ones_like(top2[:, 0])
        values = {
            "stage1_top1_confidence": top2[:, 0],
            "stage1_top2_margin": margin,
        }
        route_to_idx = {name: idx for idx, name in enumerate(self.route_names)}
        for route_name, feature_name in (
            ("poly_like", "stage1_poly_prob"),
            ("sinusoidal_like", "stage1_sinusoidal_prob"),
            ("cross_overlap_like", "stage1_cross_prob"),
            ("jump_like", "stage1_jump_prob"),
        ):
            idx = route_to_idx.get(route_name)
            values[feature_name] = probs[:, idx] if idx is not None else probs.new_zeros(probs.shape[0])
        return values


class _LoadedActiveCount:
    def __init__(self, checkpoint: str | Path, device: torch.device):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        self.sim_cfg = sim_config_from_dict(cfg["data"])
        self.stft_cfg = stft_config_from_dict(cfg["stft"])
        model_cfg = active_count_config_from_dict(ckpt.get("model_cfg", cfg.get("active_count")))
        self.model = ActiveCountClassifier(feature_channels(self.stft_cfg), model_cfg).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, signal: torch.Tensor) -> dict[str, torch.Tensor]:
        feats, _ = log_spectrogram(signal, self.stft_cfg, self.sim_cfg.fs)
        probs = torch.softmax(self.model(feats), dim=1)
        top2 = probs.topk(k=min(2, probs.shape[1]), dim=1).values
        margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else torch.ones_like(top2[:, 0])
        return {
            "active_count_confidence": top2[:, 0],
            "active_count_margin": margin,
            "active_count_two_prob": probs[:, 1] if probs.shape[1] > 1 else probs.new_zeros(probs.shape[0]),
        }
