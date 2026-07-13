from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .differentiable_iccd import DifferentiableICCD, ICCDConfig


@dataclass
class Stage2ModelConfig:
    num_candidates: int = 2
    refine_channels: int = 32
    refine_layers: int = 3
    candidate_fusion: str = "residual"
    candidate_feature_dim: int = 8
    candidate_hidden: int = 24
    refinement_mode: str = "standard"
    refine_extra_channels: int = 0
    max_refine_hz: float = 35.0
    max_jump_refine_hz: float = 0.0
    freq_min: float = 35.0
    freq_max: float = 430.0
    candidate_temperature: float = 0.02
    candidate_temperature_min: float = 0.001


def stage2_model_config_from_dict(data: dict) -> Stage2ModelConfig:
    return Stage2ModelConfig(
        num_candidates=int(data.get("num_candidates", 2)),
        refine_channels=int(data.get("refine_channels", 32)),
        refine_layers=int(data.get("refine_layers", 3)),
        candidate_fusion=str(data.get("candidate_fusion", "residual")),
        candidate_feature_dim=int(data.get("candidate_feature_dim", 8)),
        candidate_hidden=int(data.get("candidate_hidden", 24)),
        refinement_mode=str(data.get("refinement_mode", "standard")),
        refine_extra_channels=int(data.get("refine_extra_channels", 0)),
        max_refine_hz=float(data.get("max_refine_hz", 35.0)),
        max_jump_refine_hz=float(data.get("max_jump_refine_hz", data.get("max_refine_hz", 35.0))),
        freq_min=float(data.get("freq_min", 35.0)),
        freq_max=float(data.get("freq_max", 430.0)),
        candidate_temperature=float(data.get("candidate_temperature", 0.02)),
        candidate_temperature_min=float(data.get("candidate_temperature_min", 0.001)),
    )


class CandidateMixer(nn.Module):
    """Soft selector for top-k IF candidates.

    When an ICCD layer and signal are provided, candidates are scored by their
    preliminary reconstruction residual. This avoids averaging incompatible
    expert IF tracks. The trainable bias terms still let the model learn a
    persistent preference for stronger candidate sources.
    """

    def __init__(
        self,
        num_candidates: int,
        temperature: float,
        temperature_min: float,
        fusion: str = "residual",
        feature_dim: int = 8,
        hidden: int = 24,
    ):
        super().__init__()
        if num_candidates < 1:
            raise ValueError("num_candidates must be at least 1.")
        self.fusion = str(fusion)
        if self.fusion not in {"first", "bias", "residual", "feature_attention"}:
            raise ValueError(f"Unknown candidate fusion mode: {self.fusion}")
        self.bias = nn.Parameter(torch.zeros(num_candidates))
        self.raw_temperature = nn.Parameter(_inverse_softplus(torch.tensor(max(temperature - temperature_min, 1.0e-6))))
        self.temperature_min = float(temperature_min)
        self.feature_dim = max(1, int(feature_dim))
        self.quality_net = (
            nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, int(hidden)),
                nn.SiLU(),
                nn.Linear(int(hidden), 1),
            )
            if self.fusion == "feature_attention"
            else None
        )

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature) + self.temperature_min

    def forward(
        self,
        candidate_if_hz: torch.Tensor,
        signal: torch.Tensor | None = None,
        iccd: DifferentiableICCD | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_if_hz.ndim != 4:
            raise ValueError(f"Expected candidates [B, C, Q, N], got {tuple(candidate_if_hz.shape)}")
        num_candidates = candidate_if_hz.shape[1]
        bias = self.bias[:num_candidates]
        if self.fusion == "first":
            weights = candidate_if_hz.new_zeros((candidate_if_hz.shape[0], num_candidates))
            weights[:, 0] = 1.0
        elif self.fusion == "bias" or signal is None or iccd is None:
            weights = torch.softmax(bias, dim=0).view(1, -1).expand(candidate_if_hz.shape[0], -1)
        else:
            residual_score = self._residual_scores(candidate_if_hz, signal, iccd)
            logits = bias.view(1, -1) - residual_score / self.temperature.clamp_min(self.temperature_min)
            if self.quality_net is not None:
                features = self._candidate_features(candidate_if_hz, residual_score)
                quality = self.quality_net(features.reshape(-1, features.shape[-1])).view(features.shape[:2])
                logits = logits + quality
            weights = torch.softmax(logits, dim=1)
        mixed = (candidate_if_hz * weights.view(candidate_if_hz.shape[0], num_candidates, 1, 1)).sum(dim=1)
        return mixed, weights

    @staticmethod
    def _residual_scores(
        candidate_if_hz: torch.Tensor,
        signal: torch.Tensor,
        iccd: DifferentiableICCD,
    ) -> torch.Tensor:
        residual_scores = []
        with torch.no_grad():
            for idx in range(candidate_if_hz.shape[1]):
                out = iccd(signal, candidate_if_hz[:, idx])
                residual = (out["reconstruction"] - signal).pow(2).mean(dim=-1)
                residual_scores.append(residual)
        return torch.stack(residual_scores, dim=1)

    def _candidate_features(self, candidate_if_hz: torch.Tensor, residual_score: torch.Tensor) -> torch.Tensor:
        bsz, num_candidates, _num_components, _n = candidate_if_hz.shape
        first = candidate_if_hz[..., 1:] - candidate_if_hz[..., :-1]
        second = first[..., 1:] - first[..., :-1] if first.shape[-1] > 1 else first.new_zeros(first.shape)
        center = candidate_if_hz.mean(dim=1, keepdim=True)
        dist_to_center = (candidate_if_hz - center).abs().mean(dim=(-1, -2))
        stats = torch.stack(
            [
                residual_score,
                residual_score / residual_score.mean(dim=1, keepdim=True).clamp_min(1.0e-8),
                candidate_if_hz.mean(dim=(-1, -2)) / 500.0,
                candidate_if_hz.std(dim=(-1, -2)) / 500.0,
                candidate_if_hz.amax(dim=(-1, -2)) / 500.0,
                candidate_if_hz.amin(dim=(-1, -2)) / 500.0,
                first.abs().mean(dim=(-1, -2)) / 50.0,
                second.abs().amax(dim=(-1, -2)) / 50.0,
                dist_to_center / 100.0,
            ],
            dim=-1,
        )
        if stats.shape[-1] < self.feature_dim:
            pad = stats.new_zeros((bsz, num_candidates, self.feature_dim - stats.shape[-1]))
            stats = torch.cat([stats, pad], dim=-1)
        return stats[..., : self.feature_dim]


class IFRefinementHead(nn.Module):
    """Small 1D refinement head for residual IF correction."""

    def __init__(
        self,
        num_components: int,
        channels: int,
        layers: int,
        max_refine_hz: float,
        extra_channels: int = 0,
        mode: str = "standard",
        max_jump_refine_hz: float | None = None,
    ):
        super().__init__()
        self.num_components = int(num_components)
        self.max_refine_hz = float(max_refine_hz)
        self.max_jump_refine_hz = float(max_refine_hz if max_jump_refine_hz is None else max_jump_refine_hz)
        self.extra_channels = max(0, int(extra_channels))
        self.mode = str(mode)
        if self.mode not in {"standard", "segmented"}:
            raise ValueError(f"Unknown refinement mode: {self.mode}")
        hidden = max(4, int(channels))
        in_channels = num_components + 1 + self.extra_channels
        self.net = self._make_net(in_channels, hidden, layers, num_components)
        self.jump_net = self._make_net(in_channels, hidden, layers, num_components) if self.mode == "segmented" else None

    @staticmethod
    def _make_net(in_channels: int, hidden: int, layers: int, num_components: int) -> nn.Sequential:
        blocks: list[nn.Module] = [
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.SiLU(),
        ]
        for _ in range(max(0, int(layers) - 1)):
            blocks.extend(
                [
                    nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
                    nn.GroupNorm(1, hidden),
                    nn.SiLU(),
                ]
            )
        blocks.append(nn.Conv1d(hidden, num_components, kernel_size=3, padding=1))
        return nn.Sequential(*blocks)

    def forward(self, signal: torch.Tensor, if_hz: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        features = self._features(signal, if_hz, extra)
        smooth_delta = self.max_refine_hz * torch.tanh(self.net(features))
        if self.jump_net is None:
            return smooth_delta
        jump_delta = self.max_jump_refine_hz * torch.tanh(self.jump_net(features))
        gate = self._jump_gate(signal, extra)
        return smooth_delta * (1.0 - gate) + jump_delta * gate

    def _features(self, signal: torch.Tensor, if_hz: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        signal_norm = signal / signal.std(dim=-1, keepdim=True).clamp_min(1.0e-6)
        if_norm = if_hz / if_hz.detach().amax(dim=-1, keepdim=True).clamp_min(1.0)
        parts = [if_norm, signal_norm.unsqueeze(1)]
        if self.extra_channels > 0:
            if extra is None:
                extra = signal.new_zeros((signal.shape[0], self.extra_channels, signal.shape[-1]))
            if extra.shape[0] != signal.shape[0] or extra.shape[-1] != signal.shape[-1]:
                raise ValueError(f"Expected refinement extra [B,C,N], got {tuple(extra.shape)}")
            if extra.shape[1] != self.extra_channels:
                raise ValueError(f"Expected {self.extra_channels} refinement extra channels, got {extra.shape[1]}")
            parts.append(extra)
        return torch.cat(parts, dim=1)

    def _jump_gate(self, signal: torch.Tensor, extra: torch.Tensor | None) -> torch.Tensor:
        if extra is None or extra.shape[1] <= 0:
            return signal.new_zeros((signal.shape[0], self.num_components, signal.shape[-1]))
        gate = extra[:, : min(self.num_components, extra.shape[1])].clamp(0.0, 1.0)
        if gate.shape[1] < self.num_components:
            pad = gate.new_zeros((gate.shape[0], self.num_components - gate.shape[1], gate.shape[-1]))
            gate = torch.cat([gate, pad], dim=1)
        return gate


class Stage2ICCDModel(nn.Module):
    """Frozen IF candidates -> trainable mixer/refinement -> differentiable ICCD."""

    def __init__(self, iccd_cfg: ICCDConfig, model_cfg: Stage2ModelConfig, num_components: int):
        super().__init__()
        self.model_cfg = model_cfg
        self.mixer = CandidateMixer(
            model_cfg.num_candidates,
            temperature=model_cfg.candidate_temperature,
            temperature_min=model_cfg.candidate_temperature_min,
            fusion=model_cfg.candidate_fusion,
            feature_dim=model_cfg.candidate_feature_dim,
            hidden=model_cfg.candidate_hidden,
        )
        self.refine_head = IFRefinementHead(
            num_components=num_components,
            channels=model_cfg.refine_channels,
            layers=model_cfg.refine_layers,
            max_refine_hz=model_cfg.max_refine_hz,
            extra_channels=model_cfg.refine_extra_channels,
            mode=model_cfg.refinement_mode,
            max_jump_refine_hz=model_cfg.max_jump_refine_hz,
        )
        self.iccd = DifferentiableICCD(iccd_cfg)

    def forward(
        self,
        signal: torch.Tensor,
        candidate_if_hz: torch.Tensor,
        refinement_extra: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        initial_if, candidate_weights = self.mixer(candidate_if_hz, signal=signal, iccd=self.iccd)
        delta_if = self.refine_head(signal, initial_if, extra=refinement_extra)
        refined_if = (initial_if + delta_if).clamp(float(self.model_cfg.freq_min), float(self.model_cfg.freq_max))
        iccd_out = self.iccd(signal, refined_if)
        iccd_out.update(
            {
                "initial_if_hz": initial_if,
                "delta_if_hz": delta_if,
                "refined_if_hz": refined_if,
                "candidate_weights": candidate_weights,
                "candidate_temperature": self.mixer.temperature,
            }
        )
        return iccd_out


def make_smooth_candidate(if_hz: torch.Tensor, kernel_size: int = 31) -> torch.Tensor:
    if kernel_size <= 1:
        return if_hz
    kernel_size = int(kernel_size) | 1
    pad = kernel_size // 2
    weight = torch.hann_window(kernel_size, device=if_hz.device, dtype=if_hz.dtype)
    weight = weight / weight.sum().clamp_min(1.0e-8)
    bsz, q, n_samples = if_hz.shape
    smoothed = F.conv1d(
        F.pad(if_hz.reshape(bsz * q, 1, n_samples), (pad, pad), mode="replicate"),
        weight.view(1, 1, -1),
    )
    return smoothed.view(bsz, q, n_samples)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value).clamp_min(1.0e-8))
