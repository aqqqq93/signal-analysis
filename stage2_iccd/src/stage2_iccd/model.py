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
    max_refine_hz: float = 35.0
    freq_min: float = 35.0
    freq_max: float = 430.0
    candidate_temperature: float = 0.02
    candidate_temperature_min: float = 0.001


def stage2_model_config_from_dict(data: dict) -> Stage2ModelConfig:
    return Stage2ModelConfig(
        num_candidates=int(data.get("num_candidates", 2)),
        refine_channels=int(data.get("refine_channels", 32)),
        refine_layers=int(data.get("refine_layers", 3)),
        max_refine_hz=float(data.get("max_refine_hz", 35.0)),
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

    def __init__(self, num_candidates: int, temperature: float, temperature_min: float):
        super().__init__()
        if num_candidates < 1:
            raise ValueError("num_candidates must be at least 1.")
        self.bias = nn.Parameter(torch.zeros(num_candidates))
        self.raw_temperature = nn.Parameter(_inverse_softplus(torch.tensor(max(temperature - temperature_min, 1.0e-6))))
        self.temperature_min = float(temperature_min)

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
        if signal is not None and iccd is not None:
            residual_scores = []
            with torch.no_grad():
                for idx in range(num_candidates):
                    out = iccd(signal, candidate_if_hz[:, idx])
                    residual = (out["reconstruction"] - signal).pow(2).mean(dim=-1)
                    residual_scores.append(residual)
            residual_score = torch.stack(residual_scores, dim=1)
            logits = bias.view(1, -1) - residual_score / self.temperature.clamp_min(self.temperature_min)
            weights = torch.softmax(logits, dim=1)
        else:
            weights = torch.softmax(bias, dim=0).view(1, -1).expand(candidate_if_hz.shape[0], -1)
        mixed = (candidate_if_hz * weights.view(candidate_if_hz.shape[0], num_candidates, 1, 1)).sum(dim=1)
        return mixed, weights


class IFRefinementHead(nn.Module):
    """Small 1D refinement head for residual IF correction."""

    def __init__(self, num_components: int, channels: int, layers: int, max_refine_hz: float):
        super().__init__()
        self.max_refine_hz = float(max_refine_hz)
        hidden = max(4, int(channels))
        blocks: list[nn.Module] = [
            nn.Conv1d(num_components + 1, hidden, kernel_size=7, padding=3),
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
        self.net = nn.Sequential(*blocks)

    def forward(self, signal: torch.Tensor, if_hz: torch.Tensor) -> torch.Tensor:
        signal_norm = signal / signal.std(dim=-1, keepdim=True).clamp_min(1.0e-6)
        if_norm = if_hz / if_hz.detach().amax(dim=-1, keepdim=True).clamp_min(1.0)
        features = torch.cat([if_norm, signal_norm.unsqueeze(1)], dim=1)
        return self.max_refine_hz * torch.tanh(self.net(features))


class Stage2ICCDModel(nn.Module):
    """Frozen IF candidates -> trainable mixer/refinement -> differentiable ICCD."""

    def __init__(self, iccd_cfg: ICCDConfig, model_cfg: Stage2ModelConfig, num_components: int):
        super().__init__()
        self.model_cfg = model_cfg
        self.mixer = CandidateMixer(
            model_cfg.num_candidates,
            temperature=model_cfg.candidate_temperature,
            temperature_min=model_cfg.candidate_temperature_min,
        )
        self.refine_head = IFRefinementHead(
            num_components=num_components,
            channels=model_cfg.refine_channels,
            layers=model_cfg.refine_layers,
            max_refine_hz=model_cfg.max_refine_hz,
        )
        self.iccd = DifferentiableICCD(iccd_cfg)

    def forward(self, signal: torch.Tensor, candidate_if_hz: torch.Tensor) -> dict[str, torch.Tensor]:
        initial_if, candidate_weights = self.mixer(candidate_if_hz, signal=signal, iccd=self.iccd)
        delta_if = self.refine_head(signal, initial_if)
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
