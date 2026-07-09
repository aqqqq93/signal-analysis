from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


STAGE2_QUALITY_FEATURES = (
    "default_observed_snr",
    "specialist_observed_snr",
    "snr_delta",
    "default_residual_mse",
    "specialist_residual_mse",
    "residual_ratio",
    "default_delta_rms",
    "specialist_delta_rms",
    "delta_rms_diff",
    "default_smooth",
    "specialist_smooth",
    "smooth_diff",
    "default_curve_range",
    "specialist_curve_range",
    "range_diff",
    "default_curvature",
    "specialist_curvature",
    "curvature_diff",
    "if_mean_abs_diff",
    "if_max_abs_diff",
    "if_std_abs_diff",
    "default_candidate_entropy",
    "specialist_candidate_entropy",
    "candidate_entropy_diff",
)

STAGE2_QUALITY_FEATURE_DIM = len(STAGE2_QUALITY_FEATURES)


@dataclass
class Stage2QualitySelectorConfig:
    hidden: int = 64
    dropout: float = 0.08


def stage2_quality_selector_config_from_dict(data: dict | None) -> Stage2QualitySelectorConfig:
    data = data or {}
    return Stage2QualitySelectorConfig(
        hidden=int(data.get("hidden", 64)),
        dropout=float(data.get("dropout", 0.08)),
    )


class Stage2QualitySelector(nn.Module):
    """Choose between default and specialist Stage2 branches."""

    def __init__(self, cfg: Stage2QualitySelectorConfig, feature_dim: int = STAGE2_QUALITY_FEATURE_DIM):
        super().__init__()
        hidden = max(8, int(cfg.hidden))
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


@torch.no_grad()
def stage2_quality_features(
    default_out: dict[str, torch.Tensor],
    specialist_out: dict[str, torch.Tensor],
    signal: torch.Tensor,
) -> torch.Tensor:
    """Return deployment-safe features for branch selection.

    The feature vector uses only the observed signal and model outputs, never
    ground-truth IF or clean components. Labels for training are built outside
    this function from simulation targets.
    """

    default_stats = _branch_stats(default_out, signal)
    specialist_stats = _branch_stats(specialist_out, signal)
    if_diff = (default_out["refined_if_hz"] - specialist_out["refined_if_hz"]).abs()
    entropy_default = _candidate_entropy(default_out.get("candidate_weights"))
    entropy_specialist = _candidate_entropy(specialist_out.get("candidate_weights"))
    residual_ratio = default_stats["residual_mse"] / specialist_stats["residual_mse"].clamp_min(1.0e-8)
    features = [
        default_stats["observed_snr"] / 40.0,
        specialist_stats["observed_snr"] / 40.0,
        (specialist_stats["observed_snr"] - default_stats["observed_snr"]) / 20.0,
        torch.log1p(default_stats["residual_mse"]),
        torch.log1p(specialist_stats["residual_mse"]),
        torch.log(residual_ratio.clamp_min(1.0e-8)),
        default_stats["delta_rms"] / 64.0,
        specialist_stats["delta_rms"] / 64.0,
        (specialist_stats["delta_rms"] - default_stats["delta_rms"]) / 64.0,
        default_stats["smooth"] / 128.0,
        specialist_stats["smooth"] / 128.0,
        (specialist_stats["smooth"] - default_stats["smooth"]) / 128.0,
        default_stats["curve_range"] / 512.0,
        specialist_stats["curve_range"] / 512.0,
        (specialist_stats["curve_range"] - default_stats["curve_range"]) / 512.0,
        default_stats["curvature"] / 128.0,
        specialist_stats["curvature"] / 128.0,
        (specialist_stats["curvature"] - default_stats["curvature"]) / 128.0,
        if_diff.mean(dim=(1, 2)) / 128.0,
        if_diff.amax(dim=(1, 2)) / 256.0,
        if_diff.std(dim=(1, 2)) / 128.0,
        entropy_default,
        entropy_specialist,
        entropy_specialist - entropy_default,
    ]
    return torch.stack([item.reshape(signal.shape[0]) for item in features], dim=1)


def _branch_stats(out: dict[str, torch.Tensor], signal: torch.Tensor) -> dict[str, torch.Tensor]:
    rec = out["reconstruction"]
    refined_if = out["refined_if_hz"]
    residual_mse = (signal - rec).pow(2).mean(dim=-1)
    energy = signal.pow(2).mean(dim=-1).clamp_min(1.0e-12)
    observed_snr = 10.0 * torch.log10(energy / residual_mse.clamp_min(1.0e-12))
    delta_rms = torch.sqrt(out["delta_if_hz"].pow(2).mean(dim=(1, 2)).clamp_min(1.0e-12))
    slope = refined_if[..., 1:] - refined_if[..., :-1]
    if refined_if.shape[-1] >= 3:
        second = refined_if[..., 2:] - 2.0 * refined_if[..., 1:-1] + refined_if[..., :-2]
        curvature = second.abs().mean(dim=(1, 2))
    else:
        curvature = refined_if.new_zeros(refined_if.shape[0])
    return {
        "observed_snr": observed_snr,
        "residual_mse": residual_mse,
        "delta_rms": delta_rms,
        "smooth": slope.abs().mean(dim=(1, 2)),
        "curve_range": (refined_if.amax(dim=-1) - refined_if.amin(dim=-1)).mean(dim=1),
        "curvature": curvature,
    }


def _candidate_entropy(weights: torch.Tensor | None) -> torch.Tensor:
    if weights is None:
        raise ValueError("Stage2 output does not contain candidate_weights.")
    probs = weights.clamp_min(1.0e-8)
    return -(probs * probs.log()).sum(dim=1)
