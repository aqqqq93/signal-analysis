from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .routing_policy import (
    heatmap_entropy,
    jump_evidence,
    jump_mismatch,
    normalized_poly_residual,
    normalized_sine_residual,
    smoothness_mismatch,
)


QUALITY_FEATURE_DIM = 27


@dataclass
class QualitySelectorConfig:
    hidden: int = 64
    dropout: float = 0.08


def quality_selector_config_from_dict(data: dict | None) -> QualitySelectorConfig:
    data = data or {}
    return QualitySelectorConfig(
        hidden=int(data.get("hidden", 64)),
        dropout=float(data.get("dropout", 0.08)),
    )


class QualitySelector(nn.Module):
    """Small scorer that ranks candidate expert outputs by expected IF quality."""

    def __init__(self, cfg: QualitySelectorConfig, feature_dim: int = QUALITY_FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, cfg.hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(cfg.hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


@torch.no_grad()
def score_candidates_with_quality(
    selector: QualitySelector,
    route_probs: torch.Tensor,
    candidates: list[tuple[int, str, torch.Tensor, torch.Tensor]],
    route_names: tuple[str, ...],
) -> dict[str, float]:
    """Return selector scores keyed by route name for candidate diagnostics."""

    if not candidates:
        return {}
    features = torch.stack(
        [
            candidate_quality_features(
                route_probs,
                route_idx,
                pred_if,
                ridge_probs,
                num_routes=len(route_names),
            )
            for route_idx, _route_name, pred_if, ridge_probs in candidates
        ],
        dim=0,
    )
    scores = selector(features)
    return {
        route_name: float(score.detach().cpu())
        for score, (_route_idx, route_name, _pred_if, _ridge_probs) in zip(scores, candidates, strict=False)
    }


@torch.no_grad()
def select_candidate_with_quality(
    selector: QualitySelector,
    route_probs: torch.Tensor,
    candidates: list[tuple[int, str, torch.Tensor, torch.Tensor]],
    route_names: tuple[str, ...],
    margin: float = 0.10,
    protect_top_routes: set[str] | None = None,
    protect_min_prob: float = 0.0,
) -> tuple[int, str]:
    protect_top_routes = protect_top_routes or set()
    top_route_idx, top_route_name = candidates[0][:2]
    if top_route_name in protect_top_routes and float(route_probs[top_route_idx].detach().cpu()) >= float(protect_min_prob):
        return top_route_idx, top_route_name

    features = torch.stack(
        [
            candidate_quality_features(
                route_probs,
                route_idx,
                pred_if,
                ridge_probs,
                num_routes=len(route_names),
            )
            for route_idx, _route_name, pred_if, ridge_probs in candidates
        ],
        dim=0,
    )
    scores = selector(features)
    best_pos = int(scores.argmax().detach().cpu())
    if best_pos != 0 and float((scores[best_pos] - scores[0]).detach().cpu()) > float(margin):
        return candidates[best_pos][0], candidates[best_pos][1]
    return top_route_idx, top_route_name


def candidate_quality_features(
    route_probs: torch.Tensor,
    route_idx: int,
    pred_if: torch.Tensor,
    ridge_probs: torch.Tensor,
    num_routes: int,
) -> torch.Tensor:
    """Return quality features for one candidate.

    route_probs: [R]
    pred_if: [1,Q,T] or [Q,T]
    ridge_probs: [1,Q,F,T] or [Q,F,T]
    """

    if route_probs.ndim != 1:
        raise ValueError(f"Expected route_probs [R], got {tuple(route_probs.shape)}")
    curves = pred_if.reshape(-1, pred_if.shape[-1])
    probs = ridge_probs if ridge_probs.ndim == 4 else ridge_probs.unsqueeze(0)
    route_onehot = route_probs.new_zeros(num_routes)
    route_onehot[route_idx] = 1.0
    top_values = torch.topk(route_probs, k=min(2, route_probs.numel())).values
    top_prob = top_values[0]
    margin = top_values[0] - top_values[1] if top_values.numel() > 1 else route_probs.new_tensor(1.0)

    slope_abs = _slope_abs(curves)
    second_abs = _second_abs(curves)
    curve_range = (curves.max(dim=1).values - curves.min(dim=1).values).mean()
    peak = probs.max(dim=-2).values

    features = [
        route_probs,
        route_onehot,
        route_probs[route_idx].view(1),
        top_prob.view(1),
        margin.view(1),
        heatmap_entropy(probs).view(1),
        normalized_poly_residual(curves).view(1),
        normalized_sine_residual(curves).view(1),
        jump_mismatch(curves).view(1),
        jump_evidence(curves).view(1),
        smoothness_mismatch(curves).view(1),
        curves.mean().view(1) / 512.0,
        curves.std().view(1) / 256.0,
        curve_range.view(1) / 512.0,
        slope_abs.mean().view(1) / 32.0,
        slope_abs.max().view(1) / 128.0,
        second_abs.mean().view(1) / 32.0,
        second_abs.max().view(1) / 128.0,
        peak.mean().view(1),
        peak.std().view(1),
        peak.min().view(1),
    ]
    return torch.cat([item.reshape(-1) for item in features], dim=0)


def _slope_abs(curves: torch.Tensor) -> torch.Tensor:
    if curves.shape[-1] < 2:
        return curves.new_zeros((curves.shape[0], 1))
    return (curves[:, 1:] - curves[:, :-1]).abs()


def _second_abs(curves: torch.Tensor) -> torch.Tensor:
    if curves.shape[-1] < 3:
        return curves.new_zeros((curves.shape[0], 1))
    return (curves[:, 2:] - 2.0 * curves[:, 1:-1] + curves[:, :-2]).abs()
