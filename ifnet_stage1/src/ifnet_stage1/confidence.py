from __future__ import annotations

import torch


def route_confidence_features(route_probs: torch.Tensor) -> dict[str, float]:
    """Summarize router confidence for one sample."""

    if route_probs.ndim != 1:
        raise ValueError(f"Expected route_probs [R], got {tuple(route_probs.shape)}")
    values = torch.topk(route_probs, k=min(2, route_probs.numel())).values
    top_prob = values[0]
    margin = values[0] - values[1] if values.numel() > 1 else route_probs.new_tensor(1.0)
    entropy = -(route_probs.clamp_min(1.0e-8) * route_probs.clamp_min(1.0e-8).log()).sum()
    entropy = entropy / torch.log(route_probs.new_tensor(float(route_probs.numel())))
    return {
        "route_top_prob": float(top_prob.detach().cpu()),
        "route_margin": float(margin.detach().cpu()),
        "route_entropy": float(entropy.detach().cpu()),
    }


def ridge_confidence_features(ridge_probs: torch.Tensor) -> dict[str, float]:
    """Summarize heatmap concentration for one candidate IF estimate."""

    probs = ridge_probs if ridge_probs.ndim == 4 else ridge_probs.unsqueeze(0)
    if probs.ndim != 4:
        raise ValueError(f"Expected ridge_probs [B,Q,F,T] or [Q,F,T], got {tuple(ridge_probs.shape)}")
    freq_bins = probs.shape[-2]
    entropy = -(probs.clamp_min(1.0e-8) * probs.clamp_min(1.0e-8).log()).sum(dim=-2)
    entropy = entropy / torch.log(probs.new_tensor(float(freq_bins)))
    peak = probs.max(dim=-2).values
    confidence = 0.65 * (1.0 - entropy.mean()) + 0.35 * peak.mean()
    return {
        "ridge_confidence": float(confidence.clamp(0.0, 1.0).detach().cpu()),
        "ridge_peak_mean": float(peak.mean().detach().cpu()),
        "ridge_peak_min": float(peak.min().detach().cpu()),
        "ridge_entropy": float(entropy.mean().detach().cpu()),
    }


def combined_initial_confidence(route_probs: torch.Tensor, ridge_probs: torch.Tensor) -> float:
    """Single conservative confidence score for passing IF estimates to stage 2."""

    route = route_confidence_features(route_probs)
    ridge = ridge_confidence_features(ridge_probs)
    score = (
        0.38 * route["route_top_prob"]
        + 0.22 * max(0.0, min(1.0, route["route_margin"] / 0.35))
        + 0.40 * ridge["ridge_confidence"]
    )
    return float(max(0.0, min(1.0, score)))
