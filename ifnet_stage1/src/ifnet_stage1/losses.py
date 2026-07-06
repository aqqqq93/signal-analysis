from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F


def _permutations(num_components: int, device: torch.device) -> torch.Tensor:
    values = list(itertools.permutations(range(num_components)))
    return torch.tensor(values, dtype=torch.long, device=device)


def pairwise_ridge_nll(
    logits: torch.Tensor,
    target_if: torch.Tensor,
    freq_grid: torch.Tensor,
    sigma_hz: float,
) -> torch.Tensor:
    """Gaussian ridge target NLL with permutation matching.

    logits: [B, Q, F, T]
    target_if: [B, Q, T]
    """

    bsz, q, _, _ = logits.shape
    log_probs = F.log_softmax(logits, dim=2)
    diff = freq_grid.view(1, 1, -1, 1) - target_if.unsqueeze(2)
    target = torch.exp(-0.5 * (diff / float(sigma_hz)).pow(2))
    target = target / target.sum(dim=2, keepdim=True).clamp_min(1.0e-8)

    # cost[b, pred_component, target_component]
    cost = torch.empty((bsz, q, q), device=logits.device, dtype=logits.dtype)
    for pred_idx in range(q):
        lp = log_probs[:, pred_idx : pred_idx + 1]
        ce = -(target * lp).sum(dim=2).mean(dim=-1)
        cost[:, pred_idx, :] = ce
    return min_permutation_cost(cost)


def permutation_l1(pred_if: torch.Tensor, target_if: torch.Tensor) -> torch.Tensor:
    bsz, q, _ = pred_if.shape
    cost = torch.empty((bsz, q, q), device=pred_if.device, dtype=pred_if.dtype)
    for pred_idx in range(q):
        cost[:, pred_idx, :] = (pred_if[:, pred_idx : pred_idx + 1] - target_if).abs().mean(dim=-1)
    return min_permutation_cost(cost)


def permutation_slope_l1(pred_if: torch.Tensor, target_if: torch.Tensor) -> torch.Tensor:
    """Permutation-matched first-difference loss for component identity continuity.

    A ridge can have low pointwise error in easy regions while jumping tracks in
    ambiguous regions. Matching first differences discourages those mid-curve
    identity swaps without requiring a fixed ordering convention.
    """

    if pred_if.shape[-1] < 2:
        return pred_if.new_tensor(0.0)
    pred_slope = pred_if[..., 1:] - pred_if[..., :-1]
    target_slope = target_if[..., 1:] - target_if[..., :-1]
    bsz, q, _ = pred_slope.shape
    cost = torch.empty((bsz, q, q), device=pred_if.device, dtype=pred_if.dtype)
    for pred_idx in range(q):
        cost[:, pred_idx, :] = (pred_slope[:, pred_idx : pred_idx + 1] - target_slope).abs().mean(dim=-1)
    return min_permutation_cost(cost)


def min_permutation_cost(cost: torch.Tensor) -> torch.Tensor:
    bsz, q, _ = cost.shape
    perms = _permutations(q, cost.device)
    per_perm = []
    pred_idx = torch.arange(q, device=cost.device)
    for perm in perms:
        per_perm.append(cost[:, pred_idx, perm].sum(dim=1))
    stacked = torch.stack(per_perm, dim=1)
    return stacked.min(dim=1).values.mean() / q


def second_difference_smoothness(pred_if: torch.Tensor) -> torch.Tensor:
    if pred_if.shape[-1] < 3:
        return pred_if.new_tensor(0.0)
    second = pred_if[..., 2:] - 2.0 * pred_if[..., 1:-1] + pred_if[..., :-2]
    return second.pow(2).mean()


def polynomial_residual(pred_if: torch.Tensor, degree: int = 3) -> torch.Tensor:
    """Mean squared residual after projecting each IF track to a polynomial.

    This is a differentiable structural prior for polynomial-like chirps. The
    data terms still decide where the ridge should be; this term only penalizes
    local wiggles that cannot be explained by a low-order polynomial.
    """

    return polynomial_residual_per_sample(pred_if, degree=degree).mean()


def polynomial_residual_per_sample(pred_if: torch.Tensor, degree: int = 3) -> torch.Tensor:
    """Per-sample polynomial residual, reduced over components and time."""

    num_frames = pred_if.shape[-1]
    if num_frames <= degree + 1:
        return pred_if.new_zeros(pred_if.shape[0])
    time = torch.linspace(-1.0, 1.0, num_frames, device=pred_if.device, dtype=pred_if.dtype)
    basis = torch.stack([time.pow(power) for power in range(degree + 1)], dim=1)
    projection = basis @ torch.linalg.pinv(basis)
    fitted = pred_if @ projection.transpose(0, 1)
    return (pred_if - fitted).pow(2).mean(dim=(1, 2))
