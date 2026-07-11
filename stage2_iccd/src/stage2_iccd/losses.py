from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F


def reconstruction_snr_db(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    signal_power = reference.pow(2).mean(dim=-1).clamp_min(1.0e-9)
    noise_power = (reference - estimate).pow(2).mean(dim=-1).clamp_min(1.0e-9)
    return 10.0 * torch.log10(signal_power / noise_power)


def component_permutation_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[:2] != target.shape[:2]:
        raise ValueError("Predicted and target components must share [B, Q].")
    bsz, q, _ = pred.shape
    cost = torch.empty((bsz, q, q), device=pred.device, dtype=pred.dtype)
    for pred_idx in range(q):
        cost[:, pred_idx, :] = (pred[:, pred_idx : pred_idx + 1] - target).pow(2).mean(dim=-1)
    return _min_perm(cost)


def active_component_permutation_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor | None = None,
    inactive_weight: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Permutation component MSE with inactive-slot energy suppression.

    Active targets are matched one-to-one to predicted slots. Predicted slots
    assigned to inactive targets are not ignored; their energy is penalized so a
    one-component signal is not silently split into two reconstructed outputs.
    """

    if pred.shape[:2] != target.shape[:2]:
        raise ValueError("Predicted and target components must share [B, Q].")
    if active_mask is None:
        return component_permutation_mse(pred, target), pred.new_tensor(0.0)
    active_mask = active_mask.to(device=pred.device, dtype=pred.dtype)
    if active_mask.shape != pred.shape[:2]:
        raise ValueError(f"active_mask shape {tuple(active_mask.shape)} does not match [B,Q]={tuple(pred.shape[:2])}")

    bsz, q, _ = pred.shape
    comp_cost = torch.empty((bsz, q, q), device=pred.device, dtype=pred.dtype)
    for pred_idx in range(q):
        comp_cost[:, pred_idx, :] = (pred[:, pred_idx : pred_idx + 1] - target).pow(2).mean(dim=-1)
    pred_energy = pred.pow(2).mean(dim=-1)
    active_loss, inactive_loss = _min_active_perm(
        comp_cost,
        pred_energy,
        active_mask,
        inactive_match_weight=float(inactive_weight),
    )
    return active_loss + float(inactive_weight) * inactive_loss, inactive_loss


def component_permutation_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bsz, q, _ = pred.shape
    cost = torch.empty((bsz, q, q), device=pred.device, dtype=pred.dtype)
    for pred_idx in range(q):
        cost[:, pred_idx, :] = (pred[:, pred_idx : pred_idx + 1] - target).abs().mean(dim=-1)
    return _min_perm(cost)


def active_component_permutation_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if active_mask is None:
        return component_permutation_l1(pred, target)
    active_mask = active_mask.to(device=pred.device, dtype=pred.dtype)
    bsz, q, _ = pred.shape
    cost = torch.empty((bsz, q, q), device=pred.device, dtype=pred.dtype)
    for pred_idx in range(q):
        cost[:, pred_idx, :] = (pred[:, pred_idx : pred_idx + 1] - target).abs().mean(dim=-1)
    pred_energy = pred.abs().mean(dim=-1)
    active_loss, inactive_loss = _min_active_perm(cost, pred_energy, active_mask)
    return active_loss + inactive_loss


def if_smoothness(if_hz: torch.Tensor) -> torch.Tensor:
    if if_hz.shape[-1] < 3:
        return if_hz.new_tensor(0.0)
    second = if_hz[..., 2:] - 2.0 * if_hz[..., 1:-1] + if_hz[..., :-2]
    return second.pow(2).mean()


def if_third_derivative(if_hz: torch.Tensor) -> torch.Tensor:
    if if_hz.shape[-1] < 4:
        return if_hz.new_tensor(0.0)
    third = if_hz[..., 3:] - 3.0 * if_hz[..., 2:-1] + 3.0 * if_hz[..., 1:-2] - if_hz[..., :-3]
    return third.pow(2).mean()


def crossing_identity_loss(if_hz: torch.Tensor, gap_sigma_hz: float = 24.0) -> torch.Tensor:
    """Local identity-continuity penalty near two-ridge crossings.

    It does not force a fixed frequency order. Instead, it focuses on the
    short time span where component gaps are small and discourages abrupt
    velocity reversals on each output slot.
    """

    if if_hz.ndim != 3 or if_hz.shape[1] < 2 or if_hz.shape[-1] < 4:
        return if_hz.new_tensor(0.0)
    first = if_hz[:, :, 1:] - if_hz[:, :, :-1]
    previous = first[:, :, :-1]
    current = first[:, :, 1:]
    product = previous * current
    gap = (if_hz[:, 0] - if_hz[:, 1]).abs()
    near = torch.exp(-0.5 * (gap[:, 1:-1] / max(float(gap_sigma_hz), 1.0e-6)).pow(2))
    reversal = torch.relu(-product).sum(dim=1)
    accel = (current - previous).pow(2).sum(dim=1)
    return ((reversal + 0.15 * accel) * near).sum() / near.sum().clamp_min(1.0)


def min_gap_barrier(if_hz: torch.Tensor, min_gap_hz: float = 8.0) -> torch.Tensor:
    if if_hz.ndim != 3 or if_hz.shape[1] < 2:
        return if_hz.new_tensor(0.0)
    penalties = []
    for left in range(if_hz.shape[1]):
        for right in range(left + 1, if_hz.shape[1]):
            gap = (if_hz[:, left] - if_hz[:, right]).abs()
            penalties.append(torch.relu(float(min_gap_hz) - gap).pow(2).mean())
    return torch.stack(penalties).mean() if penalties else if_hz.new_tensor(0.0)


def sinusoidal_curvature_consistency(if_hz: torch.Tensor) -> torch.Tensor:
    """Weak periodic-FM prior: curvature energy should not be bursty."""

    if if_hz.shape[-1] < 5:
        return if_hz.new_tensor(0.0)
    second = if_hz[..., 2:] - 2.0 * if_hz[..., 1:-1] + if_hz[..., :-2]
    energy = second.pow(2)
    centered = energy - energy.mean(dim=-1, keepdim=True)
    return centered.pow(2).mean()


def candidate_entropy(weights: torch.Tensor) -> torch.Tensor:
    probs = weights.clamp_min(1.0e-8)
    return -(probs * probs.log()).sum()


def _min_perm(cost: torch.Tensor) -> torch.Tensor:
    bsz, q, _ = cost.shape
    perms = list(itertools.permutations(range(q)))
    rows = torch.arange(q, device=cost.device)
    values = [cost[:, rows, torch.tensor(perm, device=cost.device)].sum(dim=1) for perm in perms]
    return torch.stack(values, dim=1).min(dim=1).values.mean() / q


def _min_active_perm(
    cost: torch.Tensor,
    pred_energy: torch.Tensor,
    active_mask: torch.Tensor,
    inactive_match_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, q, _ = cost.shape
    perms = list(itertools.permutations(range(q)))
    rows = torch.arange(q, device=cost.device)
    active_terms = []
    inactive_terms = []
    for perm in perms:
        perm_tensor = torch.tensor(perm, device=cost.device)
        target_active = active_mask[:, perm_tensor]
        matched = cost[:, rows, perm_tensor]
        active_denom = target_active.sum(dim=1).clamp_min(1.0)
        inactive = 1.0 - target_active
        inactive_denom = inactive.sum(dim=1).clamp_min(1.0)
        active_terms.append((matched * target_active).sum(dim=1) / active_denom)
        inactive_terms.append((pred_energy * inactive).sum(dim=1) / inactive_denom)
    active_stack = torch.stack(active_terms, dim=1)
    inactive_stack = torch.stack(inactive_terms, dim=1)
    total = active_stack + float(inactive_match_weight) * inactive_stack
    best = total.argmin(dim=1)
    batch_idx = torch.arange(bsz, device=cost.device)
    return active_stack[batch_idx, best].mean(), inactive_stack[batch_idx, best].mean()
