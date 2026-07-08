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


def component_permutation_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bsz, q, _ = pred.shape
    cost = torch.empty((bsz, q, q), device=pred.device, dtype=pred.dtype)
    for pred_idx in range(q):
        cost[:, pred_idx, :] = (pred[:, pred_idx : pred_idx + 1] - target).abs().mean(dim=-1)
    return _min_perm(cost)


def if_smoothness(if_hz: torch.Tensor) -> torch.Tensor:
    if if_hz.shape[-1] < 3:
        return if_hz.new_tensor(0.0)
    second = if_hz[..., 2:] - 2.0 * if_hz[..., 1:-1] + if_hz[..., :-2]
    return second.pow(2).mean()


def candidate_entropy(weights: torch.Tensor) -> torch.Tensor:
    probs = weights.clamp_min(1.0e-8)
    return -(probs * probs.log()).sum()


def _min_perm(cost: torch.Tensor) -> torch.Tensor:
    bsz, q, _ = cost.shape
    perms = list(itertools.permutations(range(q)))
    rows = torch.arange(q, device=cost.device)
    values = [cost[:, rows, torch.tensor(perm, device=cost.device)].sum(dim=1) for perm in perms]
    return torch.stack(values, dim=1).min(dim=1).values.mean() / q
