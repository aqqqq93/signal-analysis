from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def polynomial_project_if(
    pred_if: torch.Tensor,
    degree: int = 3,
    robust_iters: int = 2,
    huber_hz: float = 12.0,
) -> torch.Tensor:
    """Project IF curves onto low-order polynomial tracks.

    This is intended for polynomial-like IF families (linear, quadratic, cubic).
    It removes short local ridge jumps while preserving the global chirp trend.

    pred_if: [B, Q, T] in Hz
    """

    if pred_if.ndim != 3:
        raise ValueError(f"Expected pred_if [B, Q, T], got {tuple(pred_if.shape)}")
    degree = max(0, int(degree))
    robust_iters = max(0, int(robust_iters))
    bsz, q, frames = pred_if.shape
    if frames <= degree + 1:
        return pred_if.clone()

    x = torch.linspace(-1.0, 1.0, frames, device=pred_if.device, dtype=pred_if.dtype)
    design = torch.stack([x.pow(k) for k in range(degree + 1)], dim=1)  # [T, D]
    out = torch.empty_like(pred_if)

    for b in range(bsz):
        for c in range(q):
            y = pred_if[b, c]
            weights = torch.ones_like(y)
            fitted = y
            for _ in range(robust_iters + 1):
                sqrt_w = weights.sqrt().unsqueeze(1)
                aw = design * sqrt_w
                yw = y * weights.sqrt()
                coeff = torch.linalg.lstsq(aw, yw).solution
                fitted = design @ coeff
                residual = (y - fitted).abs()
                weights = torch.clamp(float(huber_hz) / residual.clamp_min(1.0e-6), max=1.0)
            out[b, c] = fitted
    return out


@torch.no_grad()
def polynomial_project_from_probs(
    probs: torch.Tensor,
    freq_grid: torch.Tensor,
    degree: int = 3,
    topk: int = 7,
    robust_iters: int = 3,
    huber_hz: float = 10.0,
    min_weight: float = 1.0e-4,
) -> torch.Tensor:
    """Fit low-order IF curves from ridge probability maps.

    Unlike polynomial_project_if, this does not fit the already-collapsed IF
    curve. It fits a polynomial through the high-probability ridge candidates in
    the heatmap. That makes it less sensitive to short soft-argmax track jumps.

    probs: [B, Q, F, T]
    freq_grid: [F]
    returns [B, Q, T] in Hz
    """

    if probs.ndim != 4:
        raise ValueError(f"Expected probs [B, Q, F, T], got {tuple(probs.shape)}")
    bsz, q, freq_bins, frames = probs.shape
    if frames <= degree + 1:
        return (probs * freq_grid.view(1, 1, -1, 1)).sum(dim=2)

    topk = min(max(1, int(topk)), freq_bins)
    degree = max(0, int(degree))
    robust_iters = max(0, int(robust_iters))

    x_frame = torch.linspace(-1.0, 1.0, frames, device=probs.device, dtype=probs.dtype)
    x = x_frame.repeat_interleave(topk)
    design = torch.stack([x.pow(k) for k in range(degree + 1)], dim=1)
    eval_design = torch.stack([x_frame.pow(k) for k in range(degree + 1)], dim=1)

    out = torch.empty((bsz, q, frames), device=probs.device, dtype=probs.dtype)
    values, indices = torch.topk(probs, k=topk, dim=2)  # [B, Q, K, T]

    for b in range(bsz):
        for c in range(q):
            y = freq_grid[indices[b, c].transpose(0, 1).reshape(-1)]
            base_w = values[b, c].transpose(0, 1).reshape(-1).clamp_min(min_weight)
            weights = base_w / base_w.max().clamp_min(min_weight)
            fitted = y
            for _ in range(robust_iters + 1):
                sqrt_w = weights.sqrt().unsqueeze(1)
                aw = design * sqrt_w
                yw = y * weights.sqrt()
                coeff = torch.linalg.lstsq(aw, yw).solution
                fitted = design @ coeff
                residual = (y - fitted).abs()
                robust_w = torch.clamp(float(huber_hz) / residual.clamp_min(1.0e-6), max=1.0)
                weights = (base_w / base_w.max().clamp_min(min_weight)) * robust_w
            out[b, c] = eval_design @ coeff
    return out


@torch.no_grad()
def despike_if(
    pred_if: torch.Tensor,
    threshold_hz: float = 18.0,
    median_kernel: int = 5,
) -> torch.Tensor:
    """Replace isolated one-frame IF spikes while preserving monotone jumps."""

    if pred_if.ndim != 3:
        raise ValueError(f"Expected pred_if [B, Q, T], got {tuple(pred_if.shape)}")
    frames = pred_if.shape[-1]
    if frames < 3:
        return pred_if

    kernel = max(3, int(median_kernel) | 1)
    pad = kernel // 2
    padded = F.pad(pred_if.reshape(-1, 1, frames), (pad, pad), mode="replicate")
    local_median = padded.unfold(dimension=2, size=kernel, step=1).median(dim=-1).values
    local_median = local_median.reshape_as(pred_if)

    out = pred_if.clone()
    left_diff = pred_if[..., 1:-1] - pred_if[..., :-2]
    right_diff = pred_if[..., 2:] - pred_if[..., 1:-1]
    turning = left_diff * right_diff < 0.0
    large_swing = torch.minimum(left_diff.abs(), right_diff.abs()) > float(threshold_hz)
    residual = (pred_if[..., 1:-1] - local_median[..., 1:-1]).abs()
    mask = turning & large_swing & (residual > float(threshold_hz))
    out[..., 1:-1] = torch.where(mask, local_median[..., 1:-1], pred_if[..., 1:-1])
    return out


def apply_if_postprocess(
    pred_if: torch.Tensor,
    mode: str = "none",
    degree: int = 3,
    robust_iters: int = 2,
    huber_hz: float = 12.0,
    probs: torch.Tensor | None = None,
    freq_grid: torch.Tensor | None = None,
    topk: int = 7,
) -> torch.Tensor:
    mode = mode.lower()
    if mode in {"none", "raw"}:
        return pred_if
    if mode in {"poly", "polynomial", "poly3"}:
        return polynomial_project_if(pred_if, degree=degree, robust_iters=robust_iters, huber_hz=huber_hz)
    if mode in {"poly_prob", "poly_heatmap", "prob_poly"}:
        if probs is None or freq_grid is None:
            raise ValueError("poly_prob postprocess requires probs and freq_grid.")
        return polynomial_project_from_probs(
            probs,
            freq_grid,
            degree=degree,
            topk=topk,
            robust_iters=robust_iters,
            huber_hz=huber_hz,
        )
    if mode in {"despike", "median_spike", "jump_despike"}:
        return despike_if(pred_if, threshold_hz=huber_hz, median_kernel=5)
    raise ValueError(f"Unknown IF postprocess mode: {mode}")
