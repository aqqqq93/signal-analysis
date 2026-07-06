from __future__ import annotations

import torch


def candidate_route_indices(
    route_probs: torch.Tensor,
    confidence_threshold: float = 0.78,
    margin_threshold: float = 0.18,
    topk: int = 2,
) -> tuple[list[int], bool]:
    """Return route candidates for a single sample and whether fallback is used."""

    if route_probs.ndim != 1:
        raise ValueError(f"Expected route_probs [R], got {tuple(route_probs.shape)}")
    k = min(topk, route_probs.numel())
    values, indices = torch.topk(route_probs, k=k)
    margin = values[0] - values[1] if k > 1 else route_probs.new_tensor(float("inf"))
    fallback = bool(values[0] < confidence_threshold or margin < margin_threshold)
    if not fallback:
        return [int(indices[0].detach().cpu())], False
    return [int(item.detach().cpu()) for item in indices], True


def export_candidate_indices(
    route_probs: torch.Tensor,
    route_names: tuple[str, ...],
    topk: int = 2,
    policy: str = "router_topk",
    special_routes: tuple[str, ...] = ("sinusoidal_like", "jump_like"),
    special_boost: float = 0.12,
) -> list[int]:
    """Return the candidate routes exported for downstream initialization.

    `router_topk` is the literal top-k router probabilities. `guarded_special`
    keeps the top-1 route fixed, then lets under-covered special routes compete
    for the second slot with a small prior boost. This improves candidate
    coverage for sinusoidal/local-jump cases while preserving a two-candidate
    interface for Stage 2.
    """

    if route_probs.ndim != 1:
        raise ValueError(f"Expected route_probs [R], got {tuple(route_probs.shape)}")
    k = min(max(1, int(topk)), route_probs.numel())
    if policy == "router_topk" or k == 1:
        return [int(item.detach().cpu()) for item in torch.topk(route_probs, k=k).indices]
    if policy != "guarded_special":
        raise ValueError(f"Unknown candidate export policy: {policy}")

    top1 = int(route_probs.argmax().detach().cpu())
    selected = [top1]
    scores = route_probs.clone()
    scores[top1] = -torch.inf
    for route_name in special_routes:
        if route_name in route_names:
            idx = route_names.index(route_name)
            if idx != top1:
                scores[idx] = scores[idx] + float(special_boost)
    while len(selected) < k:
        next_idx = int(scores.argmax().detach().cpu())
        selected.append(next_idx)
        scores[next_idx] = -torch.inf
    return selected


def select_best_candidate(
    route_probs: torch.Tensor,
    candidates: list[tuple[int, str, torch.Tensor, torch.Tensor]],
) -> tuple[int, str]:
    """Pick one candidate using router probability plus IF/heatmap self-scores.

    candidates contain: route_idx, route_name, pred_if [1,Q,T], probs [1,Q,F,T].
    """

    best_score = None
    best_route = candidates[0][0]
    best_name = candidates[0][1]
    for route_idx, route_name, pred_if, ridge_probs in candidates:
        score = -torch.log(route_probs[route_idx].clamp_min(1.0e-6))
        score = score + 0.30 * heatmap_entropy(ridge_probs)
        score = score + 0.18 * route_shape_mismatch(route_name, pred_if)
        if route_name == "jump_like":
            score = score - guarded_jump_bonus(route_probs, route_idx, pred_if.reshape(-1, pred_if.shape[-1]))
        score_value = float(score.detach().cpu())
        if best_score is None or score_value < best_score:
            best_score = score_value
            best_route = route_idx
            best_name = route_name
    return best_route, best_name


def guarded_jump_bonus(route_probs: torch.Tensor, route_idx: int, curves: torch.Tensor) -> torch.Tensor:
    candidate_prob = route_probs[route_idx]
    top_prob, top_idx = route_probs.max(dim=0)
    close_to_top = bool(candidate_prob >= 0.20 and (top_prob - candidate_prob) <= 0.26)
    if not close_to_top:
        return route_probs.new_tensor(0.0)

    evidence = jump_evidence(curves)
    if bool(evidence < 0.35):
        return route_probs.new_tensor(0.0)

    weight = 0.16
    if int(top_idx.detach().cpu()) != int(route_idx) and bool(top_prob >= 0.58):
        weight = 0.07
    return weight * evidence


def heatmap_entropy(probs: torch.Tensor) -> torch.Tensor:
    freq_bins = probs.shape[-2]
    entropy = -(probs.clamp_min(1.0e-8) * probs.clamp_min(1.0e-8).log()).sum(dim=-2)
    return (entropy / torch.log(probs.new_tensor(float(freq_bins)))).mean()


def route_shape_mismatch(route_name: str, pred_if: torch.Tensor) -> torch.Tensor:
    curves = pred_if.reshape(-1, pred_if.shape[-1])
    if route_name == "poly_like":
        return normalized_poly_residual(curves)
    if route_name == "sinusoidal_like":
        return normalized_sine_residual(curves)
    if route_name == "jump_like":
        return jump_mismatch(curves)
    return smoothness_mismatch(curves)


def normalized_poly_residual(curves: torch.Tensor, degree: int = 3) -> torch.Tensor:
    num_frames = curves.shape[-1]
    time = torch.linspace(-1.0, 1.0, num_frames, device=curves.device, dtype=curves.dtype)
    basis = torch.stack([time.pow(power) for power in range(degree + 1)], dim=1)
    return _normalized_basis_residual(curves, basis).mean()


def normalized_sine_residual(curves: torch.Tensor) -> torch.Tensor:
    num_frames = curves.shape[-1]
    time = torch.linspace(0.0, 1.0, num_frames, device=curves.device, dtype=curves.dtype)
    residuals = []
    for cycles in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0):
        angle = 2.0 * torch.pi * float(cycles) * time
        basis = torch.stack(
            [torch.ones_like(time), time - 0.5, torch.sin(angle), torch.cos(angle)],
            dim=1,
        )
        residuals.append(_normalized_basis_residual(curves, basis))
    return torch.stack(residuals, dim=1).min(dim=1).values.mean()


def jump_mismatch(curves: torch.Tensor) -> torch.Tensor:
    if curves.shape[-1] < 2:
        return curves.new_tensor(1.0)
    slope_abs = (curves[:, 1:] - curves[:, :-1]).abs()
    jump_ratio = slope_abs.max(dim=1).values / slope_abs.mean(dim=1).clamp_min(1.0e-6)
    return (1.0 / jump_ratio.clamp_min(1.0)).mean()


def jump_evidence(curves: torch.Tensor) -> torch.Tensor:
    if curves.shape[-1] < 3:
        return curves.new_tensor(0.0)
    slope_abs = (curves[:, 1:] - curves[:, :-1]).abs()
    second_abs = (curves[:, 2:] - 2.0 * curves[:, 1:-1] + curves[:, :-2]).abs()
    jump_ratio = slope_abs.max(dim=1).values / slope_abs.mean(dim=1).clamp_min(1.0e-6)
    second_ratio = second_abs.max(dim=1).values / second_abs.mean(dim=1).clamp_min(1.0e-6)
    evidence = 0.5 * torch.clamp((jump_ratio - 3.0) / 6.0, 0.0, 1.0)
    evidence = evidence + 0.5 * torch.clamp((second_ratio - 3.0) / 6.0, 0.0, 1.0)
    return evidence.mean()


def smoothness_mismatch(curves: torch.Tensor) -> torch.Tensor:
    if curves.shape[-1] < 3:
        return curves.new_tensor(0.0)
    second = curves[:, 2:] - 2.0 * curves[:, 1:-1] + curves[:, :-2]
    scale = (curves.max(dim=1).values - curves.min(dim=1).values).clamp_min(1.0)
    return torch.sqrt(second.pow(2).mean(dim=1)).div(scale).mean()


def _normalized_basis_residual(curves: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    projection = basis @ torch.linalg.pinv(basis)
    fitted = curves @ projection.transpose(0, 1)
    residual = (curves - fitted).pow(2).mean(dim=1)
    denom = (curves - curves.mean(dim=1, keepdim=True)).pow(2).mean(dim=1).clamp_min(1.0e-8)
    return residual / denom
