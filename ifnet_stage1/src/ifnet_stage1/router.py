from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


ROUTE_NAMES = ("poly_like", "sinusoidal_like", "cross_overlap_like", "jump_like")
ROUTER_AUX_FEATURE_DIM = 12
ROUTER_AUX_FEATURE_DIM_V2 = 20

DEFAULT_SCENARIO_TO_ROUTE = {
    "linear": "poly_like",
    "quadratic": "poly_like",
    "cubic": "poly_like",
    "sinusoidal_fm": "sinusoidal_like",
    "crossing": "cross_overlap_like",
    "near_parallel": "cross_overlap_like",
    "tangent_or_overlap": "cross_overlap_like",
    "local_jump": "jump_like",
}


@dataclass
class RouterConfig:
    base_channels: int = 24
    depth: int = 3
    dropout: float = 0.10
    use_aux_features: bool = False
    aux_hidden: int = 32
    aux_feature_dim: int = ROUTER_AUX_FEATURE_DIM


def router_config_from_dict(data: dict) -> RouterConfig:
    return RouterConfig(
        base_channels=int(data.get("base_channels", 24)),
        depth=int(data.get("depth", 3)),
        dropout=float(data.get("dropout", 0.10)),
        use_aux_features=bool(data.get("use_aux_features", False)),
        aux_hidden=int(data.get("aux_hidden", 32)),
        aux_feature_dim=int(data.get("aux_feature_dim", ROUTER_AUX_FEATURE_DIM)),
    )


def scenario_to_route_labels(
    scenarios: list[str],
    route_names: tuple[str, ...] = ROUTE_NAMES,
    scenario_to_route: dict[str, str] | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    mapping = scenario_to_route or DEFAULT_SCENARIO_TO_ROUTE
    route_to_idx = {name: idx for idx, name in enumerate(route_names)}
    labels = []
    for scenario in scenarios:
        try:
            route_name = mapping[scenario]
            labels.append(route_to_idx[route_name])
        except KeyError as exc:
            raise ValueError(f"No route mapping for scenario {scenario!r}.") from exc
    return torch.tensor(labels, dtype=torch.long, device=device)


class RouterConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HardRouteClassifier(nn.Module):
    """Compact CNN that maps time-frequency features to one hard route group."""

    def __init__(self, in_channels: int, num_routes: int, cfg: RouterConfig):
        super().__init__()
        if cfg.depth < 1:
            raise ValueError("Router depth must be at least 1.")
        channels = [cfg.base_channels * (2**idx) for idx in range(cfg.depth)]
        blocks = []
        prev_channels = in_channels
        for out_channels in channels:
            blocks.append(RouterConvBlock(prev_channels, out_channels, cfg.dropout))
            prev_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.use_aux_features = cfg.use_aux_features
        head_in = channels[-1]
        if self.use_aux_features:
            self.aux_net = nn.Sequential(
                nn.LayerNorm(cfg.aux_feature_dim),
                nn.Linear(cfg.aux_feature_dim, cfg.aux_hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            )
            self.aux_feature_dim = cfg.aux_feature_dim
            head_in += cfg.aux_hidden
        else:
            self.aux_net = None
            self.aux_feature_dim = 0
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(head_in, num_routes),
        )

    def forward(self, x: torch.Tensor, aux_features: torch.Tensor | None = None) -> torch.Tensor:
        h = x
        for idx, block in enumerate(self.blocks):
            h = block(h)
            if idx != len(self.blocks) - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)
        pooled = self.head(h).flatten(1)
        if self.use_aux_features:
            aux = compute_router_aux_features(x, self.aux_feature_dim) if aux_features is None else aux_features
            pooled = torch.cat([pooled, self.aux_net(aux)], dim=1)
        return self.classifier(pooled)


def compute_router_aux_features(features: torch.Tensor, feature_dim: int = ROUTER_AUX_FEATURE_DIM) -> torch.Tensor:
    """Extract low-dimensional ridge-shape cues from normalized TF features."""

    if features.ndim != 4:
        raise ValueError(f"Expected features [B, C, F, T], got {tuple(features.shape)}")
    ridge_map = features.mean(dim=1)
    bsz, freq_bins, time_bins = ridge_map.shape
    freq_axis = torch.linspace(0.0, 1.0, freq_bins, device=features.device, dtype=features.dtype)
    probs = torch.softmax(ridge_map / 0.35, dim=1)
    ridge = (probs * freq_axis.view(1, -1, 1)).sum(dim=1)
    bandwidth = torch.sqrt((probs * (freq_axis.view(1, -1, 1) - ridge.unsqueeze(1)).pow(2)).sum(dim=1).clamp_min(1.0e-8))
    peak = probs.max(dim=1).values

    if time_bins > 1:
        slope = ridge[:, 1:] - ridge[:, :-1]
        slope_abs = slope.abs()
    else:
        slope_abs = ridge.new_zeros((bsz, 1))
    if time_bins > 2:
        second = ridge[:, 2:] - 2.0 * ridge[:, 1:-1] + ridge[:, :-2]
        second_abs = second.abs()
    else:
        second_abs = ridge.new_zeros((bsz, 1))

    poly_residual = _basis_residual(ridge, _poly_basis(time_bins, features.device, features.dtype, degree=3))
    sine_residual = _best_sine_residual(ridge)
    slope_mean = slope_abs.mean(dim=1)
    second_mean = second_abs.mean(dim=1)
    jump_ratio = slope_abs.max(dim=1).values / slope_mean.clamp_min(1.0e-5)

    base_features = torch.stack(
        [
            ridge.mean(dim=1),
            ridge.std(dim=1),
            slope_mean,
            slope_abs.max(dim=1).values,
            second_mean,
            second_abs.max(dim=1).values,
            jump_ratio,
            poly_residual,
            sine_residual,
            bandwidth.mean(dim=1),
            bandwidth.max(dim=1).values,
            peak.mean(dim=1),
        ],
        dim=1,
    )
    if feature_dim <= ROUTER_AUX_FEATURE_DIM:
        return base_features[:, :feature_dim]

    top_features = _top_ridge_jump_features(probs, freq_axis)
    aux = torch.cat([base_features, top_features], dim=1)
    if aux.shape[1] < feature_dim:
        pad = aux.new_zeros((aux.shape[0], feature_dim - aux.shape[1]))
        aux = torch.cat([aux, pad], dim=1)
    return aux[:, :feature_dim]


def _top_ridge_jump_features(probs: torch.Tensor, freq_axis: torch.Tensor) -> torch.Tensor:
    bsz, _freq_bins, time_bins = probs.shape
    if time_bins < 2:
        return probs.new_zeros((bsz, ROUTER_AUX_FEATURE_DIM_V2 - ROUTER_AUX_FEATURE_DIM))

    _values, indices = probs.topk(k=2, dim=1)
    ridges = freq_axis[indices].sort(dim=1).values
    low = ridges[:, 0]
    high = ridges[:, 1]
    gap = (high - low).abs()

    low_slope = low[:, 1:] - low[:, :-1]
    high_slope = high[:, 1:] - high[:, :-1]
    low_slope_abs = low_slope.abs()
    high_slope_abs = high_slope.abs()
    low_second_abs = _second_abs(low)
    high_second_abs = _second_abs(high)
    gap_diff_abs = (gap[:, 1:] - gap[:, :-1]).abs()
    time_edge = (probs[:, :, 1:] - probs[:, :, :-1]).abs().mean(dim=1)
    time_edge_mean = time_edge.mean(dim=1)
    time_edge_max = time_edge.max(dim=1).values

    return torch.stack(
        [
            _jump_ratio(low_slope_abs),
            _jump_ratio(high_slope_abs),
            low_second_abs.max(dim=1).values,
            high_second_abs.max(dim=1).values,
            gap.min(dim=1).values,
            gap.std(dim=1),
            gap_diff_abs.max(dim=1).values,
            time_edge_max / time_edge_mean.clamp_min(1.0e-6),
        ],
        dim=1,
    )


def _jump_ratio(slope_abs: torch.Tensor) -> torch.Tensor:
    return slope_abs.max(dim=1).values / slope_abs.mean(dim=1).clamp_min(1.0e-5)


def _second_abs(curve: torch.Tensor) -> torch.Tensor:
    if curve.shape[-1] < 3:
        return curve.new_zeros((curve.shape[0], 1))
    return (curve[:, 2:] - 2.0 * curve[:, 1:-1] + curve[:, :-2]).abs()


def _poly_basis(num_frames: int, device: torch.device, dtype: torch.dtype, degree: int) -> torch.Tensor:
    time = torch.linspace(-1.0, 1.0, num_frames, device=device, dtype=dtype)
    return torch.stack([time.pow(power) for power in range(degree + 1)], dim=1)


def _basis_residual(curve: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    projection = basis @ torch.linalg.pinv(basis)
    fitted = curve @ projection.transpose(0, 1)
    residual = (curve - fitted).pow(2).mean(dim=1)
    denom = (curve - curve.mean(dim=1, keepdim=True)).pow(2).mean(dim=1).clamp_min(1.0e-8)
    return residual / denom


def _best_sine_residual(curve: torch.Tensor) -> torch.Tensor:
    num_frames = curve.shape[-1]
    device = curve.device
    dtype = curve.dtype
    time = torch.linspace(0.0, 1.0, num_frames, device=device, dtype=dtype)
    residuals = []
    for cycles in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0):
        angle = 2.0 * torch.pi * float(cycles) * time
        basis = torch.stack(
            [
                torch.ones_like(time),
                time - 0.5,
                torch.sin(angle),
                torch.cos(angle),
            ],
            dim=1,
        )
        residuals.append(_basis_residual(curve, basis))
    return torch.stack(residuals, dim=1).min(dim=1).values
