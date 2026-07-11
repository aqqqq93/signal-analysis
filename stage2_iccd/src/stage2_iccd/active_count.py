from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ifnet_stage1.router import RouterConvBlock, compute_router_aux_features


DEFAULT_ACTIVE_COUNT_NAMES = ("active_1", "active_2")
ACTIVE_COUNT_NAMES = DEFAULT_ACTIVE_COUNT_NAMES


@dataclass
class ActiveCountConfig:
    base_channels: int = 20
    depth: int = 3
    dropout: float = 0.08
    num_classes: int = 2
    use_aux_features: bool = True
    aux_feature_dim: int = 20
    use_peak_features: bool = False
    peak_feature_dim: int = 8
    aux_hidden: int = 40


def active_count_config_from_dict(data: dict | None) -> ActiveCountConfig:
    data = data or {}
    return ActiveCountConfig(
        base_channels=int(data.get("base_channels", 20)),
        depth=int(data.get("depth", 3)),
        dropout=float(data.get("dropout", 0.08)),
        num_classes=int(data.get("num_classes", 2)),
        use_aux_features=bool(data.get("use_aux_features", True)),
        aux_feature_dim=int(data.get("aux_feature_dim", 20)),
        use_peak_features=bool(data.get("use_peak_features", False)),
        peak_feature_dim=int(data.get("peak_feature_dim", 8)),
        aux_hidden=int(data.get("aux_hidden", 40)),
    )


def active_count_names(num_classes: int) -> tuple[str, ...]:
    return tuple(f"active_{idx}" for idx in range(1, int(num_classes) + 1))


class ActiveCountClassifier(nn.Module):
    """Classify whether the mixture has one or two active components."""

    def __init__(self, in_channels: int, cfg: ActiveCountConfig, num_classes: int = 2):
        super().__init__()
        if cfg.depth < 1:
            raise ValueError("Active-count classifier depth must be at least 1.")
        channels = [cfg.base_channels * (2**idx) for idx in range(cfg.depth)]
        blocks = []
        prev = in_channels
        for out_channels in channels:
            blocks.append(RouterConvBlock(prev, out_channels, cfg.dropout))
            prev = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.use_aux_features = cfg.use_aux_features
        head_in = channels[-1]
        self.use_peak_features = bool(cfg.use_peak_features)
        peak_dim = int(cfg.peak_feature_dim) if self.use_peak_features else 0
        if cfg.use_aux_features:
            self.aux_net = nn.Sequential(
                nn.LayerNorm(cfg.aux_feature_dim + peak_dim),
                nn.Linear(cfg.aux_feature_dim + peak_dim, cfg.aux_hidden),
                nn.SiLU(inplace=True),
                nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            )
            self.aux_feature_dim = cfg.aux_feature_dim
            self.peak_feature_dim = peak_dim
            head_in += cfg.aux_hidden
        else:
            self.aux_net = None
            self.aux_feature_dim = 0
            self.peak_feature_dim = 0
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(head_in, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        h = features
        for idx, block in enumerate(self.blocks):
            h = block(h)
            if idx != len(self.blocks) - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)
        pooled = self.pool(h).flatten(1)
        if self.use_aux_features:
            aux = compute_router_aux_features(features, self.aux_feature_dim)
            if self.use_peak_features:
                aux = torch.cat([aux, compute_peak_count_features(features, self.peak_feature_dim)], dim=1)
            pooled = torch.cat([pooled, self.aux_net(aux)], dim=1)
        return self.classifier(pooled)


def active_count_labels(active_mask: torch.Tensor, num_classes: int | None = None) -> torch.Tensor:
    if num_classes is None:
        num_classes = len(ACTIVE_COUNT_NAMES)
    counts = active_mask.to(dtype=torch.float32).sum(dim=1).round().long()
    labels = counts - 1
    return labels.clamp(0, int(num_classes) - 1)


def active_count_metrics(logits: torch.Tensor, labels: torch.Tensor, names: tuple[str, ...] | None = None) -> dict[str, float]:
    names = names or active_count_names(logits.shape[1])
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    top2 = probs.topk(k=min(2, probs.shape[1]), dim=1).values
    margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else torch.ones_like(top2[:, 0])
    metrics = {
        "accuracy": float((pred == labels).float().mean().detach().cpu()),
        "confidence": float(top2[:, 0].mean().detach().cpu()),
        "margin": float(margin.mean().detach().cpu()),
    }
    for idx, name in enumerate(names):
        mask = labels == idx
        if bool(mask.any()):
            metrics[f"{name}_accuracy"] = float((pred[mask] == labels[mask]).float().mean().detach().cpu())
            metrics[f"{name}_count"] = float(mask.float().sum().detach().cpu())
        else:
            metrics[f"{name}_accuracy"] = float("nan")
            metrics[f"{name}_count"] = 0.0
    return metrics


def compute_peak_count_features(features: torch.Tensor, feature_dim: int = 8) -> torch.Tensor:
    if features.ndim != 4:
        raise ValueError(f"Expected features [B, C, F, T], got {tuple(features.shape)}")
    ridge_map = features.mean(dim=1)
    probs = torch.softmax(ridge_map / 0.35, dim=1)
    top_values = probs.topk(k=min(4, probs.shape[1]), dim=1).values
    if top_values.shape[1] < 4:
        pad = top_values.new_zeros((top_values.shape[0], 4 - top_values.shape[1], top_values.shape[2]))
        top_values = torch.cat([top_values, pad], dim=1)
    ratios = top_values[:, 1:] / top_values[:, :1].clamp_min(1.0e-6)
    peak_counts = torch.stack(
        [
            (ratios[:, 0] > 0.55).float().mean(dim=1),
            (ratios[:, 1] > 0.45).float().mean(dim=1),
            (ratios[:, 2] > 0.35).float().mean(dim=1),
            top_values[:, 0].mean(dim=1),
            top_values[:, 1].mean(dim=1),
            ratios[:, 0].mean(dim=1),
            ratios[:, 1].mean(dim=1),
            ratios[:, 2].mean(dim=1),
        ],
        dim=1,
    )
    if peak_counts.shape[1] < feature_dim:
        pad = peak_counts.new_zeros((peak_counts.shape[0], feature_dim - peak_counts.shape[1]))
        peak_counts = torch.cat([peak_counts, pad], dim=1)
    return peak_counts[:, :feature_dim]
