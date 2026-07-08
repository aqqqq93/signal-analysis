from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from ifnet_stage1.router import RouterConvBlock, compute_router_aux_features


ACTIVE_COUNT_NAMES = ("active_1", "active_2")


@dataclass
class ActiveCountConfig:
    base_channels: int = 20
    depth: int = 3
    dropout: float = 0.08
    use_aux_features: bool = True
    aux_feature_dim: int = 20
    aux_hidden: int = 40


def active_count_config_from_dict(data: dict | None) -> ActiveCountConfig:
    data = data or {}
    return ActiveCountConfig(
        base_channels=int(data.get("base_channels", 20)),
        depth=int(data.get("depth", 3)),
        dropout=float(data.get("dropout", 0.08)),
        use_aux_features=bool(data.get("use_aux_features", True)),
        aux_feature_dim=int(data.get("aux_feature_dim", 20)),
        aux_hidden=int(data.get("aux_hidden", 40)),
    )


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
        if cfg.use_aux_features:
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
            pooled = torch.cat([pooled, self.aux_net(aux)], dim=1)
        return self.classifier(pooled)


def active_count_labels(active_mask: torch.Tensor) -> torch.Tensor:
    counts = active_mask.to(dtype=torch.float32).sum(dim=1).round().long()
    labels = counts - 1
    return labels.clamp(0, len(ACTIVE_COUNT_NAMES) - 1)


def active_count_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(dim=1)
    top2 = probs.topk(k=min(2, probs.shape[1]), dim=1).values
    margin = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else torch.ones_like(top2[:, 0])
    metrics = {
        "accuracy": float((pred == labels).float().mean().detach().cpu()),
        "confidence": float(top2[:, 0].mean().detach().cpu()),
        "margin": float(margin.mean().detach().cpu()),
    }
    for idx, name in enumerate(ACTIVE_COUNT_NAMES):
        mask = labels == idx
        if bool(mask.any()):
            metrics[f"{name}_accuracy"] = float((pred[mask] == labels[mask]).float().mean().detach().cpu())
            metrics[f"{name}_count"] = float(mask.float().sum().detach().cpu())
        else:
            metrics[f"{name}_accuracy"] = float("nan")
            metrics[f"{name}_count"] = 0.0
    return metrics
