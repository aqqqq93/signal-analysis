from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    base_channels: int = 24
    depth: int = 3
    dropout: float = 0.05
    temperature: float = 0.08


def model_config_from_dict(data: dict) -> ModelConfig:
    return ModelConfig(
        base_channels=int(data.get("base_channels", 24)),
        depth=int(data.get("depth", 3)),
        dropout=float(data.get("dropout", 0.05)),
        temperature=float(data.get("temperature", 0.08)),
    )


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IFNetUNet(nn.Module):
    """Small U-Net that predicts one ridge heatmap per component."""

    def __init__(self, in_channels: int, num_components: int, cfg: ModelConfig):
        super().__init__()
        if cfg.depth < 1:
            raise ValueError("Model depth must be at least 1.")
        channels = [cfg.base_channels * (2**i) for i in range(cfg.depth + 1)]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(ConvBlock(prev, ch, cfg.dropout))
            prev = ch

        self.decoders = nn.ModuleList()
        rev_channels = list(reversed(channels))
        for idx in range(len(rev_channels) - 1):
            in_ch = rev_channels[idx] + rev_channels[idx + 1]
            out_ch = rev_channels[idx + 1]
            self.decoders.append(ConvBlock(in_ch, out_ch, cfg.dropout))

        self.head = nn.Conv2d(channels[0], num_components, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for idx, enc in enumerate(self.encoders):
            h = enc(h)
            skips.append(h)
            if idx != len(self.encoders) - 1:
                h = F.avg_pool2d(h, kernel_size=2, stride=2)

        for dec in self.decoders:
            skip = skips.pop(-2)
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        return self.head(h)


def soft_argmax_if(logits: torch.Tensor, freq_grid: torch.Tensor, temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ridge heatmaps to continuous IF estimates.

    logits: [B, Q, F, T]
    returns:
      pred_if: [B, Q, T]
      probs: [B, Q, F, T]
    """

    if logits.ndim != 4:
        raise ValueError(f"Expected logits [B, Q, F, T], got {tuple(logits.shape)}")
    temp = max(float(temperature), 1.0e-5)
    probs = torch.softmax(logits / temp, dim=2)
    pred = (probs * freq_grid.view(1, 1, -1, 1)).sum(dim=2)
    return pred, probs
