from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .model import ConvBlock, ModelConfig


@dataclass
class JumpAuxConfig:
    ridge_weight: float = 1.0
    jump_weight: float = 0.35
    jump_sigma_frames: float = 2.5
    min_jump_hz: float = 12.0


def jump_aux_config_from_dict(data: dict | None) -> JumpAuxConfig:
    data = data or {}
    return JumpAuxConfig(
        ridge_weight=float(data.get("ridge_weight", 1.0)),
        jump_weight=float(data.get("jump_weight", 0.35)),
        jump_sigma_frames=float(data.get("jump_sigma_frames", 2.5)),
        min_jump_hz=float(data.get("min_jump_hz", 12.0)),
    )


class IFNetJumpAux(nn.Module):
    """IF-Net with an additional per-component jump-location head.

    The ridge head keeps the same semantics as IFNetUNet. The jump head emits
    [B,Q,T] logits, one temporal event distribution per component.
    """

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

        self.ridge_head = nn.Conv2d(channels[0], num_components, kernel_size=1)
        self.jump_head = nn.Conv2d(channels[0], num_components, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        ridge_logits = self.ridge_head(h)
        jump_logits = self.jump_head(h).mean(dim=2)
        return ridge_logits, jump_logits


def make_jump_targets(target_if: torch.Tensor, sigma_frames: float = 2.5, min_jump_hz: float = 12.0) -> torch.Tensor:
    """Build soft temporal labels from the strongest IF slope/curvature event."""

    if target_if.ndim != 3:
        raise ValueError(f"Expected target_if [B,Q,T], got {tuple(target_if.shape)}")
    bsz, q, frames = target_if.shape
    if frames < 3:
        return target_if.new_full((bsz, q, frames), 1.0 / max(1, frames))

    slope = target_if[..., 1:] - target_if[..., :-1]
    slope_abs = slope.abs()
    curvature = target_if[..., 2:] - 2.0 * target_if[..., 1:-1] + target_if[..., :-2]
    score = slope_abs.clone()
    score[..., 1:] = score[..., 1:] + 0.5 * curvature.abs()
    event_idx = score.argmax(dim=-1)
    event_strength = score.max(dim=-1).values

    time = torch.arange(frames, device=target_if.device, dtype=target_if.dtype).view(1, 1, -1)
    center = event_idx.to(target_if.dtype).unsqueeze(-1)
    sigma = max(float(sigma_frames), 1.0e-3)
    target = torch.exp(-0.5 * ((time - center) / sigma).pow(2))
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    uniform = target_if.new_full((bsz, q, frames), 1.0 / frames)
    has_jump = (event_strength >= float(min_jump_hz)).unsqueeze(-1)
    return torch.where(has_jump, target, uniform)


def jump_nll_loss(jump_logits: torch.Tensor, jump_target: torch.Tensor) -> torch.Tensor:
    if jump_logits.shape != jump_target.shape:
        raise ValueError(f"Jump logits/target shape mismatch: {tuple(jump_logits.shape)} vs {tuple(jump_target.shape)}")
    log_probs = F.log_softmax(jump_logits, dim=-1)
    return -(jump_target * log_probs).sum(dim=-1).mean()


@torch.no_grad()
def jump_location_from_if(pred_if: torch.Tensor) -> torch.Tensor:
    if pred_if.shape[-1] < 2:
        return torch.zeros(pred_if.shape[:-1], dtype=torch.long, device=pred_if.device)
    slope_abs = (pred_if[..., 1:] - pred_if[..., :-1]).abs()
    return slope_abs.argmax(dim=-1)


@torch.no_grad()
def jump_location_from_logits(jump_logits: torch.Tensor) -> torch.Tensor:
    return jump_logits.argmax(dim=-1)
