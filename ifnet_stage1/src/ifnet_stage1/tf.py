from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class STFTScale:
    n_fft: int
    win_length: int


@dataclass
class STFTConfig:
    n_fft: int = 256
    hop_length: int = 4
    win_length: int = 128
    log_eps: float = 1.0e-6
    scales: tuple[STFTScale, ...] = ()
    target_n_fft: int | None = None


def stft_config_from_dict(data: dict) -> STFTConfig:
    scales = tuple(
        STFTScale(n_fft=int(item["n_fft"]), win_length=int(item["win_length"]))
        for item in data.get("scales", [])
    )
    target_n_fft = data.get("target_n_fft")
    return STFTConfig(
        n_fft=int(data.get("n_fft", 256)),
        hop_length=int(data.get("hop_length", 4)),
        win_length=int(data.get("win_length", 128)),
        log_eps=float(data.get("log_eps", 1.0e-6)),
        scales=scales,
        target_n_fft=int(target_n_fft) if target_n_fft is not None else None,
    )


def log_spectrogram(signal: torch.Tensor, cfg: STFTConfig, fs: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized log magnitude spectrogram and frequency grid.

    Input: signal [B, N]
    Output: features [B, C, F, T], freq_grid [F]
    """

    if signal.ndim != 2:
        raise ValueError(f"Expected signal [B, N], got {tuple(signal.shape)}")
    scales = cfg.scales or (STFTScale(n_fft=cfg.n_fft, win_length=cfg.win_length),)
    target_n_fft = cfg.target_n_fft or max(scale.n_fft for scale in scales)
    target_freq_bins = target_n_fft // 2 + 1
    features = []
    target_time_bins = None

    for scale in scales:
        feat = _single_scale_log_spectrogram(signal, scale, cfg)
        if target_time_bins is None:
            target_time_bins = feat.shape[-1]
        if feat.shape[-2:] != (target_freq_bins, target_time_bins):
            feat = F.interpolate(
                feat.unsqueeze(1),
                size=(target_freq_bins, target_time_bins),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        features.append(feat)

    feat = torch.stack(features, dim=1)
    freq_grid = torch.linspace(0.0, fs / 2.0, target_freq_bins, device=signal.device, dtype=signal.dtype)
    return feat, freq_grid


def _single_scale_log_spectrogram(signal: torch.Tensor, scale: STFTScale, cfg: STFTConfig) -> torch.Tensor:
    window = torch.hann_window(scale.win_length, device=signal.device, dtype=signal.dtype)
    spec = torch.stft(
        signal,
        n_fft=scale.n_fft,
        hop_length=cfg.hop_length,
        win_length=scale.win_length,
        window=window,
        center=True,
        return_complex=True,
        onesided=True,
        normalized=False,
    )
    mag = spec.abs()
    feat = torch.log(mag + cfg.log_eps)
    mean = feat.mean(dim=(-2, -1), keepdim=True)
    std = feat.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-5)
    feat = (feat - mean) / std
    return feat


def feature_channels(cfg: STFTConfig) -> int:
    return max(1, len(cfg.scales))


def sample_if_to_frames(if_hz: torch.Tensor, num_frames: int, hop_length: int) -> torch.Tensor:
    """Sample sample-rate IF labels at STFT frame centers."""

    if if_hz.ndim != 3:
        raise ValueError(f"Expected IF [B, Q, N], got {tuple(if_hz.shape)}")
    n_samples = if_hz.shape[-1]
    frame_idx = torch.arange(num_frames, device=if_hz.device) * hop_length
    frame_idx = frame_idx.clamp_max(n_samples - 1).long()
    return if_hz.index_select(dim=-1, index=frame_idx)
