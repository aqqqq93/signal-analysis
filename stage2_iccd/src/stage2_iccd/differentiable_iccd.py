from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ICCDConfig:
    fs: float = 1024.0
    n_samples: int = 1024
    amplitude_order: int = 8
    period_factor: float = 2.0
    alpha_init: float = 1.0
    alpha_min: float = 1.0e-5
    solve_jitter: float = 1.0e-6
    freq_min: float = 0.0
    freq_max: float | None = None


def iccd_config_from_dict(data: dict) -> ICCDConfig:
    return ICCDConfig(
        fs=float(data.get("fs", 1024.0)),
        n_samples=int(data.get("n_samples", 1024)),
        amplitude_order=int(data.get("amplitude_order", 8)),
        period_factor=float(data.get("period_factor", 2.0)),
        alpha_init=float(data.get("alpha_init", 1.0)),
        alpha_min=float(data.get("alpha_min", 1.0e-5)),
        solve_jitter=float(data.get("solve_jitter", 1.0e-6)),
        freq_min=float(data.get("freq_min", 0.0)),
        freq_max=float(data["freq_max"]) if data.get("freq_max") is not None else None,
    )


class DifferentiableICCD(nn.Module):
    """Real-valued differentiable ICCD reconstruction layer.

    Given IF curves, this layer builds the ICCD dictionary

        x_m(t) = a_m(t) cos(phi_m(t)) + b_m(t) sin(phi_m(t))

    where a_m(t) and b_m(t) are Fourier envelopes. The envelope coefficients
    are solved by a batched Tikhonov least-squares system. Gradients propagate
    through the linear solve, the phase integration, and the IF curves.
    """

    def __init__(self, cfg: ICCDConfig):
        super().__init__()
        self.cfg = cfg
        raw_alpha = _inverse_softplus(torch.tensor(max(cfg.alpha_init - cfg.alpha_min, 1.0e-6)))
        self.raw_alpha = nn.Parameter(raw_alpha)

        basis = fourier_envelope_basis(
            n_samples=cfg.n_samples,
            fs=cfg.fs,
            order=cfg.amplitude_order,
            period_factor=cfg.period_factor,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        self.register_buffer("envelope_basis", basis, persistent=False)

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha) + float(self.cfg.alpha_min)

    def forward(self, signal: torch.Tensor, if_hz: torch.Tensor) -> dict[str, torch.Tensor]:
        if signal.ndim != 2:
            raise ValueError(f"Expected signal [B, N], got {tuple(signal.shape)}")
        if if_hz.ndim != 3:
            raise ValueError(f"Expected IF [B, Q, N], got {tuple(if_hz.shape)}")
        if signal.shape[0] != if_hz.shape[0] or signal.shape[-1] != if_hz.shape[-1]:
            raise ValueError("Signal and IF batch/sample dimensions must match.")

        clipped_if = if_hz
        if self.cfg.freq_max is not None:
            clipped_if = clipped_if.clamp(float(self.cfg.freq_min), float(self.cfg.freq_max))
        else:
            clipped_if = clipped_if.clamp_min(float(self.cfg.freq_min))

        dictionary = build_real_iccd_dictionary(clipped_if, self.envelope_basis, fs=self.cfg.fs)
        coeffs = regularized_lstsq(dictionary, signal, self.alpha, jitter=float(self.cfg.solve_jitter))
        bsz, num_components, _ = if_hz.shape
        block_cols = self.envelope_basis.shape[1] * 2
        components = []
        for comp_idx in range(num_components):
            start = comp_idx * block_cols
            stop = start + block_cols
            comp = torch.bmm(dictionary[:, :, start:stop], coeffs[:, start:stop, :]).squeeze(-1)
            components.append(comp)
        reconstructed_components = torch.stack(components, dim=1)
        reconstructed = reconstructed_components.sum(dim=1)
        return {
            "reconstruction": reconstructed,
            "components": reconstructed_components,
            "coefficients": coeffs.squeeze(-1),
            "dictionary": dictionary,
            "if_hz": clipped_if,
            "alpha": self.alpha,
        }


def fourier_envelope_basis(
    n_samples: int,
    fs: float,
    order: int,
    period_factor: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    time = torch.arange(n_samples, dtype=dtype, device=device) / float(fs)
    base_freq = float(fs) / (float(period_factor) * float(n_samples))
    columns = [torch.ones_like(time)]
    for idx in range(1, order + 1):
        angle = 2.0 * torch.pi * float(idx) * base_freq * time
        columns.append(torch.cos(angle))
    for idx in range(1, order + 1):
        angle = 2.0 * torch.pi * float(idx) * base_freq * time
        columns.append(torch.sin(angle))
    return torch.stack(columns, dim=1)


def build_real_iccd_dictionary(if_hz: torch.Tensor, envelope_basis: torch.Tensor, fs: float) -> torch.Tensor:
    bsz, num_components, n_samples = if_hz.shape
    basis = envelope_basis.to(device=if_hz.device, dtype=if_hz.dtype)
    if basis.shape[0] != n_samples:
        raise ValueError(f"Envelope basis has {basis.shape[0]} samples, but IF has {n_samples}.")

    phase = 2.0 * torch.pi * torch.cumsum(if_hz / float(fs), dim=-1)
    cos_part = torch.cos(phase).unsqueeze(-1) * basis.view(1, 1, n_samples, -1)
    sin_part = torch.sin(phase).unsqueeze(-1) * basis.view(1, 1, n_samples, -1)
    blocks = torch.cat([cos_part, sin_part], dim=-1)
    return blocks.permute(0, 2, 1, 3).reshape(bsz, n_samples, -1)


def regularized_lstsq(dictionary: torch.Tensor, signal: torch.Tensor, alpha: torch.Tensor, jitter: float) -> torch.Tensor:
    lhs = torch.bmm(dictionary.transpose(1, 2), dictionary)
    rhs = torch.bmm(dictionary.transpose(1, 2), signal.unsqueeze(-1))
    eye = torch.eye(lhs.shape[-1], device=dictionary.device, dtype=dictionary.dtype).view(1, lhs.shape[-1], lhs.shape[-1])
    reg = (alpha.to(device=dictionary.device, dtype=dictionary.dtype) + float(jitter)) * eye
    return torch.linalg.solve(lhs + reg, rhs)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value).clamp_min(1.0e-8))
