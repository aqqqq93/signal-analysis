from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from ifnet_stage1.jump_aux import IFNetJumpAux
from ifnet_stage1.model import IFNetUNet, model_config_from_dict, soft_argmax_if
from ifnet_stage1.simulation import sim_config_from_dict
from ifnet_stage1.tf import feature_channels, log_spectrogram, stft_config_from_dict

from .model import make_smooth_candidate


class FrozenIFNetCandidateProvider:
    """Produce frozen stage-1 IF candidates for stage-2 training.

    A single checkpoint can be used as raw-plus-smoothed candidates. A list of
    checkpoints can be used to approximate the stage-1 top-k expert candidates.
    Every loaded IF-Net stays frozen.
    """

    def __init__(
        self,
        device: torch.device,
        checkpoint: str | Path | None = None,
        num_candidates: int = 2,
        smooth_kernel: int = 31,
        checkpoints: list[str | Path] | None = None,
    ):
        paths = list(checkpoints or [])
        if checkpoint is not None:
            paths.insert(0, checkpoint)
        if not paths:
            raise ValueError("At least one frozen IF-Net checkpoint is required.")
        self.experts = [self._load_one(path, device) for path in paths]
        self.device = device
        self.num_candidates = max(1, int(num_candidates))
        self.smooth_kernel = int(smooth_kernel)

    @torch.no_grad()
    def __call__(self, signal: torch.Tensor, n_samples: int) -> torch.Tensor:
        candidates = []
        for model, sim_cfg, stft_cfg, model_cfg in self.experts:
            feats, freq_grid = log_spectrogram(signal, stft_cfg, sim_cfg.fs)
            model_out = model(feats)
            logits = model_out[0] if isinstance(model_out, tuple) else model_out
            frame_if, _ = soft_argmax_if(logits, freq_grid, model_cfg.temperature)
            candidates.append(F.interpolate(frame_if, size=n_samples, mode="linear", align_corners=False))
            if len(candidates) >= self.num_candidates:
                break
        if self.num_candidates > 1:
            idx = 0
            while len(candidates) < self.num_candidates and idx < len(candidates):
                candidates.append(make_smooth_candidate(candidates[idx], kernel_size=self.smooth_kernel))
                idx += 1
        while len(candidates) < self.num_candidates:
            candidates.append(candidates[-1].clone())
        return torch.stack(candidates[: self.num_candidates], dim=1)

    @staticmethod
    def _load_one(path: str | Path, device: torch.device):
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        sim_cfg = sim_config_from_dict(cfg["data"])
        stft_cfg = stft_config_from_dict(cfg["stft"])
        model_cfg = model_config_from_dict(cfg["model"])
        if ckpt.get("model_type") == "IFNetJumpAux":
            model = IFNetJumpAux(feature_channels(stft_cfg), sim_cfg.num_components, model_cfg).to(device)
        else:
            model = IFNetUNet(feature_channels(stft_cfg), sim_cfg.num_components, model_cfg).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return model, sim_cfg, stft_cfg, model_cfg


class OraclePerturbedCandidateProvider:
    """Training/debug provider that mimics imperfect stage-1 top-k IF outputs."""

    def __init__(
        self,
        num_candidates: int = 2,
        noise_hz: float = 10.0,
        alt_noise_hz: float = 24.0,
        smooth_kernel: int = 31,
        seed: int = 0,
    ):
        self.num_candidates = max(1, int(num_candidates))
        self.noise_hz = float(noise_hz)
        self.alt_noise_hz = float(alt_noise_hz)
        self.smooth_kernel = int(smooth_kernel)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def __call__(self, signal: torch.Tensor, target_if_hz: torch.Tensor) -> torch.Tensor:
        del signal
        base_noise = torch.randn(target_if_hz.shape, generator=self.generator, device="cpu").to(target_if_hz.device)
        candidates = [target_if_hz + self.noise_hz * base_noise]
        if self.num_candidates > 1:
            alt_noise = torch.randn(target_if_hz.shape, generator=self.generator, device="cpu").to(target_if_hz.device)
            alt = target_if_hz + self.alt_noise_hz * alt_noise
            candidates.append(make_smooth_candidate(alt, kernel_size=self.smooth_kernel))
        while len(candidates) < self.num_candidates:
            extra_noise = torch.randn(target_if_hz.shape, generator=self.generator, device="cpu").to(target_if_hz.device)
            candidates.append(target_if_hz + self.alt_noise_hz * extra_noise)
        return torch.stack(candidates[: self.num_candidates], dim=1)
