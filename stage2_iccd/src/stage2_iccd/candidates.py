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
        trainable: bool = False,
        unfreeze_last_decoders: int = 0,
        unfreeze_head: bool = True,
    ):
        paths = list(checkpoints or [])
        if checkpoint is not None:
            paths.insert(0, checkpoint)
        if not paths:
            raise ValueError("At least one frozen IF-Net checkpoint is required.")
        self.trainable = bool(trainable)
        self.experts = [self._load_one(path, device) for path in paths]
        self.device = device
        self.num_candidates = max(1, int(num_candidates))
        self.smooth_kernel = int(smooth_kernel)
        for model, _sim_cfg, _stft_cfg, _model_cfg in self.experts:
            self._set_trainable_layers(
                model,
                trainable=self.trainable,
                unfreeze_last_decoders=int(unfreeze_last_decoders),
                unfreeze_head=bool(unfreeze_head),
            )

    def __call__(self, signal: torch.Tensor, n_samples: int) -> torch.Tensor:
        candidates = []
        grad_context = torch.enable_grad() if self.trainable else torch.no_grad()
        with grad_context:
            for model, sim_cfg, stft_cfg, model_cfg in self.experts:
                if self.trainable:
                    model.train()
                else:
                    model.eval()
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

    def parameters(self):
        for model, _sim_cfg, _stft_cfg, _model_cfg in self.experts:
            yield from model.parameters()

    def trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {f"expert_{idx}": model.state_dict() for idx, (model, *_rest) in enumerate(self.experts)}

    def load_state_dict(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        for idx, (model, *_rest) in enumerate(self.experts):
            key = f"expert_{idx}"
            if key in state:
                model.load_state_dict(state[key])

    @staticmethod
    def _set_trainable_layers(
        model: torch.nn.Module,
        trainable: bool,
        unfreeze_last_decoders: int,
        unfreeze_head: bool,
    ) -> None:
        for param in model.parameters():
            param.requires_grad_(False)
        if not trainable:
            model.eval()
            return
        if unfreeze_head and hasattr(model, "head"):
            for param in model.head.parameters():
                param.requires_grad_(True)
        decoders = getattr(model, "decoders", None)
        if decoders is not None and unfreeze_last_decoders > 0:
            for block in list(decoders)[-unfreeze_last_decoders:]:
                for param in block.parameters():
                    param.requires_grad_(True)
        model.train()

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


class STFTPeakCandidateProvider:
    """Deployment-safe IF candidates from STFT peak ridges.

    This provider does not use ground-truth IF. It extracts the strongest
    separated frequency peaks in each STFT frame, converts them to coarse IF
    tracks, and returns raw/smoothed variants as Stage2 candidates.
    """

    signal_only = True

    def __init__(
        self,
        num_components: int,
        num_candidates: int = 2,
        fs: float = 1024.0,
        n_fft: int = 256,
        win_length: int = 128,
        hop_length: int = 4,
        freq_min: float = 35.0,
        freq_max: float = 430.0,
        min_gap_hz: float = 24.0,
        centroid_radius: int = 2,
        smooth_kernel: int = 31,
        alt_smooth_kernel: int = 61,
        device: torch.device | None = None,
    ):
        self.num_components = max(1, int(num_components))
        self.num_candidates = max(1, int(num_candidates))
        self.fs = float(fs)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self.min_gap_hz = float(min_gap_hz)
        self.centroid_radius = max(0, int(centroid_radius))
        self.smooth_kernel = int(smooth_kernel)
        self.alt_smooth_kernel = int(alt_smooth_kernel)
        self.device = device

    def __call__(self, signal: torch.Tensor, n_samples: int) -> torch.Tensor:
        device = signal.device if self.device is None else self.device
        signal = signal.to(device)
        window = torch.hann_window(self.win_length, device=device, dtype=signal.dtype)
        spec = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        mag = spec.abs()
        freqs = torch.linspace(0.0, self.fs / 2.0, self.n_fft // 2 + 1, device=device, dtype=signal.dtype)
        freq_mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        if int(freq_mask.sum().item()) < self.num_components:
            raise ValueError("STFT frequency mask is too narrow for the requested component count.")
        local_mag = mag[:, freq_mask, :]
        local_freqs = freqs[freq_mask]
        frame_tracks = self._frame_peaks(local_mag, local_freqs)
        frame_tracks = frame_tracks.sort(dim=1).values
        tracks = F.interpolate(frame_tracks, size=n_samples, mode="linear", align_corners=False)
        tracks = tracks.clamp(self.freq_min, self.freq_max)

        candidates = [tracks]
        if self.num_candidates > 1:
            candidates.append(make_smooth_candidate(tracks, kernel_size=self.smooth_kernel))
        if self.num_candidates > 2:
            candidates.append(make_smooth_candidate(tracks, kernel_size=self.alt_smooth_kernel))
        while len(candidates) < self.num_candidates:
            candidates.append(candidates[-1].clone())
        return torch.stack(candidates[: self.num_candidates], dim=1)

    def _frame_peaks(self, mag: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        bsz, num_bins, num_frames = mag.shape
        work = mag.clone()
        tracks = []
        bin_hz = float(freqs[1].detach().cpu() - freqs[0].detach().cpu()) if num_bins > 1 else self.min_gap_hz
        suppress_bins = max(1, int(round(self.min_gap_hz / max(bin_hz, 1.0e-6))))
        flat_work = work.permute(0, 2, 1).reshape(-1, num_bins)
        flat_mag = mag.permute(0, 2, 1).reshape(-1, num_bins)
        rows = torch.arange(flat_work.shape[0], device=mag.device)
        floor = flat_work.new_full((flat_work.shape[0], 1), float("-inf"))

        for _ in range(self.num_components):
            peak_idx = flat_work.argmax(dim=1)
            peak_freq = self._centroid_frequency(flat_mag, freqs, peak_idx, rows)
            tracks.append(peak_freq.view(bsz, num_frames))
            for offset in range(-suppress_bins, suppress_bins + 1):
                idx = (peak_idx + offset).clamp(0, num_bins - 1).view(-1, 1)
                flat_work.scatter_(1, idx, floor)
        return torch.stack(tracks, dim=1)

    def _centroid_frequency(
        self,
        flat_mag: torch.Tensor,
        freqs: torch.Tensor,
        peak_idx: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        if self.centroid_radius <= 0:
            return freqs[peak_idx]
        weights = []
        values = []
        for offset in range(-self.centroid_radius, self.centroid_radius + 1):
            idx = (peak_idx + offset).clamp(0, freqs.numel() - 1)
            weight = flat_mag[rows, idx]
            weights.append(weight)
            values.append(weight * freqs[idx])
        weight_sum = torch.stack(weights, dim=0).sum(dim=0).clamp_min(1.0e-8)
        return torch.stack(values, dim=0).sum(dim=0) / weight_sum
