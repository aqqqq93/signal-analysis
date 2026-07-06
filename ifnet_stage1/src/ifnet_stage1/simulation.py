from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


SCENARIOS = (
    "linear",
    "quadratic",
    "cubic",
    "sinusoidal_fm",
    "crossing",
    "near_parallel",
    "local_jump",
    "tangent_or_overlap",
)


@dataclass
class SimConfig:
    fs: float = 1024.0
    n_samples: int = 1024
    num_components: int = 2
    freq_min: float = 35.0
    freq_max: float = 430.0
    snr_db_min: float = -10.0
    snr_db_max: float = 20.0
    scenario_weights: dict[str, float] | None = None
    noise_types: dict[str, float] | None = None
    scenario_params: dict[str, dict] | None = None


class ChirpSimulator:
    """On-the-fly simulator for supervised IF estimation.

    Frequencies are represented in Hz. Each generated sample returns a fixed
    number of AM-FM components plus the mixture, ground-truth IF curves, and
    clean components.
    """

    def __init__(self, cfg: SimConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(seed)
        self.u = torch.linspace(0.0, 1.0, cfg.n_samples)
        self.t = torch.arange(cfg.n_samples, dtype=torch.float32) / cfg.fs

        weights = cfg.scenario_weights or {name: 1.0 for name in SCENARIOS}
        missing = [name for name in SCENARIOS if name not in weights]
        if missing:
            raise ValueError(f"Missing scenario weights for: {missing}")
        self.scenario_names = list(weights.keys())
        self.scenario_probs = self._normalize_probs([weights[k] for k in self.scenario_names])

        noise_types = cfg.noise_types or {"white": 1.0}
        self.noise_names = list(noise_types.keys())
        self.noise_probs = self._normalize_probs([noise_types[k] for k in self.noise_names])

    def generate_batch(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        scenarios: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor | list[str]]:
        signals = []
        clean = []
        components = []
        if_curves = []
        amp_curves = []
        scenario_out: list[str] = []
        noise_out: list[str] = []

        for idx in range(batch_size):
            scenario = scenarios[idx % len(scenarios)] if scenarios else self._choice(
                self.scenario_names, self.scenario_probs
            )
            noise_name = self._choice(self.noise_names, self.noise_probs)
            if_hz = self._make_if_curves(scenario)
            amps = self._make_amplitudes()
            comps = self._render_components(if_hz, amps)
            clean_sig = comps.sum(dim=0)
            noisy = self._add_noise(clean_sig, noise_name)
            scale = noisy.std().clamp_min(1.0e-6)

            signals.append(noisy / scale)
            clean.append(clean_sig / scale)
            components.append(comps / scale)
            if_curves.append(if_hz)
            amp_curves.append(amps / scale)
            scenario_out.append(scenario)
            noise_out.append(noise_name)

        target_device = torch.device(device)
        return {
            "signal": torch.stack(signals).to(target_device),
            "clean": torch.stack(clean).to(target_device),
            "components": torch.stack(components).to(target_device),
            "if_hz": torch.stack(if_curves).to(target_device),
            "amplitude": torch.stack(amp_curves).to(target_device),
            "scenario": scenario_out,
            "noise_type": noise_out,
        }

    def _make_if_curves(self, scenario: str) -> torch.Tensor:
        maker = getattr(self, f"_scenario_{scenario}", None)
        if maker is None:
            raise ValueError(f"Unknown scenario: {scenario}")
        curves = maker()
        if curves.shape[0] < self.cfg.num_components:
            repeats = self.cfg.num_components - curves.shape[0]
            curves = torch.cat([curves, curves[-1:].repeat(repeats, 1)], dim=0)
        curves = curves[: self.cfg.num_components]
        return curves.clamp(self.cfg.freq_min, self.cfg.freq_max)

    def _scenario_linear(self) -> torch.Tensor:
        q = self.cfg.num_components
        curves = []
        base = self._rand(60.0, 140.0)
        for i in range(q):
            offset = i * self._rand(80.0, 135.0)
            slope = self._rand(-65.0, 90.0)
            curves.append(base + offset + slope * self.u)
        return torch.stack(curves)

    def _scenario_quadratic(self) -> torch.Tensor:
        curves = []
        for i in range(self.cfg.num_components):
            center = self._rand(95.0 + 70.0 * i, 175.0 + 80.0 * i)
            a1 = self._rand(-80.0, 80.0)
            a2 = self._rand(-180.0, 180.0)
            curves.append(center + a1 * (self.u - 0.5) + a2 * (self.u - 0.5).pow(2))
        return torch.stack(curves)

    def _scenario_cubic(self) -> torch.Tensor:
        curves = []
        for i in range(self.cfg.num_components):
            center = self._rand(90.0 + 65.0 * i, 170.0 + 75.0 * i)
            z = self.u - 0.5
            curves.append(
                center
                + self._rand(-95.0, 95.0) * z
                + self._rand(-160.0, 160.0) * z.pow(2)
                + self._rand(-260.0, 260.0) * z.pow(3)
            )
        return torch.stack(curves)

    def _scenario_sinusoidal_fm(self) -> torch.Tensor:
        params = self._scenario_params("sinusoidal_fm")
        curves = []
        for i in range(self.cfg.num_components):
            center = self._rand(95.0 + 80.0 * i, 170.0 + 80.0 * i)
            depth = self._rand_range(params, "depth_hz", 18.0, 65.0)
            rate = self._rand_range(params, "rate_cycles", 0.6, 2.5)
            phase = self._rand(0.0, 2.0 * torch.pi)
            trend = self._rand_range(params, "trend_hz", -35.0, 35.0) * (self.u - 0.5)
            curve = center + trend + depth * torch.sin(2.0 * torch.pi * rate * self.u + phase)
            if self._rand(0.0, 1.0) < float(params.get("second_harmonic_prob", 0.0)):
                harmonic_depth = self._rand_range(params, "second_harmonic_depth_hz", 5.0, 22.0)
                harmonic_rate = rate * self._rand_range(params, "second_harmonic_rate_scale", 1.6, 2.4)
                harmonic_phase = self._rand(0.0, 2.0 * torch.pi)
                curve = curve + harmonic_depth * torch.sin(2.0 * torch.pi * harmonic_rate * self.u + harmonic_phase)
            curves.append(curve)
        return torch.stack(curves)

    def _scenario_crossing(self) -> torch.Tensor:
        low = self._rand(65.0, 130.0)
        high = self._rand(270.0, 390.0)
        wiggle = self._rand(0.0, 16.0) * torch.sin(2.0 * torch.pi * self._rand(0.7, 1.4) * self.u)
        first = low + (high - low) * self.u + wiggle
        second = high - (high - low) * self.u - 0.6 * wiggle
        curves = [first, second]
        for i in range(2, self.cfg.num_components):
            curves.append(self._rand(80.0, 360.0) + self._rand(-35.0, 35.0) * self.u)
        return torch.stack(curves)

    def _scenario_near_parallel(self) -> torch.Tensor:
        base = self._rand(90.0, 230.0)
        gap = self._rand(12.0, 32.0)
        common = (
            base
            + self._rand(-45.0, 60.0) * (self.u - 0.5)
            + self._rand(15.0, 45.0) * torch.sin(2.0 * torch.pi * self._rand(0.4, 1.2) * self.u)
        )
        curves = [common + (i - 0.5) * gap for i in range(self.cfg.num_components)]
        return torch.stack(curves)

    def _scenario_local_jump(self) -> torch.Tensor:
        params = self._scenario_params("local_jump")
        curves = []
        for i in range(self.cfg.num_components):
            base = self._rand(80.0 + 75.0 * i, 155.0 + 75.0 * i)
            slope = self._rand_range(params, "slope_hz", -60.0, 70.0)
            jump = self._rand_signed_range(params, "jump_hz", -95.0, 95.0, "jump_abs_min_hz")
            if i > 0 and self._rand(0.0, 1.0) > float(params.get("component_jump_prob", 1.0)):
                jump = 0.0
            center = self._rand_range(params, "center", 0.25, 0.75)
            width = self._rand_range(params, "width", 0.006, 0.025)
            step = torch.sigmoid((self.u - center) / width)
            bump_width = self._rand_range(params, "bump_width", 0.025, 0.055)
            local_bump = self._rand_range(params, "bump_hz", -35.0, 35.0) * torch.exp(
                -0.5 * ((self.u - center) / bump_width).pow(2)
            )
            curves.append(base + slope * self.u + jump * step + local_bump)
        return torch.stack(curves)

    def _scenario_tangent_or_overlap(self) -> torch.Tensor:
        if self._rand(0.0, 1.0) < 0.5:
            return self._tangent_curves()
        return self._short_overlap_curves()

    def _tangent_curves(self) -> torch.Tensor:
        c = self._rand(0.35, 0.65)
        fc = self._rand(145.0, 260.0)
        tilt = self._rand(-35.0, 35.0)
        z = self.u - c
        curv1 = self._rand(280.0, 520.0)
        curv2 = self._rand(70.0, 210.0)
        first = fc + tilt * z + curv1 * z.pow(2)
        second = fc + tilt * z + curv2 * z.pow(2)
        curves = [first, second]
        for i in range(2, self.cfg.num_components):
            curves.append(fc + 90.0 + self._rand(-50.0, 50.0) * self.u)
        return torch.stack(curves)

    def _short_overlap_curves(self) -> torch.Tensor:
        c = self._rand(0.35, 0.65)
        half_width = self._rand(0.045, 0.09)
        edge = self._rand(0.006, 0.018)
        common = (
            self._rand(140.0, 250.0)
            + self._rand(-55.0, 55.0) * (self.u - 0.5)
            + self._rand(12.0, 38.0) * torch.sin(2.0 * torch.pi * self._rand(0.5, 1.1) * self.u)
        )
        left = torch.sigmoid((self.u - (c - half_width)) / edge)
        right = torch.sigmoid(((c + half_width) - self.u) / edge)
        overlap_window = left * right
        gap = self._rand(45.0, 95.0) * (1.0 - overlap_window)
        curves = [common - 0.5 * gap, common + 0.5 * gap]
        for i in range(2, self.cfg.num_components):
            curves.append(common + 100.0 + self._rand(-25.0, 25.0) * self.u)
        return torch.stack(curves)

    def _make_amplitudes(self) -> torch.Tensor:
        amps = []
        for i in range(self.cfg.num_components):
            base = self._rand(0.55, 1.15)
            if i > 0:
                db_drop = self._rand(0.0, 12.0)
                base = base * float(10.0 ** (-db_drop / 20.0))
            mod_depth = self._rand(0.04, 0.28)
            rate = self._rand(0.3, 2.0)
            phase = self._rand(0.0, 2.0 * torch.pi)
            envelope = base * (1.0 + mod_depth * torch.sin(2.0 * torch.pi * rate * self.u + phase))
            if self._rand(0.0, 1.0) < 0.35:
                center = self._rand(0.18, 0.82)
                width = self._rand(0.04, 0.16)
                envelope = envelope * (0.35 + 0.65 * torch.exp(-0.5 * ((self.u - center) / width).pow(2)))
            amps.append(envelope.clamp_min(0.03))
        return torch.stack(amps)

    def _render_components(self, if_hz: torch.Tensor, amps: torch.Tensor) -> torch.Tensor:
        phase0 = 2.0 * torch.pi * torch.rand(self.cfg.num_components, generator=self.rng)
        phase = 2.0 * torch.pi * torch.cumsum(if_hz / self.cfg.fs, dim=-1) + phase0[:, None]
        return amps * torch.cos(phase)

    def _add_noise(self, clean: torch.Tensor, noise_name: str) -> torch.Tensor:
        snr = self._rand(self.cfg.snr_db_min, self.cfg.snr_db_max)
        noise = torch.randn(clean.shape, generator=self.rng)
        if noise_name == "colored":
            kernel_size = int(self._rand(7.0, 23.0))
            kernel_size = max(3, kernel_size | 1)
            kernel = torch.hann_window(kernel_size)
            kernel = kernel / kernel.sum().clamp_min(1.0e-6)
            noise = F.conv1d(
                noise.view(1, 1, -1),
                kernel.view(1, 1, -1),
                padding=kernel_size // 2,
            ).view(-1)
        elif noise_name == "impulsive":
            mask = torch.rand(clean.shape, generator=self.rng) < self._rand(0.002, 0.014)
            spikes = torch.randn(clean.shape, generator=self.rng) * self._rand(4.0, 10.0)
            noise = noise + mask.float() * spikes
        elif noise_name == "trend":
            trend = self._rand(-1.0, 1.0) * (self.u - 0.5)
            trend = trend + self._rand(0.1, 0.8) * torch.sin(2.0 * torch.pi * self._rand(0.4, 1.2) * self.u)
            noise = noise + trend
        elif noise_name != "white":
            raise ValueError(f"Unknown noise type: {noise_name}")

        sig_power = clean.pow(2).mean().clamp_min(1.0e-9)
        noise_power = noise.pow(2).mean().clamp_min(1.0e-9)
        target_noise_power = sig_power / (10.0 ** (float(snr) / 10.0))
        noise = noise * torch.sqrt(target_noise_power / noise_power)
        return clean + noise

    def _choice(self, names: Sequence[str], probs: torch.Tensor) -> str:
        idx = torch.multinomial(probs, 1, generator=self.rng).item()
        return names[idx]

    @staticmethod
    def _normalize_probs(values: Sequence[float]) -> torch.Tensor:
        probs = torch.tensor(values, dtype=torch.float32)
        probs = probs.clamp_min(0)
        if probs.sum() <= 0:
            raise ValueError("At least one sampling weight must be positive.")
        return probs / probs.sum()

    def _rand(self, low: float, high: float) -> float:
        return float(torch.empty((), dtype=torch.float32).uniform_(low, high, generator=self.rng))

    def _scenario_params(self, scenario: str) -> dict:
        if not self.cfg.scenario_params:
            return {}
        value = self.cfg.scenario_params.get(scenario, {})
        return dict(value) if isinstance(value, dict) else {}

    def _rand_range(self, params: dict, key: str, low: float, high: float) -> float:
        value = params.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return self._rand(float(value[0]), float(value[1]))
        if isinstance(value, (int, float)):
            return float(value)
        return self._rand(low, high)

    def _rand_signed_range(
        self,
        params: dict,
        key: str,
        low: float,
        high: float,
        min_abs_key: str,
    ) -> float:
        min_abs = abs(float(params.get(min_abs_key, 0.0)))
        for _ in range(12):
            value = self._rand_range(params, key, low, high)
            if abs(value) >= min_abs:
                return value
        sign = -1.0 if self._rand(0.0, 1.0) < 0.5 else 1.0
        upper = max(abs(float(low)), abs(float(high)), min_abs)
        return sign * self._rand(min_abs, upper)


def sim_config_from_dict(data: dict) -> SimConfig:
    return SimConfig(
        fs=float(data.get("fs", 1024.0)),
        n_samples=int(data.get("n_samples", 1024)),
        num_components=int(data.get("num_components", 2)),
        freq_min=float(data.get("freq_min", 35.0)),
        freq_max=float(data.get("freq_max", 430.0)),
        snr_db_min=float(data.get("snr_db_min", -10.0)),
        snr_db_max=float(data.get("snr_db_max", 20.0)),
        scenario_weights=dict(data.get("scenario_weights", {name: 1.0 for name in SCENARIOS})),
        noise_types=dict(data.get("noise_types", {"white": 1.0})),
        scenario_params=dict(data.get("scenario_params", {})),
    )
