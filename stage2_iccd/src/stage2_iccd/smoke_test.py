from __future__ import annotations

import json

import torch

from ifnet_stage1.simulation import ChirpSimulator, SimConfig

from .candidates import OraclePerturbedCandidateProvider
from .differentiable_iccd import ICCDConfig
from .losses import reconstruction_snr_db
from .model import Stage2ICCDModel, Stage2ModelConfig


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim_cfg = SimConfig(
        n_samples=256,
        num_components=2,
        snr_db_min=12.0,
        snr_db_max=20.0,
        scenario_weights={
            "linear": 1.0,
            "quadratic": 1.0,
            "cubic": 1.0,
            "sinusoidal_fm": 1.0,
            "crossing": 1.0,
            "near_parallel": 1.0,
            "local_jump": 1.0,
            "tangent_or_overlap": 1.0,
        },
    )
    simulator = ChirpSimulator(sim_cfg, seed=7)
    batch = simulator.generate_batch(3, device=device)
    provider = OraclePerturbedCandidateProvider(num_candidates=2, noise_hz=5.0, alt_noise_hz=14.0, seed=7)
    candidates = provider(batch["signal"], batch["if_hz"]).clamp(sim_cfg.freq_min, sim_cfg.freq_max)
    model = Stage2ICCDModel(
        ICCDConfig(fs=sim_cfg.fs, n_samples=sim_cfg.n_samples, amplitude_order=5, alpha_init=0.5, freq_min=sim_cfg.freq_min, freq_max=sim_cfg.freq_max),
        Stage2ModelConfig(num_candidates=2, refine_channels=12, refine_layers=1, max_refine_hz=8.0, freq_min=sim_cfg.freq_min, freq_max=sim_cfg.freq_max),
        num_components=sim_cfg.num_components,
    ).to(device)
    out = model(batch["signal"], candidates)
    loss = (out["reconstruction"] - batch["clean"]).pow(2).mean()
    loss.backward()
    result = {
        "loss": float(loss.detach().cpu()),
        "snr_db": float(reconstruction_snr_db(batch["clean"], out["reconstruction"]).mean().detach().cpu()),
        "alpha_grad_ok": model.iccd.raw_alpha.grad is not None,
        "refine_grad_ok": any(param.grad is not None for param in model.refine_head.parameters()),
        "candidate_weights": [float(v) for v in out["candidate_weights"].detach().cpu()],
        "output_shape": list(out["components"].shape),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
