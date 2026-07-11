from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import trange

from ifnet_stage1.config import choose_device
from ifnet_stage1.simulation import SCENARIOS, ChirpSimulator, sim_config_from_dict
from ifnet_stage1.tf import feature_channels, log_spectrogram, stft_config_from_dict

from .pipeline import P15PipelineConfig, P15Stage2Pipeline
from .train_stage2 import masked_permutation_l1


class TinyIFNet(nn.Module):
    def __init__(self, in_channels: int, num_components: int = 2, hidden: int = 32, freq_min: float = 35.0, freq_max: float = 430.0):
        super().__init__()
        self.num_components = int(num_components)
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.AvgPool2d((2, 1)),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden, self.num_components, kernel_size=5, padding=2),
        )

    def forward(self, features: torch.Tensor, n_samples: int) -> torch.Tensor:
        h = self.encoder(features)
        h = h.mean(dim=2)
        h = F.interpolate(h, size=n_samples, mode="linear", align_corners=False)
        raw = self.head(h)
        return self.freq_min + (self.freq_max - self.freq_min) * torch.sigmoid(raw)


def train_tiny_distill(
    run_dir: str | Path,
    steps: int = 120,
    batch_size: int = 8,
    device_name: str = "auto",
    scenarios: list[str] | None = None,
) -> dict[str, Any]:
    device = choose_device(device_name)
    teacher = P15Stage2Pipeline(P15PipelineConfig(use_scenario_hints=True), device=device)
    reference_cfg = teacher.branches["all_expert"].cfg
    sim_cfg = sim_config_from_dict(reference_cfg["data"])
    stft_cfg = stft_config_from_dict(teacher.active_cfg["stft"])
    scenario_weights = {name: 0.0 for name in SCENARIOS}
    selected = scenarios or ["linear", "quadratic", "cubic", "near_parallel", "local_jump"]
    for name in selected:
        scenario_weights[name] = 1.0
    sim_data = dict(reference_cfg["data"])
    sim_data["scenario_weights"] = scenario_weights
    sim_data["active_components"] = {1: 0.45, 2: 0.55}
    simulator = ChirpSimulator(sim_config_from_dict(sim_data), seed=20260802)
    model = TinyIFNet(
        feature_channels(stft_cfg),
        num_components=sim_cfg.num_components,
        hidden=32,
        freq_min=sim_cfg.freq_min,
        freq_max=sim_cfg.freq_max,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=1.0e-5)
    history = []
    for step in trange(1, int(steps) + 1, desc="tiny-distill"):
        batch = simulator.generate_batch(batch_size, device=device)
        feats, _ = log_spectrogram(batch["signal"], stft_cfg, sim_cfg.fs)
        with torch.no_grad():
            teacher_out, _route = teacher.run(batch, sim_cfg.fs, scenario_hints=batch["scenario"])
            target_if = teacher_out["identity_stable_if_hz"].detach()
        pred_if = model(feats, sim_cfg.n_samples)
        loss_if = masked_permutation_l1(pred_if, target_if, batch.get("active_mask"))
        smooth = (pred_if[..., 2:] - 2.0 * pred_if[..., 1:-1] + pred_if[..., :-2]).pow(2).mean()
        loss = loss_if / sim_cfg.fs + 0.0005 * smooth / (sim_cfg.fs**2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 10 == 0 or step == steps:
            history.append({"step": step, "loss": float(loss.detach().cpu()), "if_mae_hz": float(loss_if.detach().cpu())})

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "stft": teacher.active_cfg["stft"],
        "data": sim_data,
        "history": history,
        "num_components": sim_cfg.num_components,
        "freq_min": sim_cfg.freq_min,
        "freq_max": sim_cfg.freq_max,
    }
    torch.save(payload, run_path / "latest.pt")
    (run_path / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"run_dir": str(run_path), "last": history[-1] if history else {}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="stage2_iccd/runs/tiny_ifnet_distill_p2")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--scenarios", nargs="*", default=None)
    args = parser.parse_args()
    result = train_tiny_distill(
        run_dir=args.run_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        device_name=args.device,
        scenarios=args.scenarios,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
