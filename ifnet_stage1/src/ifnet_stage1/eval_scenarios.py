from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .losses import permutation_l1
from .model import IFNetUNet, model_config_from_dict, soft_argmax_if
from .postprocess import apply_if_postprocess
from .simulation import ChirpSimulator, SCENARIOS, sim_config_from_dict
from .tf import feature_channels, log_spectrogram, sample_if_to_frames, stft_config_from_dict


def best_align(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Align predicted components to target components for one sample.

    pred, target: [Q, T]
    """

    q = pred.shape[0]
    best_cost = None
    best_perm = tuple(range(q))
    for perm in itertools.permutations(range(q)):
        cost = (pred[list(perm)] - target).abs().mean()
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm
    return pred[list(best_perm)], best_perm


@torch.no_grad()
def evaluate_scenarios(
    checkpoint: str | Path,
    scenarios: list[str],
    output_dir: str | Path,
    batch_size: int,
    batches: int,
    seed: int,
    device_name: str,
    postprocess: str = "none",
    poly_degree: int = 3,
    poly_topk: int = 7,
    poly_robust_iters: int = 2,
    poly_huber_hz: float = 12.0,
) -> dict[str, dict[str, float]]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    sim_cfg = sim_config_from_dict(cfg["data"])
    stft_cfg = stft_config_from_dict(cfg["stft"])
    model_cfg = model_config_from_dict(cfg["model"])
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )

    simulator = ChirpSimulator(sim_cfg, seed=seed)
    model = IFNetUNet(feature_channels(stft_cfg), sim_cfg.num_components, model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = Path(output_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, float]] = {}
    for scenario in scenarios:
        maes = []
        all_pred = []
        all_target = []
        example = None
        for _ in range(batches):
            batch = simulator.generate_batch(batch_size, device=device, scenarios=[scenario])
            feats, freq_grid = log_spectrogram(batch["signal"], stft_cfg, sim_cfg.fs)
            logits = model(feats)
            target_if = sample_if_to_frames(batch["if_hz"], logits.shape[-1], stft_cfg.hop_length)
            pred_if, probs = soft_argmax_if(logits, freq_grid, model_cfg.temperature)
            pred_if = apply_if_postprocess(
                pred_if,
                mode=postprocess,
                degree=poly_degree,
                robust_iters=poly_robust_iters,
                huber_hz=poly_huber_hz,
                probs=probs,
                freq_grid=freq_grid,
                topk=poly_topk,
            )
            maes.append(permutation_l1(pred_if, target_if).detach().cpu())
            all_pred.append(pred_if.detach().cpu())
            all_target.append(target_if.detach().cpu())
            if example is None:
                example = {
                    "feat": feats[0].mean(dim=0).detach().cpu(),
                    "freq_grid": freq_grid.detach().cpu(),
                    "pred": pred_if[0].detach().cpu(),
                    "target": target_if[0].detach().cpu(),
                    "probs": probs[0].detach().cpu(),
                }

        pred_cat = torch.cat(all_pred, dim=0)
        target_cat = torch.cat(all_target, dim=0)
        sample_maes = []
        for idx in range(pred_cat.shape[0]):
            aligned, _ = best_align(pred_cat[idx], target_cat[idx])
            sample_maes.append((aligned - target_cat[idx]).abs().mean().item())

        results[scenario] = {
            "if_mae_hz": float(torch.stack(maes).mean()),
            "if_mae_hz_median": float(np.median(sample_maes)),
            "if_mae_hz_p90": float(np.percentile(sample_maes, 90)),
            "if_mae_hz_p95": float(np.percentile(sample_maes, 95)),
            "if_mae_hz_std": float(np.std(sample_maes)),
            "if_mae_hz_max": float(np.max(sample_maes)),
            "num_examples": int(len(sample_maes)),
        }
        if example is not None:
            save_example_plot(example, plot_dir / f"{scenario}.png", scenario, stft_cfg.hop_length, sim_cfg.fs)

    (out_dir / "scenario_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def save_example_plot(example: dict, path: Path, scenario: str, hop_length: int, fs: float) -> None:
    feat = example["feat"].numpy()
    freq_grid = example["freq_grid"].numpy()
    pred = example["pred"]
    target = example["target"]
    aligned_pred, _ = best_align(pred, target)
    pred_np = aligned_pred.numpy()
    target_np = target.numpy()
    times = np.arange(target_np.shape[-1]) * hop_length / fs

    plt.figure(figsize=(9.5, 4.8))
    plt.imshow(
        feat,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], freq_grid[0], freq_grid[-1]],
        cmap="magma",
    )
    for idx in range(target_np.shape[0]):
        plt.plot(times, target_np[idx], color="#38d973", linewidth=1.4, label="true IF" if idx == 0 else None)
        plt.plot(times, pred_np[idx], color="#4da3ff", linewidth=1.2, linestyle="--", label="pred IF" if idx == 0 else None)
    plt.title(f"IF-Net ridge estimate: {scenario}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(float(freq_grid[0]), float(freq_grid[-1]))
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenarios", nargs="+", default=["linear", "quadratic", "cubic"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--postprocess",
        default="none",
        choices=[
            "none",
            "poly",
            "polynomial",
            "poly3",
            "poly_prob",
            "poly_heatmap",
            "prob_poly",
            "despike",
            "median_spike",
            "jump_despike",
        ],
    )
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument("--poly-topk", type=int, default=7)
    parser.add_argument("--poly-robust-iters", type=int, default=2)
    parser.add_argument("--poly-huber-hz", type=float, default=12.0)
    args = parser.parse_args()

    unknown = sorted(set(args.scenarios) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}. Valid: {SCENARIOS}")

    results = evaluate_scenarios(
        checkpoint=args.checkpoint,
        scenarios=args.scenarios,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        batches=args.batches,
        seed=args.seed,
        device_name=args.device,
        postprocess=args.postprocess,
        poly_degree=args.poly_degree,
        poly_topk=args.poly_topk,
        poly_robust_iters=args.poly_robust_iters,
        poly_huber_hz=args.poly_huber_hz,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
