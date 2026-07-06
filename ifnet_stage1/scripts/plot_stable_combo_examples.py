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

from ifnet_stage1.model import IFNetUNet, model_config_from_dict, soft_argmax_if
from ifnet_stage1.postprocess import apply_if_postprocess
from ifnet_stage1.predict_routed import DEFAULT_EXPERTS, DEFAULT_POSTPROCESS
from ifnet_stage1.router import (
    DEFAULT_SCENARIO_TO_ROUTE,
    ROUTE_NAMES,
    HardRouteClassifier,
    router_config_from_dict,
    scenario_to_route_labels,
)
from ifnet_stage1.routing_policy import candidate_route_indices, select_best_candidate
from ifnet_stage1.simulation import ChirpSimulator, SCENARIOS, sim_config_from_dict
from ifnet_stage1.tf import feature_channels, log_spectrogram, sample_if_to_frames, stft_config_from_dict


def best_align(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    q = pred.shape[0]
    best_cost = None
    best_perm = tuple(range(q))
    for perm in itertools.permutations(range(q)):
        cost = (pred[list(perm)] - target).abs().mean()
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_perm = perm
    return pred[list(best_perm)], best_perm


def load_expert(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    sim_cfg = sim_config_from_dict(cfg["data"])
    stft_cfg = stft_config_from_dict(cfg["stft"])
    model_cfg = model_config_from_dict(cfg["model"])
    model = IFNetUNet(feature_channels(stft_cfg), sim_cfg.num_components, model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, sim_cfg, stft_cfg, model_cfg


def load_router(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["config"]
    route_names = tuple(ckpt.get("route_names", cfg.get("route_names", ROUTE_NAMES)))
    scenario_to_route = dict(ckpt.get("scenario_to_route", cfg.get("scenario_to_route", DEFAULT_SCENARIO_TO_ROUTE)))
    sim_cfg = sim_config_from_dict(cfg["data"])
    stft_cfg = stft_config_from_dict(cfg["stft"])
    router_cfg = router_config_from_dict(cfg["router"])
    router = HardRouteClassifier(feature_channels(stft_cfg), len(route_names), router_cfg).to(device)
    router.load_state_dict(ckpt["model"])
    router.eval()
    return router, sim_cfg, stft_cfg, route_names, scenario_to_route


@torch.no_grad()
def process_examples(args: argparse.Namespace) -> dict:
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    router, router_sim_cfg, router_stft_cfg, route_names, scenario_to_route = load_router(args.router_checkpoint, device)
    experts = {name: load_expert(path, device) for name, path in DEFAULT_EXPERTS.items() if name in route_names}
    simulator = ChirpSimulator(router_sim_cfg, seed=args.seed)

    out_dir = Path(args.output_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    records = {}
    overview_items = []
    for scenario in args.scenarios:
        batch = simulator.generate_batch(1, device=device, scenarios=[scenario])
        router_feats, _ = log_spectrogram(batch["signal"], router_stft_cfg, router_sim_cfg.fs)
        route_logits = router(router_feats)
        route_probs = torch.softmax(route_logits, dim=1)[0]
        top1_idx = int(route_probs.argmax().detach().cpu())
        candidate_indices, used_fallback = candidate_route_indices(
            route_probs,
            confidence_threshold=args.fallback_confidence,
            margin_threshold=args.fallback_margin,
            topk=args.fallback_topk,
        )

        outputs = {}
        candidates = []
        for candidate_idx in candidate_indices:
            route_name = route_names[candidate_idx]
            model, sim_cfg, stft_cfg, model_cfg = experts[route_name]
            feats, freq_grid = log_spectrogram(batch["signal"], stft_cfg, sim_cfg.fs)
            logits = model(feats)
            pred_if, probs = soft_argmax_if(logits, freq_grid, model_cfg.temperature)
            pred_if = apply_if_postprocess(
                pred_if,
                mode=DEFAULT_POSTPROCESS[route_name],
                degree=3,
                robust_iters=4,
                huber_hz=10.0,
                probs=probs,
                freq_grid=freq_grid,
                topk=9,
            )
            target_if = sample_if_to_frames(batch["if_hz"], logits.shape[-1], stft_cfg.hop_length)
            outputs[route_name] = {
                "pred_if": pred_if,
                "target_if": target_if,
                "probs": probs,
                "feat": feats,
                "freq_grid": freq_grid,
                "stft_cfg": stft_cfg,
                "sim_cfg": sim_cfg,
            }
            candidates.append((candidate_idx, route_name, pred_if, probs))

        selected_idx, selected_name = select_best_candidate(route_probs, candidates) if used_fallback else candidates[0][:2]
        selected = outputs[selected_name]
        pred = selected["pred_if"][0].detach().cpu()
        target = selected["target_if"][0].detach().cpu()
        aligned_pred, perm = best_align(pred, target)
        sample_mae = float((aligned_pred - target).abs().mean())

        true_route_idx = int(
            scenario_to_route_labels(
                [scenario],
                route_names=route_names,
                scenario_to_route=scenario_to_route,
                device=device,
            )[0].detach().cpu()
        )
        image_path = plot_dir / f"{scenario}.png"
        save_plot(
            image_path,
            scenario=scenario,
            signal=batch["signal"][0].detach().cpu(),
            fs=router_sim_cfg.fs,
            feat=selected["feat"][0].mean(dim=0).detach().cpu(),
            freq_grid=selected["freq_grid"].detach().cpu(),
            hop_length=selected["stft_cfg"].hop_length,
            target_if=target,
            pred_if=aligned_pred,
            route_names=route_names,
            route_probs=route_probs.detach().cpu(),
            selected_name=selected_name,
            top1_name=route_names[top1_idx],
            used_fallback=used_fallback,
            mae_hz=sample_mae,
        )
        overview_items.append(
            {
                "scenario": scenario,
                "feat": selected["feat"][0].mean(dim=0).detach().cpu(),
                "freq_grid": selected["freq_grid"].detach().cpu(),
                "hop_length": selected["stft_cfg"].hop_length,
                "fs": selected["sim_cfg"].fs,
                "target_if": target,
                "pred_if": aligned_pred,
                "selected_name": selected_name,
                "mae_hz": sample_mae,
            }
        )
        records[scenario] = {
            "image": str(image_path),
            "sample_if_mae_hz": sample_mae,
            "top_route": route_names[top1_idx],
            "selected_route": selected_name,
            "true_route": route_names[true_route_idx],
            "fallback_used": bool(used_fallback),
            "candidate_routes": [route_names[idx] for idx in candidate_indices],
            "route_probs": {name: float(route_probs[idx]) for idx, name in enumerate(route_names)},
            "component_permutation": list(perm),
        }

    save_overview(out_dir / "all_scenarios_overview.png", overview_items, fs=router_sim_cfg.fs)
    (out_dir / "example_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def save_plot(
    path: Path,
    *,
    scenario: str,
    signal: torch.Tensor,
    fs: float,
    feat: torch.Tensor,
    freq_grid: torch.Tensor,
    hop_length: int,
    target_if: torch.Tensor,
    pred_if: torch.Tensor,
    route_names: tuple[str, ...],
    route_probs: torch.Tensor,
    selected_name: str,
    top1_name: str,
    used_fallback: bool,
    mae_hz: float,
) -> None:
    signal_np = signal.numpy()
    feat_np = feat.numpy()
    freq_np = freq_grid.numpy()
    target_np = target_if.numpy()
    pred_np = pred_if.numpy()
    route_np = route_probs.numpy()
    signal_times = np.arange(signal_np.shape[-1]) / fs
    frame_times = np.arange(target_np.shape[-1]) * hop_length / fs

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.5, 8.4),
        gridspec_kw={"height_ratios": [1.0, 0.85, 3.0]},
        constrained_layout=True,
    )
    axes[0].plot(signal_times, signal_np, color="#243b53", linewidth=0.9)
    axes[0].set_title(f"{scenario} | selected={selected_name} | top={top1_name} | fallback={used_fallback} | MAE={mae_hz:.2f} Hz")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.18)

    bar_colors = ["#2f80ed" if name == selected_name else "#9aa6b2" for name in route_names]
    axes[1].bar(np.arange(len(route_names)), route_np, color=bar_colors)
    axes[1].set_xticks(np.arange(len(route_names)))
    axes[1].set_xticklabels(route_names, rotation=12, ha="right")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Route prob.")
    axes[1].grid(True, axis="y", alpha=0.18)

    axes[2].imshow(
        feat_np,
        origin="lower",
        aspect="auto",
        extent=[frame_times[0], frame_times[-1], freq_np[0], freq_np[-1]],
        cmap="magma",
    )
    for idx in range(target_np.shape[0]):
        axes[2].plot(frame_times, target_np[idx], color="#30d158", linewidth=1.45, label="true IF" if idx == 0 else None)
        axes[2].plot(frame_times, pred_np[idx], color="#4da3ff", linewidth=1.2, linestyle="--", label="pred IF" if idx == 0 else None)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_ylim(float(freq_np[0]), float(freq_np[-1]))
    axes[2].legend(loc="upper right")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_overview(path: Path, items: list[dict], fs: float) -> None:
    cols = 2
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13.0, 4.0 * rows), constrained_layout=True)
    axes_flat = np.atleast_1d(axes).reshape(-1)
    for ax, item in zip(axes_flat, items):
        feat_np = item["feat"].numpy()
        freq_np = item["freq_grid"].numpy()
        target_np = item["target_if"].numpy()
        pred_np = item["pred_if"].numpy()
        frame_times = np.arange(target_np.shape[-1]) * item["hop_length"] / item.get("fs", fs)
        ax.imshow(
            feat_np,
            origin="lower",
            aspect="auto",
            extent=[frame_times[0], frame_times[-1], freq_np[0], freq_np[-1]],
            cmap="magma",
        )
        for idx in range(target_np.shape[0]):
            ax.plot(frame_times, target_np[idx], color="#30d158", linewidth=1.1)
            ax.plot(frame_times, pred_np[idx], color="#4da3ff", linewidth=0.95, linestyle="--")
        ax.set_title(f"{item['scenario']} | {item['selected_name']} | MAE={item['mae_hz']:.2f} Hz")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hz")
        ax.set_ylim(float(freq_np[0]), float(freq_np[-1]))
    for ax in axes_flat[len(items) :]:
        ax.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-checkpoint", default="ifnet_stage1/runs/router_hard_v3/latest.pt")
    parser.add_argument("--output-dir", default="ifnet_stage1/runs/router_hard_v3/stable_combo_all_figures")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--seed", type=int, default=24680)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fallback-confidence", type=float, default=0.78)
    parser.add_argument("--fallback-margin", type=float, default=0.18)
    parser.add_argument("--fallback-topk", type=int, default=2)
    args = parser.parse_args()

    unknown = sorted(set(args.scenarios) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}. Valid: {SCENARIOS}")

    records = process_examples(args)
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
