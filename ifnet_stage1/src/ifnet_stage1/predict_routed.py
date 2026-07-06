from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat
import torch

from .model import IFNetUNet, model_config_from_dict, soft_argmax_if
from .postprocess import apply_if_postprocess
from .quality_selector import QualitySelector, quality_selector_config_from_dict, select_candidate_with_quality
from .router import ROUTE_NAMES, HardRouteClassifier, router_config_from_dict
from .routing_policy import candidate_route_indices, select_best_candidate
from .simulation import sim_config_from_dict
from .tf import feature_channels, log_spectrogram, stft_config_from_dict


DEFAULT_EXPERTS = {
    "poly_like": "ifnet_stage1/runs/polynomial_refit_resume/latest.pt",
    "sinusoidal_like": "ifnet_stage1/runs/sinusoidal_refit/latest.pt",
    "cross_overlap_like": "ifnet_stage1/runs/balanced_refit_resume/latest.pt",
    "jump_like": "ifnet_stage1/runs/local_jump_refit/latest.pt",
}

DEFAULT_POSTPROCESS = {
    "poly_like": "poly_prob",
    "sinusoidal_like": "none",
    "cross_overlap_like": "none",
    "jump_like": "none",
}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Predict IF curves with a hard router and expert IF-Net checkpoints.")
    parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Path to .npy array shaped [N] or [B, N].")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--poly-checkpoint", default=DEFAULT_EXPERTS["poly_like"])
    parser.add_argument("--sinusoidal-checkpoint", default=DEFAULT_EXPERTS["sinusoidal_like"])
    parser.add_argument("--cross-checkpoint", default=DEFAULT_EXPERTS["cross_overlap_like"])
    parser.add_argument("--jump-checkpoint", default=DEFAULT_EXPERTS["jump_like"])
    parser.add_argument("--poly-topk", type=int, default=9)
    parser.add_argument("--poly-degree", type=int, default=3)
    parser.add_argument("--poly-robust-iters", type=int, default=4)
    parser.add_argument("--poly-huber-hz", type=float, default=10.0)
    parser.add_argument("--fallback", action="store_true", help="Run top-2 experts for low-confidence router decisions.")
    parser.add_argument("--fallback-confidence", type=float, default=0.78)
    parser.add_argument("--fallback-margin", type=float, default=0.18)
    parser.add_argument("--fallback-topk", type=int, default=2)
    parser.add_argument("--quality-selector-checkpoint", default=None)
    parser.add_argument("--quality-selector-margin", type=float, default=0.10)
    parser.add_argument("--quality-protect-top-routes", nargs="*", default=[])
    parser.add_argument("--quality-protect-min-prob", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    arr = np.load(args.input).astype("float32")
    if arr.ndim == 1:
        arr = arr[None, :]
    signal = torch.from_numpy(arr).to(device)

    router_ckpt = torch.load(args.router_checkpoint, map_location="cpu")
    router_cfg_all = router_ckpt["config"]
    route_names = tuple(router_ckpt.get("route_names", router_cfg_all.get("route_names", ROUTE_NAMES)))
    router_sim_cfg = sim_config_from_dict(router_cfg_all["data"])
    router_stft_cfg = stft_config_from_dict(router_cfg_all["stft"])
    router_cfg = router_config_from_dict(router_cfg_all["router"])
    router = HardRouteClassifier(feature_channels(router_stft_cfg), len(route_names), router_cfg).to(device)
    router.load_state_dict(router_ckpt["model"])
    router.eval()

    router_feats, _ = log_spectrogram(signal, router_stft_cfg, router_sim_cfg.fs)
    route_logits = router(router_feats)
    route_probs = torch.softmax(route_logits, dim=1)
    route_idx = route_probs.argmax(dim=1)

    expert_paths = {
        "poly_like": args.poly_checkpoint,
        "sinusoidal_like": args.sinusoidal_checkpoint,
        "cross_overlap_like": args.cross_checkpoint,
        "jump_like": args.jump_checkpoint,
    }
    loaded_experts = {
        route_name: load_expert(path, device)
        for route_name, path in expert_paths.items()
        if route_name in route_names
    }
    quality_selector = load_quality_selector(args.quality_selector_checkpoint, device) if args.quality_selector_checkpoint else None

    pred_chunks = []
    probs_chunks = []
    route_labels = []
    top_route_labels = []
    expert_used = []
    fallback_used = []
    freq_grid_ref = None
    for sample_idx in range(signal.shape[0]):
        top_route_name = route_names[int(route_idx[sample_idx].detach().cpu())]
        top_route_labels.append(top_route_name)
        if args.fallback:
            candidate_indices, used_fallback = candidate_route_indices(
                route_probs[sample_idx],
                confidence_threshold=args.fallback_confidence,
                margin_threshold=args.fallback_margin,
                topk=args.fallback_topk,
            )
        else:
            candidate_indices, used_fallback = [int(route_idx[sample_idx].detach().cpu())], False

        outputs = {}
        candidates = []
        for candidate_idx in candidate_indices:
            candidate_name = route_names[candidate_idx]
            model, sim_cfg, stft_cfg, model_cfg = loaded_experts[candidate_name]
            feats, freq_grid = log_spectrogram(signal[sample_idx : sample_idx + 1], stft_cfg, sim_cfg.fs)
            logits = model(feats)
            candidate_pred_if, candidate_probs = soft_argmax_if(logits, freq_grid, model_cfg.temperature)
            candidate_pred_if = apply_if_postprocess(
                candidate_pred_if,
                mode=DEFAULT_POSTPROCESS[candidate_name],
                degree=args.poly_degree,
                robust_iters=args.poly_robust_iters,
                huber_hz=args.poly_huber_hz,
                probs=candidate_probs,
                freq_grid=freq_grid,
                topk=args.poly_topk,
            )
            outputs[candidate_name] = (candidate_pred_if, candidate_probs, freq_grid)
            candidates.append((candidate_idx, candidate_name, candidate_pred_if, candidate_probs))

        if used_fallback and quality_selector is not None:
            _, route_name = select_candidate_with_quality(
                quality_selector,
                route_probs[sample_idx],
                candidates,
                route_names,
                margin=args.quality_selector_margin,
                protect_top_routes=set(args.quality_protect_top_routes),
                protect_min_prob=args.quality_protect_min_prob,
            )
        else:
            _, route_name = select_best_candidate(route_probs[sample_idx], candidates) if used_fallback else candidates[0][:2]
        pred_if, probs, freq_grid = outputs[route_name]
        route_labels.append(route_name)
        fallback_used.append(used_fallback)
        expert_used.append(expert_paths[route_name])
        pred_chunks.append(pred_if.detach().cpu())
        probs_chunks.append(probs.detach().cpu())
        if freq_grid_ref is None:
            freq_grid_ref = freq_grid.detach().cpu()

    payload = {
        "if_hz": torch.cat(pred_chunks, dim=0).numpy(),
        "freq_grid": freq_grid_ref.numpy() if freq_grid_ref is not None else np.array([], dtype=np.float32),
        "ridge_probs": torch.cat(probs_chunks, dim=0).numpy(),
        "route_probs": route_probs.detach().cpu().numpy(),
        "route_index": route_idx.detach().cpu().numpy(),
        "route_names": np.array(route_names, dtype=object),
        "top_route": np.array(top_route_labels, dtype=object),
        "selected_route": np.array(route_labels, dtype=object),
        "fallback_used": np.array(fallback_used, dtype=bool),
        "expert_checkpoint": np.array(expert_used, dtype=object),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.output).suffix.lower() == ".mat":
        savemat(args.output, payload)
    else:
        np.savez(args.output, **payload)


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


def load_quality_selector(path: str | Path, device: torch.device) -> QualitySelector:
    ckpt = torch.load(path, map_location="cpu")
    selector = QualitySelector(quality_selector_config_from_dict(ckpt["config"].get("quality_selector")), ckpt["feature_dim"]).to(device)
    selector.load_state_dict(ckpt["model"])
    selector.eval()
    return selector


if __name__ == "__main__":
    main()
