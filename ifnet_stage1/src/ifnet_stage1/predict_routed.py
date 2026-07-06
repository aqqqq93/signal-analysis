from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat
import torch

from .confidence import combined_initial_confidence, ridge_confidence_features, route_confidence_features
from .jump_aux import IFNetJumpAux
from .model import IFNetUNet, model_config_from_dict, soft_argmax_if
from .postprocess import apply_if_postprocess
from .quality_selector import (
    QualitySelector,
    quality_selector_config_from_dict,
    score_candidates_with_quality,
    select_candidate_with_quality,
)
from .router import ROUTE_NAMES, HardRouteClassifier, router_config_from_dict
from .routing_policy import candidate_route_indices, export_candidate_indices, select_best_candidate
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
    "cross_overlap_like": "identity_viterbi",
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
    parser.add_argument("--export-candidates-topk", type=int, default=2, help="Store this many expert IF candidates in the output.")
    parser.add_argument("--export-candidate-policy", choices=["router_topk", "guarded_special"], default="guarded_special")
    parser.add_argument("--export-special-boost", type=float, default=0.12)
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
    confidence_rows = []
    candidate_if_rows = []
    candidate_route_rows = []
    candidate_route_prob_rows = []
    candidate_ridge_conf_rows = []
    candidate_quality_rows = []
    route_labels = []
    top_route_labels = []
    expert_used = []
    fallback_used = []
    freq_grid_ref = None
    for sample_idx in range(signal.shape[0]):
        top_route_name = route_names[int(route_idx[sample_idx].detach().cpu())]
        top_route_labels.append(top_route_name)
        fallback_indices, used_fallback = candidate_route_indices(
            route_probs[sample_idx],
            confidence_threshold=args.fallback_confidence,
            margin_threshold=args.fallback_margin,
            topk=args.fallback_topk,
        )
        if not args.fallback:
            fallback_indices, used_fallback = [int(route_idx[sample_idx].detach().cpu())], False
        export_topk = max(1, int(args.export_candidates_topk))
        export_indices = export_candidate_indices(
            route_probs[sample_idx],
            route_names,
            topk=export_topk,
            policy=args.export_candidate_policy,
            special_boost=args.export_special_boost,
        )
        candidate_indices = []
        for idx in fallback_indices + export_indices:
            if idx not in candidate_indices:
                candidate_indices.append(idx)

        outputs = {}
        candidates = []
        for candidate_idx in candidate_indices:
            candidate_name = route_names[candidate_idx]
            model, sim_cfg, stft_cfg, model_cfg = loaded_experts[candidate_name]
            feats, freq_grid = log_spectrogram(signal[sample_idx : sample_idx + 1], stft_cfg, sim_cfg.fs)
            model_out = model(feats)
            logits = model_out[0] if isinstance(model_out, tuple) else model_out
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

        selection_candidates = [item for item in candidates if item[0] in fallback_indices]
        if used_fallback and quality_selector is not None:
            _, route_name = select_candidate_with_quality(
                quality_selector,
                route_probs[sample_idx],
                selection_candidates,
                route_names,
                margin=args.quality_selector_margin,
                protect_top_routes=set(args.quality_protect_top_routes),
                protect_min_prob=args.quality_protect_min_prob,
            )
        else:
            _, route_name = select_best_candidate(route_probs[sample_idx], selection_candidates) if used_fallback else selection_candidates[0][:2]
        pred_if, probs, freq_grid = outputs[route_name]
        route_conf = route_confidence_features(route_probs[sample_idx])
        ridge_conf = ridge_confidence_features(probs)
        confidence_rows.append(
            {
                **route_conf,
                **ridge_conf,
                "initial_confidence": combined_initial_confidence(route_probs[sample_idx], probs),
            }
        )

        export_candidates = [item for item in candidates if item[0] in export_indices]
        export_candidates = sorted(export_candidates, key=lambda item: export_indices.index(item[0]))
        quality_scores = {}
        if quality_selector is not None:
            quality_scores = score_candidates_with_quality(quality_selector, route_probs[sample_idx], export_candidates, route_names)
        candidate_if_rows.append(torch.cat([outputs[name][0].detach().cpu() for _idx, name, _pred, _probs in export_candidates], dim=0))
        candidate_route_rows.append([name for _idx, name, _pred, _probs in export_candidates])
        candidate_route_prob_rows.append([float(route_probs[sample_idx, idx].detach().cpu()) for idx, _name, _pred, _probs in export_candidates])
        candidate_ridge_conf_rows.append([ridge_confidence_features(outputs[name][1])["ridge_confidence"] for _idx, name, _pred, _probs in export_candidates])
        candidate_quality_rows.append([quality_scores.get(name, float("nan")) for _idx, name, _pred, _probs in export_candidates])
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
        "initial_confidence": np.array([row["initial_confidence"] for row in confidence_rows], dtype=np.float32),
        "route_top_prob": np.array([row["route_top_prob"] for row in confidence_rows], dtype=np.float32),
        "route_margin": np.array([row["route_margin"] for row in confidence_rows], dtype=np.float32),
        "ridge_confidence": np.array([row["ridge_confidence"] for row in confidence_rows], dtype=np.float32),
        "ridge_entropy": np.array([row["ridge_entropy"] for row in confidence_rows], dtype=np.float32),
        "candidate_if_hz": torch.stack(candidate_if_rows, dim=0).numpy(),
        "candidate_route": np.array(candidate_route_rows, dtype=object),
        "candidate_route_prob": np.array(candidate_route_prob_rows, dtype=np.float32),
        "candidate_ridge_confidence": np.array(candidate_ridge_conf_rows, dtype=np.float32),
        "candidate_quality_score": np.array(candidate_quality_rows, dtype=np.float32),
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
    if ckpt.get("model_type") == "IFNetJumpAux":
        model = IFNetJumpAux(feature_channels(stft_cfg), sim_cfg.num_components, model_cfg).to(device)
    else:
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
