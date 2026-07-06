from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .model import IFNetUNet, model_config_from_dict, soft_argmax_if
from .postprocess import apply_if_postprocess
from .predict_routed import DEFAULT_EXPERTS, DEFAULT_POSTPROCESS
from .quality_selector import QualitySelector, quality_selector_config_from_dict, select_candidate_with_quality
from .router import DEFAULT_SCENARIO_TO_ROUTE, ROUTE_NAMES, HardRouteClassifier, router_config_from_dict, scenario_to_route_labels
from .routing_policy import candidate_route_indices, select_best_candidate
from .simulation import ChirpSimulator, SCENARIOS, sim_config_from_dict
from .tf import feature_channels, log_spectrogram, sample_if_to_frames, stft_config_from_dict


def best_align(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    q = pred.shape[0]
    best_cost = None
    best_pred = pred
    for perm in itertools.permutations(range(q)):
        candidate = pred[list(perm)]
        cost = (candidate - target).abs().mean()
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_pred = candidate
    return best_pred


@torch.no_grad()
def evaluate_routed(
    router_checkpoint: str | Path,
    output_dir: str | Path,
    scenarios: list[str],
    batch_size: int,
    batches: int,
    seed: int,
    device_name: str,
    expert_paths: dict[str, str],
    fallback: bool = False,
    fallback_confidence: float = 0.78,
    fallback_margin: float = 0.18,
    fallback_topk: int = 2,
    quality_selector_checkpoint: str | Path | None = None,
    quality_selector_margin: float = 0.10,
    quality_protect_top_routes: set[str] | None = None,
    quality_protect_min_prob: float = 0.0,
) -> dict:
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )

    router_ckpt = torch.load(router_checkpoint, map_location="cpu")
    router_cfg_all = router_ckpt["config"]
    route_names = tuple(router_ckpt.get("route_names", router_cfg_all.get("route_names", ROUTE_NAMES)))
    scenario_to_route = dict(router_ckpt.get("scenario_to_route", router_cfg_all.get("scenario_to_route", DEFAULT_SCENARIO_TO_ROUTE)))
    router_sim_cfg = sim_config_from_dict(router_cfg_all["data"])
    router_stft_cfg = stft_config_from_dict(router_cfg_all["stft"])
    router_cfg = router_config_from_dict(router_cfg_all["router"])
    router = HardRouteClassifier(feature_channels(router_stft_cfg), len(route_names), router_cfg).to(device)
    router.load_state_dict(router_ckpt["model"])
    router.eval()

    experts = {
        route_name: load_expert(expert_paths[route_name], device)
        for route_name in route_names
        if route_name in expert_paths
    }
    quality_selector = load_quality_selector(quality_selector_checkpoint, device) if quality_selector_checkpoint else None
    quality_protect_top_routes = quality_protect_top_routes or set()

    simulator = ChirpSimulator(router_sim_cfg, seed=seed)
    results: dict[str, dict] = {}
    for scenario in scenarios:
        sample_maes = []
        route_counts = {name: 0 for name in route_names}
        top1_route_counts = {name: 0 for name in route_names}
        route_correct = 0
        top1_route_correct = 0
        route_total = 0
        fallback_count = 0
        for _ in range(batches):
            batch = simulator.generate_batch(batch_size, device=device, scenarios=[scenario])
            router_feats, _ = log_spectrogram(batch["signal"], router_stft_cfg, router_sim_cfg.fs)
            route_logits = router(router_feats)
            route_probs = torch.softmax(route_logits, dim=1)
            pred_route_idx = route_logits.argmax(dim=1)
            target_route_idx = scenario_to_route_labels(
                batch["scenario"],
                route_names=route_names,
                scenario_to_route=scenario_to_route,
                device=device,
            )
            route_total += int(pred_route_idx.numel())

            for sample_idx in range(batch_size):
                top1_idx = int(pred_route_idx[sample_idx].detach().cpu())
                top1_route_counts[route_names[top1_idx]] += 1
                top1_route_correct += int(top1_idx == int(target_route_idx[sample_idx].detach().cpu()))
                if fallback:
                    candidate_indices, used_fallback = candidate_route_indices(
                        route_probs[sample_idx],
                        confidence_threshold=fallback_confidence,
                        margin_threshold=fallback_margin,
                        topk=fallback_topk,
                    )
                else:
                    candidate_indices, used_fallback = [top1_idx], False

                outputs = {}
                candidates = []
                for candidate_idx in candidate_indices:
                    route_name = route_names[candidate_idx]
                    model, sim_cfg, stft_cfg, model_cfg = experts[route_name]
                    signal = batch["signal"][sample_idx : sample_idx + 1]
                    feats, freq_grid = log_spectrogram(signal, stft_cfg, sim_cfg.fs)
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
                    target_if = sample_if_to_frames(
                        batch["if_hz"][sample_idx : sample_idx + 1],
                        logits.shape[-1],
                        stft_cfg.hop_length,
                    )
                    outputs[route_name] = (pred_if, target_if, probs)
                    candidates.append((candidate_idx, route_name, pred_if, probs))

                if used_fallback and quality_selector is not None:
                    selected_idx, selected_name = select_candidate_with_quality(
                        quality_selector,
                        route_probs[sample_idx],
                        candidates,
                        route_names,
                        margin=quality_selector_margin,
                        protect_top_routes=quality_protect_top_routes,
                        protect_min_prob=quality_protect_min_prob,
                    )
                else:
                    selected_idx, selected_name = select_best_candidate(route_probs[sample_idx], candidates) if used_fallback else candidates[0][:2]
                fallback_count += int(used_fallback)
                route_counts[selected_name] += 1
                route_correct += int(selected_idx == int(target_route_idx[sample_idx].detach().cpu()))
                pred_if, target_if, _ = outputs[selected_name]
                aligned = best_align(pred_if[0].detach().cpu(), target_if[0].detach().cpu())
                sample_maes.append((aligned - target_if[0].detach().cpu()).abs().mean().item())

        results[scenario] = {
            "if_mae_hz": float(np.mean(sample_maes)) if sample_maes else float("nan"),
            "if_mae_hz_median": float(np.median(sample_maes)) if sample_maes else float("nan"),
            "if_mae_hz_p90": float(np.percentile(sample_maes, 90)) if sample_maes else float("nan"),
            "if_mae_hz_p95": float(np.percentile(sample_maes, 95)) if sample_maes else float("nan"),
            "route_accuracy": route_correct / max(1, route_total),
            "top1_route_accuracy": top1_route_correct / max(1, route_total),
            "fallback_fraction": fallback_count / max(1, route_total),
            "predicted_route_counts": route_counts,
            "top1_route_counts": top1_route_counts,
            "num_examples": len(sample_maes),
        }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "routed_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=67890)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--poly-checkpoint", default=DEFAULT_EXPERTS["poly_like"])
    parser.add_argument("--sinusoidal-checkpoint", default=DEFAULT_EXPERTS["sinusoidal_like"])
    parser.add_argument("--cross-checkpoint", default=DEFAULT_EXPERTS["cross_overlap_like"])
    parser.add_argument("--jump-checkpoint", default=DEFAULT_EXPERTS["jump_like"])
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--fallback-confidence", type=float, default=0.78)
    parser.add_argument("--fallback-margin", type=float, default=0.18)
    parser.add_argument("--fallback-topk", type=int, default=2)
    parser.add_argument("--quality-selector-checkpoint", default=None)
    parser.add_argument("--quality-selector-margin", type=float, default=0.10)
    parser.add_argument("--quality-protect-top-routes", nargs="*", default=[])
    parser.add_argument("--quality-protect-min-prob", type=float, default=0.0)
    args = parser.parse_args()

    expert_paths = {
        "poly_like": args.poly_checkpoint,
        "sinusoidal_like": args.sinusoidal_checkpoint,
        "cross_overlap_like": args.cross_checkpoint,
        "jump_like": args.jump_checkpoint,
    }
    result = evaluate_routed(
        router_checkpoint=args.router_checkpoint,
        output_dir=args.output_dir,
        scenarios=args.scenarios,
        batch_size=args.batch_size,
        batches=args.batches,
        seed=args.seed,
        device_name=args.device,
        expert_paths=expert_paths,
        fallback=args.fallback,
        fallback_confidence=args.fallback_confidence,
        fallback_margin=args.fallback_margin,
        fallback_topk=args.fallback_topk,
        quality_selector_checkpoint=args.quality_selector_checkpoint,
        quality_selector_margin=args.quality_selector_margin,
        quality_protect_top_routes=set(args.quality_protect_top_routes),
        quality_protect_min_prob=args.quality_protect_min_prob,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
