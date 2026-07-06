from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from .confidence import combined_initial_confidence
from .jump_aux import jump_location_from_if, jump_location_from_logits
from .model import soft_argmax_if
from .postprocess import apply_if_postprocess
from .predict_routed import DEFAULT_EXPERTS, DEFAULT_POSTPROCESS, load_expert, load_quality_selector
from .quality_selector import select_candidate_with_quality
from .router import DEFAULT_SCENARIO_TO_ROUTE, ROUTE_NAMES, HardRouteClassifier, router_config_from_dict, scenario_to_route_labels
from .routing_policy import candidate_route_indices, export_candidate_indices, select_best_candidate
from .simulation import ChirpSimulator, SCENARIOS, sim_config_from_dict
from .tf import feature_channels, log_spectrogram, sample_if_to_frames, stft_config_from_dict


DEFAULT_THRESHOLDS = {
    "avg_mae_hz": 5.5,
    "candidate_oracle_coverage_10hz": 0.88,
    "high_conf_mae_hz": 5.0,
    "high_conf_min_coverage": 0.55,
    "crossing_identity_excess_hz": 8.0,
    "crossing_fixed_mae_hz": 12.0,
    "local_jump_mae_hz": 10.5,
    "local_jump_p95_hz": 40.0,
    "local_jump_event_mae_ms": 80.0,
    "sinusoidal_mae_hz": 8.5,
}


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


def best_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((best_align(pred, target) - target).abs().mean().detach().cpu())


@torch.no_grad()
def evaluate_readiness(
    router_checkpoint: str | Path,
    output_dir: str | Path,
    scenarios: list[str],
    batch_size: int,
    batches: int,
    seed: int,
    device_name: str,
    expert_paths: dict[str, str],
    fallback_confidence: float,
    fallback_margin: float,
    fallback_topk: int,
    quality_selector_checkpoint: str | Path | None,
    quality_selector_margin: float,
    quality_protect_top_routes: set[str],
    confidence_threshold: float,
    jump_aux_checkpoint: str | Path | None = None,
    candidate_policy: str = "guarded_special",
    candidate_special_boost: float = 0.12,
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

    experts = {route_name: load_expert(expert_paths[route_name], device) for route_name in route_names if route_name in expert_paths}
    jump_aux = load_expert(jump_aux_checkpoint, device) if jump_aux_checkpoint else None
    quality_selector = load_quality_selector(quality_selector_checkpoint, device) if quality_selector_checkpoint else None
    simulator = ChirpSimulator(router_sim_cfg, seed=seed)

    results: dict[str, dict] = {}
    all_mae = []
    all_conf = []
    all_high_conf_mae = []
    all_oracle_under_10 = []

    for scenario in scenarios:
        selected_maes = []
        fixed_maes = []
        oracle_maes = []
        confidences = []
        route_correct = 0
        top1_route_correct = 0
        total = 0
        fallback_count = 0
        local_jump_ms = []

        for _ in range(batches):
            batch = simulator.generate_batch(batch_size, device=device, scenarios=[scenario])
            router_feats, _ = log_spectrogram(batch["signal"], router_stft_cfg, router_sim_cfg.fs)
            route_probs = torch.softmax(router(router_feats), dim=1)
            top1_idx = route_probs.argmax(dim=1)
            target_route_idx = scenario_to_route_labels(
                batch["scenario"],
                route_names=route_names,
                scenario_to_route=scenario_to_route,
                device=device,
            )

            for sample_idx in range(batch_size):
                total += 1
                top1_route_correct += int(int(top1_idx[sample_idx].detach().cpu()) == int(target_route_idx[sample_idx].detach().cpu()))
                selection_indices, used_fallback = candidate_route_indices(
                    route_probs[sample_idx],
                    confidence_threshold=fallback_confidence,
                    margin_threshold=fallback_margin,
                    topk=fallback_topk,
                )
                export_indices = export_candidate_indices(
                    route_probs[sample_idx],
                    route_names,
                    topk=fallback_topk,
                    policy=candidate_policy,
                    special_boost=candidate_special_boost,
                )
                candidate_indices = []
                for idx in selection_indices + export_indices:
                    if idx not in candidate_indices:
                        candidate_indices.append(idx)
                fallback_count += int(used_fallback)

                candidates = []
                outputs = {}
                for candidate_idx in candidate_indices:
                    route_name = route_names[candidate_idx]
                    model, sim_cfg, stft_cfg, model_cfg = experts[route_name]
                    signal = batch["signal"][sample_idx : sample_idx + 1]
                    feats, freq_grid = log_spectrogram(signal, stft_cfg, sim_cfg.fs)
                    model_out = model(feats)
                    if isinstance(model_out, tuple):
                        logits, jump_logits = model_out
                    else:
                        logits, jump_logits = model_out, None
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
                    candidates.append((candidate_idx, route_name, pred_if, probs))
                    outputs[route_name] = (pred_if, target_if, probs, jump_logits)

                selection_candidates = [item for item in candidates if item[0] in selection_indices]
                if used_fallback and quality_selector is not None:
                    selected_idx, selected_name = select_candidate_with_quality(
                        quality_selector,
                        route_probs[sample_idx],
                        selection_candidates,
                        route_names,
                        margin=quality_selector_margin,
                        protect_top_routes=quality_protect_top_routes,
                    )
                else:
                    selected_idx, selected_name = select_best_candidate(route_probs[sample_idx], selection_candidates) if used_fallback else selection_candidates[0][:2]

                route_correct += int(selected_idx == int(target_route_idx[sample_idx].detach().cpu()))
                pred_if, target_if, ridge_probs, jump_logits = outputs[selected_name]
                selected_mae = best_mae(pred_if[0].detach().cpu(), target_if[0].detach().cpu())
                fixed_mae = float((pred_if[0].detach().cpu() - target_if[0].detach().cpu()).abs().mean())
                oracle_mae = min(
                    best_mae(outputs[name][0][0].detach().cpu(), outputs[name][1][0].detach().cpu())
                    for _idx, name, _pred, _probs in candidates
                    if _idx in export_indices
                )
                confidence = combined_initial_confidence(route_probs[sample_idx], ridge_probs)

                selected_maes.append(selected_mae)
                fixed_maes.append(fixed_mae)
                oracle_maes.append(oracle_mae)
                confidences.append(confidence)
                all_mae.append(selected_mae)
                all_conf.append(confidence)
                all_oracle_under_10.append(oracle_mae <= 10.0)
                if confidence >= confidence_threshold:
                    all_high_conf_mae.append(selected_mae)

                if scenario == "local_jump":
                    if jump_aux is not None:
                        aux_model, aux_sim_cfg, aux_stft_cfg, _aux_model_cfg = jump_aux
                        aux_feats, _aux_freq_grid = log_spectrogram(batch["signal"][sample_idx : sample_idx + 1], aux_stft_cfg, aux_sim_cfg.fs)
                        aux_out = aux_model(aux_feats)
                        aux_jump_logits = aux_out[1] if isinstance(aux_out, tuple) else None
                        aux_target_if = sample_if_to_frames(
                            batch["if_hz"][sample_idx : sample_idx + 1],
                            aux_jump_logits.shape[-1] if aux_jump_logits is not None else target_if.shape[-1],
                            aux_stft_cfg.hop_length,
                        )
                        pred_jump = (
                            jump_location_from_logits(aux_jump_logits).detach().cpu()
                            if aux_jump_logits is not None
                            else jump_location_from_if(aux_target_if).detach().cpu()
                        )
                        target_jump = jump_location_from_if(aux_target_if).detach().cpu()
                        frame_err = (pred_jump - target_jump).abs().float().mean().item()
                        local_jump_ms.append(frame_err * aux_stft_cfg.hop_length / aux_sim_cfg.fs * 1000.0)
                    else:
                        pred_jump = (
                            jump_location_from_logits(jump_logits).detach().cpu()
                            if jump_logits is not None
                            else jump_location_from_if(pred_if).detach().cpu()
                        )
                        target_jump = jump_location_from_if(target_if).detach().cpu()
                        frame_err = (pred_jump - target_jump).abs().float().mean().item()
                        local_jump_ms.append(frame_err * stft_cfg.hop_length / sim_cfg.fs * 1000.0)

        scenario_result = {
            "if_mae_hz": float(np.mean(selected_maes)),
            "if_mae_hz_median": float(np.median(selected_maes)),
            "if_mae_hz_p95": float(np.percentile(selected_maes, 95)),
            "fixed_identity_mae_hz": float(np.mean(fixed_maes)),
            "identity_excess_hz": float(np.mean(fixed_maes) - np.mean(selected_maes)),
            "top2_oracle_mae_hz": float(np.mean(oracle_maes)),
            "top2_oracle_coverage_10hz": float(np.mean(np.array(oracle_maes) <= 10.0)),
            "initial_confidence_mean": float(np.mean(confidences)),
            "route_accuracy": route_correct / max(1, total),
            "top1_route_accuracy": top1_route_correct / max(1, total),
            "fallback_fraction": fallback_count / max(1, total),
            "num_examples": total,
        }
        if scenario == "local_jump":
            scenario_result["jump_event_mae_ms"] = float(np.mean(local_jump_ms))
            scenario_result["jump_event_p95_ms"] = float(np.percentile(local_jump_ms, 95))
        results[scenario] = scenario_result

    aggregate = {
        "avg_mae_hz": float(np.mean(all_mae)),
        "avg_initial_confidence": float(np.mean(all_conf)),
        "high_confidence_threshold": confidence_threshold,
        "high_confidence_coverage": float(np.mean(np.array(all_conf) >= confidence_threshold)),
        "high_confidence_mae_hz": float(np.mean(all_high_conf_mae)) if all_high_conf_mae else float("nan"),
        "candidate_oracle_coverage_10hz": float(np.mean(all_oracle_under_10)),
    }
    gates = build_gates(results, aggregate, DEFAULT_THRESHOLDS)
    payload = {
        "thresholds": DEFAULT_THRESHOLDS,
        "aggregate": aggregate,
        "scenarios": results,
        "gates": gates,
        "ready_for_stage2": all(item["pass"] for item in gates.values()),
        "recommendation": recommendation_from_gates(gates),
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "readiness_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def build_gates(results: dict[str, dict], aggregate: dict[str, float], thresholds: dict[str, float]) -> dict[str, dict]:
    local = results.get("local_jump", {})
    crossing = results.get("crossing", {})
    sinusoidal = results.get("sinusoidal_fm", {})
    gates = {
        "overall_mae": {
            "value": aggregate["avg_mae_hz"],
            "threshold": thresholds["avg_mae_hz"],
            "pass": aggregate["avg_mae_hz"] <= thresholds["avg_mae_hz"],
        },
        "top2_candidate_coverage": {
            "value": aggregate["candidate_oracle_coverage_10hz"],
            "threshold": thresholds["candidate_oracle_coverage_10hz"],
            "pass": aggregate["candidate_oracle_coverage_10hz"] >= thresholds["candidate_oracle_coverage_10hz"],
        },
        "high_confidence_quality": {
            "value": aggregate["high_confidence_mae_hz"],
            "threshold": thresholds["high_conf_mae_hz"],
            "pass": aggregate["high_confidence_coverage"] >= thresholds["high_conf_min_coverage"]
            and aggregate["high_confidence_mae_hz"] <= thresholds["high_conf_mae_hz"],
        },
        "crossing_identity": {
            "value": crossing.get("identity_excess_hz", float("nan")),
            "threshold": thresholds["crossing_identity_excess_hz"],
            "pass": crossing.get("identity_excess_hz", float("inf")) <= thresholds["crossing_identity_excess_hz"]
            and crossing.get("fixed_identity_mae_hz", float("inf")) <= thresholds["crossing_fixed_mae_hz"],
        },
        "local_jump_if": {
            "value": local.get("if_mae_hz", float("nan")),
            "threshold": thresholds["local_jump_mae_hz"],
            "pass": local.get("if_mae_hz", float("inf")) <= thresholds["local_jump_mae_hz"]
            and local.get("if_mae_hz_p95", float("inf")) <= thresholds["local_jump_p95_hz"],
        },
        "local_jump_event": {
            "value": local.get("jump_event_mae_ms", float("nan")),
            "threshold": thresholds["local_jump_event_mae_ms"],
            "pass": local.get("jump_event_mae_ms", float("inf")) <= thresholds["local_jump_event_mae_ms"],
        },
        "sinusoidal_quality": {
            "value": sinusoidal.get("if_mae_hz", float("nan")),
            "threshold": thresholds["sinusoidal_mae_hz"],
            "pass": sinusoidal.get("if_mae_hz", float("inf")) <= thresholds["sinusoidal_mae_hz"],
        },
    }
    return gates


def recommendation_from_gates(gates: dict[str, dict]) -> str:
    failed = [name for name, row in gates.items() if not row["pass"]]
    if not failed:
        return "Stage 1 is ready to provide initial IF estimates to Stage 2 ICCD."
    return "Do not freeze Stage 1 yet. Focus next on: " + ", ".join(failed) + "."


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
    parser.add_argument("--fallback-confidence", type=float, default=0.78)
    parser.add_argument("--fallback-margin", type=float, default=0.18)
    parser.add_argument("--fallback-topk", type=int, default=2)
    parser.add_argument("--quality-selector-checkpoint", default=None)
    parser.add_argument("--quality-selector-margin", type=float, default=0.10)
    parser.add_argument("--quality-protect-top-routes", nargs="*", default=["cross_overlap_like"])
    parser.add_argument("--confidence-threshold", type=float, default=0.62)
    parser.add_argument("--jump-aux-checkpoint", default=None, help="Optional IFNetJumpAux checkpoint used only for local-jump event timing.")
    parser.add_argument("--candidate-policy", choices=["router_topk", "guarded_special"], default="guarded_special")
    parser.add_argument("--candidate-special-boost", type=float, default=0.12)
    args = parser.parse_args()

    expert_paths = {
        "poly_like": args.poly_checkpoint,
        "sinusoidal_like": args.sinusoidal_checkpoint,
        "cross_overlap_like": args.cross_checkpoint,
        "jump_like": args.jump_checkpoint,
    }
    result = evaluate_readiness(
        router_checkpoint=args.router_checkpoint,
        output_dir=args.output_dir,
        scenarios=args.scenarios,
        batch_size=args.batch_size,
        batches=args.batches,
        seed=args.seed,
        device_name=args.device,
        expert_paths=expert_paths,
        fallback_confidence=args.fallback_confidence,
        fallback_margin=args.fallback_margin,
        fallback_topk=args.fallback_topk,
        quality_selector_checkpoint=args.quality_selector_checkpoint,
        quality_selector_margin=args.quality_selector_margin,
        quality_protect_top_routes=set(args.quality_protect_top_routes),
        confidence_threshold=args.confidence_threshold,
        jump_aux_checkpoint=args.jump_aux_checkpoint,
        candidate_policy=args.candidate_policy,
        candidate_special_boost=args.candidate_special_boost,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
