from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .router import DEFAULT_SCENARIO_TO_ROUTE, ROUTE_NAMES, HardRouteClassifier, router_config_from_dict, scenario_to_route_labels
from .simulation import ChirpSimulator, SCENARIOS, sim_config_from_dict
from .tf import feature_channels, log_spectrogram, stft_config_from_dict


@torch.no_grad()
def evaluate_router(
    checkpoint: str | Path,
    output_dir: str | Path,
    scenarios: list[str],
    batch_size: int,
    batches: int,
    seed: int,
    device_name: str,
) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    route_names = tuple(ckpt.get("route_names", cfg.get("route_names", ROUTE_NAMES)))
    scenario_to_route = dict(ckpt.get("scenario_to_route", cfg.get("scenario_to_route", DEFAULT_SCENARIO_TO_ROUTE)))
    sim_cfg = sim_config_from_dict(cfg["data"])
    stft_cfg = stft_config_from_dict(cfg["stft"])
    router_cfg = router_config_from_dict(cfg["router"])
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )

    simulator = ChirpSimulator(sim_cfg, seed=seed)
    model = HardRouteClassifier(feature_channels(stft_cfg), len(route_names), router_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    confusion = torch.zeros((len(route_names), len(route_names)), dtype=torch.long)
    scenario_results: dict[str, dict] = {}

    for scenario in scenarios:
        counts = torch.zeros(len(route_names), dtype=torch.long)
        total = 0
        correct = 0
        confidence_sum = 0.0
        for _ in range(batches):
            batch = simulator.generate_batch(batch_size, device=device, scenarios=[scenario])
            feats, _ = log_spectrogram(batch["signal"], stft_cfg, sim_cfg.fs)
            labels = scenario_to_route_labels(
                batch["scenario"],
                route_names=route_names,
                scenario_to_route=scenario_to_route,
                device=device,
            )
            logits = model(feats)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            for label, pred_label in zip(labels.cpu(), pred.cpu()):
                confusion[label, pred_label] += 1
                counts[pred_label] += 1
            total += int(labels.numel())
            correct += int((pred == labels).sum().detach().cpu())
            confidence_sum += float(probs.max(dim=1).values.sum().detach().cpu())

        scenario_results[scenario] = {
            "target_route": scenario_to_route[scenario],
            "accuracy": correct / max(1, total),
            "mean_confidence": confidence_sum / max(1, total),
            "predicted_route_counts": {route_names[idx]: int(counts[idx]) for idx in range(len(route_names))},
            "num_examples": total,
        }

    overall_correct = int(confusion.diag().sum())
    overall_total = int(confusion.sum())
    result = {
        "route_names": list(route_names),
        "overall_accuracy": overall_correct / max(1, overall_total),
        "confusion_matrix_rows_true_cols_pred": confusion.tolist(),
        "scenarios": scenario_results,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "router_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=54321)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    unknown = sorted(set(args.scenarios) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown scenarios: {unknown}. Valid: {SCENARIOS}")

    result = evaluate_router(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        scenarios=args.scenarios,
        batch_size=args.batch_size,
        batches=args.batches,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
