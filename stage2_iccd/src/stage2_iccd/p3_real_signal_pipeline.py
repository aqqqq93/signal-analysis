from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from ifnet_stage1.config import load_config

from .domain_adaptation import stft_domain_summary
from .infer_p15_signal import infer_signal


def run_p3_entry(
    npy_dir: str | Path,
    output_dir: str | Path,
    fs: float = 1024.0,
    device_name: str = "auto",
    max_files: int = 16,
    crossing_checkpoint: str = "stage2_iccd/runs/crossing_first_candidate_p25/latest.pt",
    reference_config: str | Path = "stage2_iccd/runs/active_count_simple_near_parallel/config.json",
    target_samples: int = 1024,
    scenario_hint: str | None = None,
    scenario_hints: dict[str, str] | None = None,
) -> dict[str, Any]:
    input_dir = Path(npy_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.npy"))[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}.")

    ref_cfg = load_config(reference_config)
    domain_json = out_dir / "domain_summary.json"
    domain = stft_domain_summary(
        input_dir,
        ref_cfg,
        domain_json,
        device_name=device_name,
        max_files=max_files,
    )

    signal_dir = out_dir / "signals"
    rows = []
    for path in files:
        hint = (scenario_hints or {}).get(path.name, (scenario_hints or {}).get(path.stem, scenario_hint))
        item_dir = signal_dir / path.stem
        result = infer_signal(
            input_npy=path,
            output_dir=item_dir,
            fs=fs,
            device_name=device_name,
            scenario_hint=hint,
            crossing_checkpoint=crossing_checkpoint,
            target_samples=target_samples,
        )
        rows.append(
            {
                "name": path.name,
                "scenario_hint": hint or "",
                "branch": result["branch"],
                "active_pred": result["active_pred"],
                "active_confidence": result["active_confidence"],
                "top2_weight_sum": sum(result["candidate_top2_weights"]),
                "plot_path": result["plot_path"],
                "if_path": result["if_path"],
                "active_if_path": result["active_if_path"],
                "plot_if_source": result["plot_if_source"],
                "resampled": result["resampled"],
                "original_samples": result["original_samples"],
                "model_samples": result["model_samples"],
            }
        )

    payload = {
        "stage": "P3 real-signal entry",
        "npy_dir": str(input_dir),
        "output_dir": str(out_dir),
        "fs": float(fs),
        "target_samples": int(target_samples),
        "crossing_checkpoint": str(crossing_checkpoint),
        "reference_config": str(reference_config),
        "domain_summary": domain,
        "rows": rows,
        "decision_note": p3_decision_note(domain),
    }
    summary_path = out_dir / "p3_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = build_p3_html_report(payload, out_dir / "p3_report.html")
    payload["report_path"] = str(report_path)
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def p3_decision_note(domain: dict[str, Any]) -> str:
    l2 = float(domain.get("feature_l2", 0.0))
    l1 = float(domain.get("feature_l1", 0.0))
    if l2 <= 0.75 and l1 <= 0.16:
        return "Domain gap is modest for the current simulator baseline; start with Stage2-only refinement or candidate-selection tuning."
    if l2 <= 1.25 and l1 <= 0.28:
        return "Domain gap is noticeable; inspect IF overlays first, then prefer Stage2-only adaptation before touching Stage1."
    return "Domain gap is large; expand or reweight simulation data before any Stage1 unfreeze."


def build_p3_html_report(payload: dict[str, Any], output_html: str | Path) -> Path:
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    domain = payload["domain_summary"]
    rows = payload["rows"]
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>Stage2 P3 Real Signal Entry</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:28px;background:#f7f9fc;color:#1f2933}",
        "h1,h2{color:#19324d} table{border-collapse:collapse;width:100%;background:#fff;margin:12px 0}",
        "th,td{border:1px solid #c7d5e7;padding:7px 9px;font-size:13px;text-align:left}",
        "th{background:#e7f0fb}.note{background:#fff8e6;border-left:4px solid #d99a23;padding:10px 12px;margin:12px 0}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(390px,1fr));gap:16px}",
        "figure{background:#fff;border:1px solid #d7e1ee;padding:10px;margin:0}img{max-width:100%;height:auto}",
        "</style></head><body>",
        "<h1>Stage2 P3 Real Signal Entry</h1>",
        f"<p>Input: <code>{html.escape(payload['npy_dir'])}</code></p>",
        f"<p>Decision: {html.escape(payload['decision_note'])}</p>",
        "<h2>Domain Gap</h2>",
        _table(
            [
                {
                    "num_files": domain.get("num_files"),
                    "feature_l1": domain.get("feature_l1"),
                    "feature_l2": domain.get("feature_l2"),
                }
            ]
        ),
        "<div class='note'>P3 entry keeps Stage1 frozen. Use this report to decide whether Stage2-only tuning is enough.</div>",
        "<h2>Signals</h2>",
        _table(rows),
        "<h2>IF Overlays</h2><div class='grid'>",
    ]
    for row in rows:
        plot_path = Path(row["plot_path"])
        rel = _rel(plot_path, output.parent)
        caption = f"{row['name']} | branch={row['branch']} | active={row['active_pred']} | conf={row['active_confidence']:.3f}"
        parts.append(f"<figure><img src='{html.escape(rel)}'><figcaption>{html.escape(caption)}</figcaption></figure>")
    parts.append("</div></body></html>")
    output.write_text("\n".join(parts), encoding="utf-8")
    return output


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    keys = list(rows[0].keys())
    head = "".join(f"<th>{html.escape(str(key))}</th>" for key in keys)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(_format(row.get(key, '')))}</td>" for key in keys)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy-dir", required=True)
    parser.add_argument("--output-dir", default="stage2_iccd/runs/p3_real_signal_entry")
    parser.add_argument("--fs", type=float, default=1024.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-files", type=int, default=16)
    parser.add_argument("--target-samples", type=int, default=1024)
    parser.add_argument("--scenario-hint", default=None)
    parser.add_argument("--scenario-hints-json", default=None)
    parser.add_argument("--crossing-checkpoint", default="stage2_iccd/runs/crossing_first_candidate_p25/latest.pt")
    parser.add_argument("--reference-config", default="stage2_iccd/runs/active_count_simple_near_parallel/config.json")
    args = parser.parse_args()
    hints = parse_hints(args.scenario_hints_json) if args.scenario_hints_json else None
    result = run_p3_entry(
        npy_dir=args.npy_dir,
        output_dir=args.output_dir,
        fs=args.fs,
        device_name=args.device,
        max_files=args.max_files,
        crossing_checkpoint=args.crossing_checkpoint,
        reference_config=args.reference_config,
        target_samples=args.target_samples,
        scenario_hint=args.scenario_hint,
        scenario_hints=hints,
    )
    print(json.dumps(result, indent=2))


def parse_hints(value: str) -> dict[str, str]:
    try:
        parsed = json.loads(value)
        return {str(key): str(item) for key, item in parsed.items()}
    except json.JSONDecodeError:
        items = value.strip().strip("{}")
        result: dict[str, str] = {}
        for item in items.split(","):
            if not item.strip():
                continue
            key, raw = item.split(":", 1)
            result[key.strip().strip("'\"")] = raw.strip().strip("'\"")
        return result


if __name__ == "__main__":
    main()
