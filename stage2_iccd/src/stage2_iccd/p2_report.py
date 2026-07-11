from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def build_html_report(eval_dir: str | Path, output_html: str | Path | None = None, title: str = "Stage2 P2 Report") -> Path:
    eval_path = Path(eval_dir)
    metrics_path = _find_metrics(eval_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    output = Path(output_html) if output_html else eval_path / "p2_report.html"
    images = sorted((eval_path / "plots").glob("*.png"))
    overview = eval_path / "overview.png"
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:28px;color:#1d2733;background:#f7f9fc}",
        "h1,h2{color:#18324f} table{border-collapse:collapse;width:100%;background:white;margin:12px 0}",
        "th,td{border:1px solid #c9d6e6;padding:7px 9px;text-align:left;font-size:13px}",
        "th{background:#e8f1fb} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}",
        "figure{background:white;border:1px solid #d7e1ee;padding:10px;margin:0} img{max-width:100%;height:auto}",
        ".note{background:#fff8e6;border-left:4px solid #e4a72d;padding:10px 12px;margin:14px 0}",
        "</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p>Source metrics: <code>{html.escape(str(metrics_path))}</code></p>",
    ]
    aggregate = next((row for row in rows if row.get("scenario") == "aggregate"), None)
    if aggregate:
        parts.append("<h2>Aggregate</h2>")
        parts.append(_table([aggregate]))
        top2 = aggregate.get("top2_candidate_coverage")
        crossing_rows = [row for row in rows if row.get("scenario") == "crossing"]
        if top2 is not None and float(top2) < 0.88:
            parts.append(
                "<div class='note'>Top-2 candidate coverage is still below 88%; keep candidate-fusion work active in P2.</div>"
            )
        if crossing_rows:
            worst = max(float(row.get("if_mae_hz", 0.0)) for row in crossing_rows)
            if worst > 15.0:
                parts.append(
                    "<div class='note'>Crossing remains the dominant P2 risk; use identity consistency and crossing-specific fusion before domain adaptation.</div>"
                )
    if rows:
        parts.append("<h2>Per Scenario</h2>")
        parts.append(_table(rows))
    if overview.exists():
        parts.append("<h2>Overview</h2>")
        parts.append(f"<figure><img src='{html.escape(_rel(overview, output.parent))}'></figure>")
    if images:
        parts.append("<h2>Plots</h2><div class='grid'>")
        for image in images:
            parts.append(
                f"<figure><img src='{html.escape(_rel(image, output.parent))}'><figcaption>{html.escape(image.name)}</figcaption></figure>"
            )
        parts.append("</div>")
    parts.append("</body></html>")
    output.write_text("\n".join(parts), encoding="utf-8")
    return output


def _find_metrics(eval_path: Path) -> Path:
    for name in ("p15_pipeline_metrics.json", "routed_stage2_metrics.json", "scenario_metrics.json"):
        candidate = eval_path / name
        if candidate.exists():
            return candidate
    matches = sorted(eval_path.glob("*metrics*.json"))
    if not matches:
        raise FileNotFoundError(f"No metrics JSON found under {eval_path}.")
    return matches[0]


def _table(rows: list[dict[str, Any]]) -> str:
    keys = sorted({key for row in rows for key in row})
    head = "".join(f"<th>{html.escape(key)}</th>" for key in keys)
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
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-html", default=None)
    parser.add_argument("--title", default="Stage2 P2 Report")
    args = parser.parse_args()
    output = build_html_report(args.eval_dir, args.output_html, title=args.title)
    print(output)


if __name__ == "__main__":
    main()
