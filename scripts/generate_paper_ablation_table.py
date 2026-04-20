"""Generate a compact paper-ready ablation table from benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

TABLE_COLUMNS = [
    "label",
    "detector_backend",
    "skip_clip",
    "total_runs",
    "mean_raw_num_detections",
    "mean_num_ranked_candidates",
    "fraction_top1_changed_by_rerank",
    "fraction_with_3d_target",
    "pick_success_rate",
    "mean_runtime_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paper-ready ablation table from benchmark_summary.json files.")
    parser.add_argument(
        "--benchmark",
        action="append",
        required=True,
        help="Benchmark entry as LABEL=DIR or just DIR. Repeat for each row.",
    )
    parser.add_argument("--output-md", type=Path, default=Path("outputs") / "paper_ablation_table.md")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs") / "paper_ablation_table.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_table_rows(args.benchmark)
    write_rows_csv(rows, args.output_csv)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown_table(rows), encoding="utf-8")
    print(f"Wrote paper ablation markdown: {args.output_md}")
    print(f"Wrote paper ablation CSV:      {args.output_csv}")


def build_table_rows(benchmark_specs: list[str]) -> list[dict[str, Any]]:
    """Build one ablation table row per benchmark directory."""

    return [row_from_benchmark(label, benchmark_dir) for label, benchmark_dir in map(parse_benchmark_spec, benchmark_specs)]


def parse_benchmark_spec(spec: str) -> tuple[str, Path]:
    """Parse ``LABEL=DIR`` or ``DIR`` into a display label and path."""

    if "=" in spec:
        label, path_text = spec.split("=", 1)
        label = label.strip()
        path_text = path_text.strip()
        if not label or not path_text:
            raise ValueError(f"Invalid benchmark spec {spec!r}; expected LABEL=DIR.")
        return label, Path(path_text)
    path = Path(spec)
    return path.name or str(path), path


def row_from_benchmark(label: str, benchmark_dir: Path) -> dict[str, Any]:
    """Load one benchmark summary into a flat table row."""

    summary = load_benchmark_summary(benchmark_dir)
    metrics = summary.get("aggregate_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    row = {
        "label": label,
        "detector_backend": _value(summary.get("detector_backend")),
        "skip_clip": _value(summary.get("skip_clip")),
        "total_runs": _as_int(summary.get("total_runs", metrics.get("total_runs"))),
        "mean_raw_num_detections": _as_float(metrics.get("mean_raw_num_detections", metrics.get("mean_num_detections"))),
        "mean_num_ranked_candidates": _as_float(metrics.get("mean_num_ranked_candidates")),
        "fraction_top1_changed_by_rerank": _as_float(metrics.get("fraction_top1_changed_by_rerank")),
        "fraction_with_3d_target": _as_float(metrics.get("fraction_with_3d_target")),
        "pick_success_rate": _as_float(metrics.get("pick_success_rate")),
        "mean_runtime_seconds": _as_float(metrics.get("mean_runtime_seconds")),
    }
    return row


def load_benchmark_summary(benchmark_dir: str | Path) -> dict[str, Any]:
    """Load ``benchmark_summary.json`` from a benchmark directory."""

    path = Path(benchmark_dir) / "benchmark_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Required benchmark summary not found: {path}")
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the ablation table as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render the ablation table as Markdown."""

    lines = [
        "# Paper Ablation Table",
        "",
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(TABLE_COLUMNS) - 1)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column)) for column in TABLE_COLUMNS) + " |")
    lines.append("")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return _format_float(value)
    return _escape_table_cell(_value(value))


def _format_float(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.4f}"


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _value(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    main()
