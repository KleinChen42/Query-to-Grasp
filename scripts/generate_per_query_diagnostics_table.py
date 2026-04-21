"""Generate a compact per-query diagnostics table from benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

try:
    from scripts.generate_paper_ablation_table import (
        benchmark_summary_path,
        format_missing_summaries_message,
        parse_benchmark_spec,
    )
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from generate_paper_ablation_table import (  # type: ignore
        benchmark_summary_path,
        format_missing_summaries_message,
        parse_benchmark_spec,
    )


TABLE_COLUMNS = [
    "benchmark",
    "query",
    "detector_backend",
    "skip_clip",
    "total_runs",
    "mean_raw_num_detections",
    "mean_num_detections",
    "mean_num_ranked_candidates",
    "fraction_top1_changed_by_rerank",
    "fraction_with_3d_target",
    "pick_success_rate",
    "mean_runtime_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a per-query diagnostics table from benchmark_summary.json files."
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        required=True,
        help="Benchmark entry as LABEL=DIR or just DIR. Repeat for each benchmark.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs") / "per_query_diagnostics_table.md",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs") / "per_query_diagnostics_table.csv",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip benchmark entries without benchmark_summary.json instead of failing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = build_table_rows(args.benchmark, skip_missing=args.skip_missing)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    write_rows_csv(rows, args.output_csv)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown_table(rows), encoding="utf-8")
    print(f"Wrote per-query diagnostics markdown: {args.output_md}")
    print(f"Wrote per-query diagnostics CSV:      {args.output_csv}")
    return 0


def build_table_rows(benchmark_specs: list[str], skip_missing: bool = False) -> list[dict[str, Any]]:
    """Build one diagnostics row per benchmark/query pair."""

    entries = [parse_benchmark_spec(spec) for spec in benchmark_specs]
    missing_entries = [
        (label, benchmark_dir, benchmark_summary_path(benchmark_dir))
        for label, benchmark_dir in entries
        if not benchmark_summary_path(benchmark_dir).exists()
    ]
    if missing_entries and not skip_missing:
        raise FileNotFoundError(format_missing_summaries_message(missing_entries))

    rows: list[dict[str, Any]] = []
    for label, benchmark_dir in entries:
        if not benchmark_summary_path(benchmark_dir).exists():
            continue
        rows.extend(rows_from_benchmark(label, benchmark_dir))

    if not rows:
        raise ValueError("No per-query metrics were available to include in the diagnostics table.")
    return rows


def rows_from_benchmark(label: str, benchmark_dir: str | Path) -> list[dict[str, Any]]:
    """Load per-query metrics from one benchmark summary."""

    summary = load_benchmark_summary(benchmark_dir)
    per_query_metrics = summary.get("per_query_metrics")
    if not isinstance(per_query_metrics, dict) or not per_query_metrics:
        return []

    rows: list[dict[str, Any]] = []
    detector_backend = _value(summary.get("detector_backend"))
    skip_clip = _value(summary.get("skip_clip"))
    for query in sorted(per_query_metrics):
        metrics = per_query_metrics[query]
        if not isinstance(metrics, dict):
            continue
        rows.append(row_from_query_metrics(label, str(query), detector_backend, skip_clip, metrics))
    return rows


def row_from_query_metrics(
    label: str,
    query: str,
    detector_backend: str,
    skip_clip: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Flatten one query's metrics into a diagnostics table row."""

    mean_num_detections = metrics.get("mean_num_detections")
    return {
        "benchmark": label,
        "query": query,
        "detector_backend": detector_backend,
        "skip_clip": skip_clip,
        "total_runs": _as_int(metrics.get("total_runs")),
        "mean_raw_num_detections": _as_float(metrics.get("mean_raw_num_detections", mean_num_detections)),
        "mean_num_detections": _as_float(mean_num_detections),
        "mean_num_ranked_candidates": _as_float(metrics.get("mean_num_ranked_candidates")),
        "fraction_top1_changed_by_rerank": _as_float(metrics.get("fraction_top1_changed_by_rerank")),
        "fraction_with_3d_target": _as_float(metrics.get("fraction_with_3d_target")),
        "pick_success_rate": _as_float(metrics.get("pick_success_rate")),
        "mean_runtime_seconds": _as_float(metrics.get("mean_runtime_seconds")),
    }


def load_benchmark_summary(benchmark_dir: str | Path) -> dict[str, Any]:
    """Load ``benchmark_summary.json`` from a benchmark directory."""

    path = benchmark_summary_path(benchmark_dir)
    if not path.exists():
        raise FileNotFoundError(f"Required benchmark summary not found: {path}")
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the per-query diagnostics table as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render the per-query diagnostics table as Markdown."""

    lines = [
        "# Per-Query Diagnostics Table",
        "",
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "| " + " | ".join(["---"] * 4 + ["---:"] * (len(TABLE_COLUMNS) - 4)) + " |",
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
    raise SystemExit(main())
