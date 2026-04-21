"""Generate a compact single-view vs fusion-debug comparison table."""

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
    "label",
    "benchmark_type",
    "detector_backend",
    "skip_clip",
    "total_runs",
    "primary_rate_name",
    "primary_rate",
    "mean_runtime_seconds",
    "mean_raw_num_detections",
    "mean_num_ranked_candidates",
    "mean_num_views",
    "mean_num_memory_objects",
    "mean_num_observations_added",
    "mean_selected_overall_confidence",
    "pick_success_rate",
]

NA = "n/a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-ready comparison table for single-view and fusion-debug benchmark summaries."
    )
    parser.add_argument(
        "--single-view",
        action="append",
        default=[],
        help="Single-view benchmark entry as LABEL=DIR or DIR. Repeat for each row.",
    )
    parser.add_argument(
        "--fusion",
        action="append",
        default=[],
        help="Fusion-debug benchmark entry as LABEL=DIR or DIR. Repeat for each row.",
    )
    parser.add_argument("--output-md", type=Path, default=Path("outputs") / "fusion_comparison_table.md")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs") / "fusion_comparison_table.csv")
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing benchmark summaries instead of failing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = build_table_rows(
            single_view_specs=args.single_view,
            fusion_specs=args.fusion,
            skip_missing=args.skip_missing,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    write_rows_csv(rows, args.output_csv)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown_table(rows), encoding="utf-8")
    print(f"Wrote fusion comparison markdown: {args.output_md}")
    print(f"Wrote fusion comparison CSV:      {args.output_csv}")
    return 0


def build_table_rows(
    single_view_specs: list[str],
    fusion_specs: list[str],
    skip_missing: bool = False,
) -> list[dict[str, Any]]:
    """Build comparison rows from single-view and fusion benchmark summaries."""

    if not single_view_specs and not fusion_specs:
        raise ValueError("Provide at least one --single-view or --fusion benchmark.")

    single_entries = [parse_benchmark_spec(spec) for spec in single_view_specs]
    fusion_entries = [parse_benchmark_spec(spec) for spec in fusion_specs]
    all_entries = [*single_entries, *fusion_entries]
    missing_entries = [
        (label, benchmark_dir, benchmark_summary_path(benchmark_dir))
        for label, benchmark_dir in all_entries
        if not benchmark_summary_path(benchmark_dir).exists()
    ]
    if missing_entries and not skip_missing:
        raise FileNotFoundError(format_missing_summaries_message(missing_entries))

    rows: list[dict[str, Any]] = []
    for label, benchmark_dir in single_entries:
        if benchmark_summary_path(benchmark_dir).exists():
            rows.append(single_view_row_from_benchmark(label, benchmark_dir))
    for label, benchmark_dir in fusion_entries:
        if benchmark_summary_path(benchmark_dir).exists():
            rows.append(fusion_row_from_benchmark(label, benchmark_dir))

    if not rows:
        raise ValueError("No benchmark summaries were available to include in the fusion comparison table.")
    return rows


def single_view_row_from_benchmark(label: str, benchmark_dir: str | Path) -> dict[str, Any]:
    """Load one single-view benchmark summary into a comparison row."""

    summary = load_benchmark_summary(benchmark_dir)
    metrics = _metrics(summary)
    fraction_with_3d_target = _as_float(metrics.get("fraction_with_3d_target"))
    return {
        "label": label,
        "benchmark_type": "single_view_pick",
        "detector_backend": _value(summary.get("detector_backend")),
        "skip_clip": _value(summary.get("skip_clip")),
        "total_runs": _as_int(summary.get("total_runs", metrics.get("total_runs"))),
        "primary_rate_name": "fraction_with_3d_target",
        "primary_rate": fraction_with_3d_target,
        "mean_runtime_seconds": _as_float(metrics.get("mean_runtime_seconds")),
        "mean_raw_num_detections": _as_float(metrics.get("mean_raw_num_detections", metrics.get("mean_num_detections"))),
        "mean_num_ranked_candidates": _as_float(metrics.get("mean_num_ranked_candidates")),
        "mean_num_views": 1.0,
        "mean_num_memory_objects": NA,
        "mean_num_observations_added": NA,
        "mean_selected_overall_confidence": NA,
        "pick_success_rate": _as_float(metrics.get("pick_success_rate")),
    }


def fusion_row_from_benchmark(label: str, benchmark_dir: str | Path) -> dict[str, Any]:
    """Load one fusion-debug benchmark summary into a comparison row."""

    summary = load_benchmark_summary(benchmark_dir)
    metrics = _metrics(summary)
    fraction_with_selected_object = _as_float(metrics.get("fraction_with_selected_object"))
    return {
        "label": label,
        "benchmark_type": "fusion_debug",
        "detector_backend": _value(summary.get("detector_backend")),
        "skip_clip": _value(summary.get("skip_clip")),
        "total_runs": _as_int(summary.get("total_runs", metrics.get("total_runs"))),
        "primary_rate_name": "fraction_with_selected_object",
        "primary_rate": fraction_with_selected_object,
        "mean_runtime_seconds": _as_float(metrics.get("mean_runtime_seconds")),
        "mean_raw_num_detections": NA,
        "mean_num_ranked_candidates": NA,
        "mean_num_views": _as_float(metrics.get("mean_num_views")),
        "mean_num_memory_objects": _as_float(metrics.get("mean_num_memory_objects")),
        "mean_num_observations_added": _as_float(metrics.get("mean_num_observations_added")),
        "mean_selected_overall_confidence": _as_float(metrics.get("mean_selected_overall_confidence")),
        "pick_success_rate": NA,
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
    """Write the comparison table as CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render the comparison table as Markdown."""

    lines = [
        "# Single-View vs Fusion Comparison Table",
        "",
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "| " + " | ".join(["---"] * 6 + ["---:"] * (len(TABLE_COLUMNS) - 6)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column)) for column in TABLE_COLUMNS) + " |")
    lines.append("")
    return "\n".join(lines)


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("aggregate_metrics")
    return metrics if isinstance(metrics, dict) else {}


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
