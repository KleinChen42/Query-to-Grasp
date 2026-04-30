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
    "env_id",
    "obs_mode",
    "pick_executor",
    "grasp_target_mode",
    "pick_target_source_counts",
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
    "mean_same_label_pairwise_distance",
    "mean_selected_overall_confidence",
    "reobserve_trigger_rate",
    "initial_reobserve_trigger_rate",
    "final_reobserve_trigger_rate",
    "closed_loop_execution_rate",
    "closed_loop_resolution_rate",
    "closed_loop_still_needed_rate",
    "mean_closed_loop_delta_selected_overall_confidence",
    "mean_closed_loop_delta_selected_num_views",
    "mean_closed_loop_delta_num_memory_objects",
    "reobserve_reason_counts",
    "grasp_attempted_rate",
    "pick_success_rate",
    "task_success_rate",
    "pick_stage_counts",
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
    parser.add_argument(
        "--oracle-pick",
        action="append",
        default=[],
        help="Oracle simulated-pick benchmark entry as LABEL=DIR or DIR. Repeat for each row.",
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
            oracle_pick_specs=args.oracle_pick,
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
    oracle_pick_specs: list[str] | None = None,
    skip_missing: bool = False,
) -> list[dict[str, Any]]:
    """Build comparison rows from single-view and fusion benchmark summaries."""

    oracle_pick_specs = oracle_pick_specs or []
    if not single_view_specs and not fusion_specs and not oracle_pick_specs:
        raise ValueError("Provide at least one --single-view, --fusion, or --oracle-pick benchmark.")

    single_entries = [parse_benchmark_spec(spec) for spec in single_view_specs]
    fusion_entries = [parse_benchmark_spec(spec) for spec in fusion_specs]
    oracle_entries = [parse_benchmark_spec(spec) for spec in oracle_pick_specs]
    all_entries = [*single_entries, *fusion_entries, *oracle_entries]
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
    for label, benchmark_dir in oracle_entries:
        if benchmark_summary_path(benchmark_dir).exists():
            rows.append(oracle_pick_row_from_benchmark(label, benchmark_dir))

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
        "env_id": _value(summary.get("env_id")),
        "obs_mode": _value(summary.get("obs_mode")),
        "pick_executor": _value(summary.get("pick_executor")),
        "grasp_target_mode": _value(summary.get("grasp_target_mode")),
        "pick_target_source_counts": pick_target_source_counts(summary, benchmark_dir),
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
        "mean_same_label_pairwise_distance": NA,
        "mean_selected_overall_confidence": NA,
        "reobserve_trigger_rate": NA,
        "initial_reobserve_trigger_rate": NA,
        "final_reobserve_trigger_rate": NA,
        "closed_loop_execution_rate": NA,
        "closed_loop_resolution_rate": NA,
        "closed_loop_still_needed_rate": NA,
        "mean_closed_loop_delta_selected_overall_confidence": NA,
        "mean_closed_loop_delta_selected_num_views": NA,
        "mean_closed_loop_delta_num_memory_objects": NA,
        "reobserve_reason_counts": NA,
        "grasp_attempted_rate": _optional_float(metrics.get("grasp_attempted_rate")),
        "pick_success_rate": _as_float(metrics.get("pick_success_rate")),
        "task_success_rate": _optional_float(metrics.get("task_success_rate")),
        "pick_stage_counts": format_reason_counts(metrics.get("pick_stage_counts")),
    }


def fusion_row_from_benchmark(label: str, benchmark_dir: str | Path) -> dict[str, Any]:
    """Load one fusion-debug benchmark summary into a comparison row."""

    summary = load_benchmark_summary(benchmark_dir)
    metrics = _metrics(summary)
    memory_metrics = load_memory_diagnostics_metrics(benchmark_dir)
    fraction_with_selected_object = _as_float(metrics.get("fraction_with_selected_object"))
    return {
        "label": label,
        "benchmark_type": "fusion_debug",
        "env_id": _value(summary.get("env_id")),
        "obs_mode": _value(summary.get("obs_mode")),
        "pick_executor": _value(summary.get("pick_executor")),
        "grasp_target_mode": _value(summary.get("grasp_target_mode")),
        "pick_target_source_counts": pick_target_source_counts(summary, benchmark_dir),
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
        "mean_same_label_pairwise_distance": _optional_float(
            memory_metrics.get("mean_same_label_pairwise_distance")
        ),
        "mean_selected_overall_confidence": _as_float(metrics.get("mean_selected_overall_confidence")),
        "reobserve_trigger_rate": _optional_float(metrics.get("reobserve_trigger_rate")),
        "initial_reobserve_trigger_rate": _optional_float(metrics.get("initial_reobserve_trigger_rate")),
        "final_reobserve_trigger_rate": _optional_float(metrics.get("final_reobserve_trigger_rate")),
        "closed_loop_execution_rate": _optional_float(metrics.get("closed_loop_execution_rate")),
        "closed_loop_resolution_rate": _optional_float(metrics.get("closed_loop_resolution_rate")),
        "closed_loop_still_needed_rate": _optional_float(metrics.get("closed_loop_still_needed_rate")),
        "mean_closed_loop_delta_selected_overall_confidence": _optional_float(
            metrics.get("mean_closed_loop_delta_selected_overall_confidence")
        ),
        "mean_closed_loop_delta_selected_num_views": _optional_float(
            metrics.get("mean_closed_loop_delta_selected_num_views")
        ),
        "mean_closed_loop_delta_num_memory_objects": _optional_float(
            metrics.get("mean_closed_loop_delta_num_memory_objects")
        ),
        "reobserve_reason_counts": format_reason_counts(metrics.get("reobserve_reason_counts")),
        "grasp_attempted_rate": _optional_float(metrics.get("grasp_attempted_rate")),
        "pick_success_rate": _optional_float(metrics.get("pick_success_rate")),
        "task_success_rate": _optional_float(metrics.get("task_success_rate")),
        "pick_stage_counts": format_reason_counts(metrics.get("pick_stage_counts")),
    }


def oracle_pick_row_from_benchmark(label: str, benchmark_dir: str | Path) -> dict[str, Any]:
    """Load one oracle simulated-pick benchmark summary into a comparison row."""

    summary = load_benchmark_summary(benchmark_dir)
    metrics = _metrics(summary)
    return {
        "label": label,
        "benchmark_type": "oracle_pick",
        "env_id": _value(summary.get("env_id")),
        "obs_mode": _value(summary.get("obs_mode")),
        "pick_executor": _value(summary.get("pick_executor")),
        "grasp_target_mode": _value(summary.get("grasp_target_mode")),
        "pick_target_source_counts": pick_target_source_counts(summary, benchmark_dir),
        "detector_backend": _value(summary.get("detector_backend")),
        "skip_clip": _value(summary.get("skip_clip")),
        "total_runs": _as_int(summary.get("total_runs", metrics.get("total_runs"))),
        "primary_rate_name": "pick_success_rate",
        "primary_rate": _as_float(metrics.get("pick_success_rate")),
        "mean_runtime_seconds": _as_float(metrics.get("mean_runtime_seconds")),
        "mean_raw_num_detections": NA,
        "mean_num_ranked_candidates": NA,
        "mean_num_views": NA,
        "mean_num_memory_objects": NA,
        "mean_num_observations_added": NA,
        "mean_same_label_pairwise_distance": NA,
        "mean_selected_overall_confidence": NA,
        "reobserve_trigger_rate": NA,
        "initial_reobserve_trigger_rate": NA,
        "final_reobserve_trigger_rate": NA,
        "closed_loop_execution_rate": NA,
        "closed_loop_resolution_rate": NA,
        "closed_loop_still_needed_rate": NA,
        "mean_closed_loop_delta_selected_overall_confidence": NA,
        "mean_closed_loop_delta_selected_num_views": NA,
        "mean_closed_loop_delta_num_memory_objects": NA,
        "reobserve_reason_counts": NA,
        "grasp_attempted_rate": _optional_float(metrics.get("grasp_attempted_rate")),
        "pick_success_rate": _optional_float(metrics.get("pick_success_rate")),
        "task_success_rate": _optional_float(metrics.get("task_success_rate")),
        "pick_stage_counts": format_reason_counts(metrics.get("pick_stage_counts")),
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


def load_memory_diagnostics_metrics(benchmark_dir: str | Path) -> dict[str, Any]:
    """Load optional memory diagnostics aggregate metrics for fusion rows."""

    path = Path(benchmark_dir) / "memory_diagnostics.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        return {}
    aggregate = data.get("aggregate")
    return aggregate if isinstance(aggregate, dict) else {}


def pick_target_source_counts(summary: dict[str, Any], benchmark_dir: str | Path) -> str:
    """Return compact pick-target source counts from summary metrics or rows."""

    metrics = _metrics(summary)
    counts = metrics.get("pick_target_source_counts")
    if isinstance(counts, dict) and counts:
        return format_reason_counts(counts)
    source = summary.get("pick_target_source")
    total_runs = _as_int(summary.get("total_runs", metrics.get("total_runs")))
    if source is not None and total_runs > 0:
        return format_reason_counts({str(source): total_runs})
    row_counts = load_pick_target_source_counts(benchmark_dir)
    return format_reason_counts(row_counts)


def load_pick_target_source_counts(benchmark_dir: str | Path) -> dict[str, int]:
    """Load optional per-row pick target source counts from benchmark rows."""

    benchmark_dir = Path(benchmark_dir)
    rows_json = benchmark_dir / "benchmark_rows.json"
    rows_csv = benchmark_dir / "benchmark_rows.csv"
    counts: dict[str, int] = {}
    try:
        if rows_json.exists():
            data = json.loads(rows_json.read_text(encoding="utf-8-sig"))
            rows = data if isinstance(data, list) else []
            for row in rows:
                if isinstance(row, dict):
                    _add_source_count(counts, row.get("pick_target_source"))
        elif rows_csv.exists():
            with rows_csv.open("r", encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    _add_source_count(counts, row.get("pick_target_source"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(sorted(counts.items()))


def _add_source_count(counts: dict[str, int], source: Any) -> None:
    if source is None or str(source).strip() == "":
        return
    key = str(source)
    counts[key] = counts.get(key, 0) + 1


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
        "| " + " | ".join(["---"] * 10 + ["---:"] * (len(TABLE_COLUMNS) - 10)) + " |",
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


def format_reason_counts(value: Any) -> str:
    """Render re-observation reason counts in one compact table cell."""

    if not isinstance(value, dict) or not value:
        return NA
    parts = [
        f"{reason}: {count}"
        for reason, count in sorted(value.items(), key=lambda item: (-_as_int(item[1]), str(item[0])))
    ]
    return "; ".join(parts)


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


def _optional_float(value: Any) -> float | str:
    try:
        if value is None:
            return NA
        return float(value)
    except (TypeError, ValueError):
        return NA


def _as_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
