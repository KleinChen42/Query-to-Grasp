"""Generate diagnostics for StackCube multi-view simulated-pick failures."""

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


FAILURE_CSV_COLUMNS = [
    "benchmark_label",
    "query",
    "seed",
    "pick_success",
    "failure_class",
    "pick_stage",
    "pick_target_source",
    "selected_semantic_to_grasp_xy_distance",
    "selected_grasp_observation_count",
    "selected_grasp_observation_xy_spread",
    "selected_grasp_observation_max_distance_to_fused",
    "closed_loop_extra_view_third_object_involved",
    "final_reobserve_reason",
    "artifacts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify StackCube multi-view grasp failures from benchmark rows and per-run artifacts."
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Benchmark entry as LABEL=DIR or DIR. Repeat for tabletop/closed-loop modes.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs") / "stackcube_grasp_failure_diagnosis.md",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs") / "stackcube_grasp_failure_diagnosis.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs") / "stackcube_grasp_failure_diagnosis.csv",
    )
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing benchmark summaries/rows.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_failure_report(args.benchmark, skip_missing=args.skip_missing)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_failure_csv(report["failure_rows"], args.output_csv)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote StackCube grasp diagnosis markdown: {args.output_md}")
    print(f"Wrote StackCube grasp diagnosis JSON:     {args.output_json}")
    print(f"Wrote StackCube grasp diagnosis CSV:      {args.output_csv}")
    return 0


def build_failure_report(benchmark_specs: list[str], skip_missing: bool = False) -> dict[str, Any]:
    """Build a diagnosis report from one or more fusion benchmark directories."""

    if not benchmark_specs:
        raise ValueError("Provide at least one --benchmark entry.")

    entries = [parse_benchmark_spec(spec) for spec in benchmark_specs]
    missing = [
        (label, benchmark_dir, benchmark_summary_path(benchmark_dir))
        for label, benchmark_dir in entries
        if not benchmark_summary_path(benchmark_dir).exists()
    ]
    missing_rows = [
        (label, benchmark_dir, Path(benchmark_dir) / "benchmark_rows.csv")
        for label, benchmark_dir in entries
        if not (Path(benchmark_dir) / "benchmark_rows.csv").exists()
    ]
    if (missing or missing_rows) and not skip_missing:
        messages = []
        if missing:
            messages.append(format_missing_summaries_message(missing))
        if missing_rows:
            messages.extend(f"Missing benchmark_rows.csv for {label}: {path}" for label, _, path in missing_rows)
        raise FileNotFoundError("\n".join(messages))

    benchmark_reports: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for label, benchmark_dir in entries:
        benchmark_dir = Path(benchmark_dir)
        summary_path = benchmark_summary_path(benchmark_dir)
        rows_path = benchmark_dir / "benchmark_rows.csv"
        if not summary_path.exists() or not rows_path.exists():
            continue
        summary = load_json_dict(summary_path)
        rows = load_csv_rows(rows_path)
        diagnosed_rows = [diagnose_row(label, row) for row in rows]
        failure_rows.extend(row for row in diagnosed_rows if row["failure_class"] != "success")
        benchmark_reports.append(benchmark_report(label, benchmark_dir, summary, diagnosed_rows))

    if not benchmark_reports:
        raise ValueError("No benchmark summaries and rows were available to diagnose.")

    return {
        "benchmarks": benchmark_reports,
        "failure_class_counts": count_values(row["failure_class"] for row in failure_rows),
        "failure_rows": failure_rows,
        "conclusion": build_conclusion(failure_rows),
    }


def benchmark_report(
    label: str,
    benchmark_dir: Path,
    summary: dict[str, Any],
    diagnosed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = summary.get("aggregate_metrics") if isinstance(summary.get("aggregate_metrics"), dict) else {}
    failure_rows = [row for row in diagnosed_rows if row["failure_class"] != "success"]
    return {
        "label": label,
        "benchmark_dir": str(benchmark_dir),
        "env_id": summary.get("env_id", "unknown"),
        "view_preset": summary.get("view_preset", "unknown"),
        "closed_loop_reobserve_enabled": bool(summary.get("closed_loop_reobserve_enabled", False)),
        "total_runs": as_int(summary.get("total_runs", metrics.get("total_runs"))),
        "failed_runs": as_int(metrics.get("failed_runs")),
        "pick_success_rate": as_float(metrics.get("pick_success_rate")),
        "failure_count": len(failure_rows),
        "failure_class_counts": count_values(row["failure_class"] for row in failure_rows),
        "mean_selected_semantic_to_grasp_xy_distance": as_float(
            metrics.get("mean_selected_semantic_to_grasp_xy_distance")
        ),
        "mean_selected_grasp_observation_xy_spread": as_float(
            metrics.get("mean_selected_grasp_observation_xy_spread")
        ),
        "mean_selected_grasp_observation_max_distance_to_fused": as_float(
            metrics.get("mean_selected_grasp_observation_max_distance_to_fused")
        ),
    }


def diagnose_row(label: str, row: dict[str, str]) -> dict[str, Any]:
    pick_success = as_bool(row.get("pick_success"))
    failure_class = classify_failure(row)
    return {
        "benchmark_label": label,
        "query": row.get("query", ""),
        "seed": as_int(row.get("seed")),
        "pick_success": pick_success,
        "failure_class": failure_class,
        "pick_stage": row.get("pick_stage", ""),
        "pick_target_source": row.get("pick_target_source", ""),
        "selected_semantic_to_grasp_xy_distance": as_float(
            row.get("selected_semantic_to_grasp_xy_distance")
        ),
        "selected_grasp_observation_count": as_int(row.get("selected_grasp_observation_count")),
        "selected_grasp_observation_xy_spread": as_float(row.get("selected_grasp_observation_xy_spread")),
        "selected_grasp_observation_max_distance_to_fused": as_float(
            row.get("selected_grasp_observation_max_distance_to_fused")
        ),
        "closed_loop_extra_view_third_object_involved": as_bool(
            row.get("closed_loop_extra_view_third_object_involved")
        ),
        "final_reobserve_reason": row.get("final_reobserve_reason") or row.get("reobserve_reason") or "",
        "artifacts": row.get("artifacts", ""),
    }


def classify_failure(row: dict[str, str]) -> str:
    """Classify a row using behavior-preserving diagnostic heuristics."""

    if as_bool(row.get("pick_success")):
        return "success"
    if as_bool(row.get("run_failed")):
        return "run_failed"
    if not as_bool(row.get("has_selected_object")):
        return "selected_object_missing"
    if not as_bool(row.get("grasp_attempted")):
        return "pick_not_attempted"
    if as_bool(row.get("closed_loop_extra_view_third_object_involved")):
        return "third_object_absorption"

    semantic_grasp_xy = as_float(row.get("selected_semantic_to_grasp_xy_distance"))
    grasp_xy_spread = as_float(row.get("selected_grasp_observation_xy_spread"))
    max_to_fused = as_float(row.get("selected_grasp_observation_max_distance_to_fused"))
    if max(semantic_grasp_xy, grasp_xy_spread, max_to_fused) >= 0.08:
        return "wrong_fused_grasp_observation"

    if as_bool(row.get("final_should_reobserve")) or as_bool(row.get("closed_loop_reobserve_still_needed")):
        return "memory_fragmentation_or_low_support"
    return "controller_contact_failure"


def build_conclusion(failure_rows: list[dict[str, Any]]) -> str:
    if not failure_rows:
        return "No pick failures were present in the diagnosed rows."
    counts = count_values(row["failure_class"] for row in failure_rows)
    dominant, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return f"Dominant failure class: {dominant} ({count}/{len(failure_rows)} failures)."


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# StackCube Multi-View Grasp Failure Diagnosis",
        "",
        report["conclusion"],
        "",
        "## Benchmarks",
        "",
        "| label | runs | pick success | failures | top failure classes | mean semantic-grasp XY | mean grasp XY spread |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for benchmark in report["benchmarks"]:
        lines.append(
            "| {label} | {total_runs} | {pick_success_rate:.4f} | {failure_count} | {failure_classes} | "
            "{semantic_xy:.4f} | {spread:.4f} |".format(
                label=benchmark["label"],
                total_runs=benchmark["total_runs"],
                pick_success_rate=benchmark["pick_success_rate"],
                failure_count=benchmark["failure_count"],
                failure_classes=format_counts(benchmark["failure_class_counts"]),
                semantic_xy=benchmark["mean_selected_semantic_to_grasp_xy_distance"],
                spread=benchmark["mean_selected_grasp_observation_xy_spread"],
            )
        )
    lines.extend(["", "## Failure Rows", ""])
    lines.extend(render_failure_examples(report["failure_rows"]))
    return "\n".join(lines) + "\n"


def render_failure_examples(rows: list[dict[str, Any]], max_rows: int = 20) -> list[str]:
    if not rows:
        return ["No failures."]
    lines = [
        "| benchmark | seed | class | stage | source | semantic-grasp XY | grasp XY spread | third object |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in rows[:max_rows]:
        lines.append(
            "| {benchmark_label} | {seed} | {failure_class} | {pick_stage} | {pick_target_source} | "
            "{selected_semantic_to_grasp_xy_distance:.4f} | {selected_grasp_observation_xy_spread:.4f} | "
            "{closed_loop_extra_view_third_object_involved} |".format(**row)
        )
    if len(rows) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(rows)} failures.")
    return lines


def write_failure_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FAILURE_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def load_json_dict(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


def count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "none")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def format_counts(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return "none"
    return "; ".join(f"{key}: {value[key]}" for key in sorted(value))


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def as_int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def as_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
