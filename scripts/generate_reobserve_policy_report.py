"""Generate a compact report for rule-based re-observation policy diagnostics."""

from __future__ import annotations

import argparse
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


NON_TRIGGER_REASONS = {"confident_enough", "none"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-facing Markdown/JSON report for re-observation policy decisions."
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help="Fusion benchmark entry as LABEL=DIR or DIR. Repeat for each benchmark.",
    )
    parser.add_argument(
        "--benchmark-dir",
        action="append",
        default=[],
        help="Compatibility alias for --benchmark using the directory name as the label.",
    )
    parser.add_argument("--output-md", type=Path, default=Path("outputs") / "reobserve_policy_report.md")
    parser.add_argument("--output-json", type=Path, default=Path("outputs") / "reobserve_policy_report.json")
    parser.add_argument("--max-examples", type=int, default=8, help="Maximum triggered per-run examples to include.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing benchmark summaries instead of failing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = list(args.benchmark) + list(args.benchmark_dir)
    try:
        report = build_policy_report(
            specs,
            skip_missing=args.skip_missing,
            max_examples=args.max_examples,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote re-observation policy markdown: {args.output_md}")
    print(f"Wrote re-observation policy JSON:     {args.output_json}")
    return 0


def build_policy_report(
    benchmark_specs: list[str],
    skip_missing: bool = False,
    max_examples: int = 8,
) -> dict[str, Any]:
    """Build a re-observation policy report from fusion benchmark summaries."""

    if not benchmark_specs:
        raise ValueError("Provide at least one --benchmark or --benchmark-dir.")

    entries = [parse_benchmark_spec(spec) for spec in benchmark_specs]
    missing_entries = [
        (label, benchmark_dir, benchmark_summary_path(benchmark_dir))
        for label, benchmark_dir in entries
        if not benchmark_summary_path(benchmark_dir).exists()
    ]
    if missing_entries and not skip_missing:
        raise FileNotFoundError(format_missing_summaries_message(missing_entries))

    benchmark_reports = [
        benchmark_report(label, benchmark_dir, max_examples=max_examples)
        for label, benchmark_dir in entries
        if benchmark_summary_path(benchmark_dir).exists()
    ]
    if not benchmark_reports:
        raise ValueError("No benchmark summaries were available to include in the re-observation policy report.")

    return {
        "benchmarks": benchmark_reports,
        "conclusion": build_conclusion(benchmark_reports),
    }


def benchmark_report(label: str, benchmark_dir: Path, max_examples: int) -> dict[str, Any]:
    """Build report fields for one benchmark directory."""

    summary = load_json_dict(benchmark_summary_path(benchmark_dir))
    metrics = _metrics(summary)
    rows = load_optional_json_list(Path(benchmark_dir) / "benchmark_rows.json")
    has_reobserve_metrics = has_policy_metrics(metrics)
    return {
        "label": label,
        "benchmark_dir": str(benchmark_dir),
        "detector_backend": _value(summary.get("detector_backend")),
        "skip_clip": _value(summary.get("skip_clip")),
        "view_preset": _value(summary.get("view_preset", "none")),
        "closed_loop_reobserve_enabled": _as_bool(summary.get("closed_loop_reobserve_enabled")),
        "total_runs": _as_int(summary.get("total_runs", metrics.get("total_runs"))),
        "fraction_with_selected_object": _as_float(metrics.get("fraction_with_selected_object")),
        "mean_selected_overall_confidence": _as_float(metrics.get("mean_selected_overall_confidence")),
        "mean_num_views": _as_float(metrics.get("mean_num_views")),
        "mean_num_memory_objects": _as_float(metrics.get("mean_num_memory_objects")),
        "reobserve_metrics_available": has_reobserve_metrics,
        "reobserve_trigger_rate": _optional_float(metrics.get("reobserve_trigger_rate")),
        "initial_reobserve_trigger_rate": _optional_float(metrics.get("initial_reobserve_trigger_rate")),
        "final_reobserve_trigger_rate": _optional_float(metrics.get("final_reobserve_trigger_rate")),
        "closed_loop_execution_rate": _optional_float(metrics.get("closed_loop_execution_rate")),
        "reobserve_reason_counts": _reason_counts(metrics.get("reobserve_reason_counts")),
        "initial_reobserve_reason_counts": _reason_counts(metrics.get("initial_reobserve_reason_counts")),
        "final_reobserve_reason_counts": _reason_counts(metrics.get("final_reobserve_reason_counts")),
        "per_query": per_query_report(summary.get("per_query_metrics")),
        "trigger_examples": trigger_examples(rows, max_examples=max_examples),
    }


def per_query_report(value: Any) -> list[dict[str, Any]]:
    """Flatten per-query policy metrics for Markdown/JSON output."""

    if not isinstance(value, dict):
        return []
    rows: list[dict[str, Any]] = []
    for query, metrics in sorted(value.items()):
        if not isinstance(metrics, dict):
            continue
        rows.append(
            {
                "query": str(query),
                "total_runs": _as_int(metrics.get("total_runs")),
                "fraction_with_selected_object": _as_float(metrics.get("fraction_with_selected_object")),
                "reobserve_metrics_available": has_policy_metrics(metrics),
                "reobserve_trigger_rate": _optional_float(metrics.get("reobserve_trigger_rate")),
                "initial_reobserve_trigger_rate": _optional_float(metrics.get("initial_reobserve_trigger_rate")),
                "final_reobserve_trigger_rate": _optional_float(metrics.get("final_reobserve_trigger_rate")),
                "closed_loop_execution_rate": _optional_float(metrics.get("closed_loop_execution_rate")),
                "reobserve_reason_counts": _reason_counts(metrics.get("reobserve_reason_counts")),
                "mean_selected_overall_confidence": _as_float(metrics.get("mean_selected_overall_confidence")),
            }
        )
    return rows


def has_policy_metrics(metrics: dict[str, Any]) -> bool:
    """Return whether a metrics dictionary contains re-observation policy fields."""

    return any(
        key in metrics
        for key in (
            "reobserve_trigger_rate",
            "initial_reobserve_trigger_rate",
            "final_reobserve_trigger_rate",
            "closed_loop_execution_rate",
            "reobserve_reason_counts",
        )
    )


def trigger_examples(rows: list[dict[str, Any]], max_examples: int) -> list[dict[str, Any]]:
    """Return compact examples where the policy requested re-observation."""

    examples: list[dict[str, Any]] = []
    for row in rows:
        if not _as_bool(row.get("should_reobserve")):
            continue
        examples.append(
            {
                "query": _value(row.get("query")),
                "seed": _as_int(row.get("seed")),
                "reason": _value(row.get("reobserve_reason")),
                "selected_overall_confidence": _as_float(row.get("selected_overall_confidence")),
                "artifacts": _value(row.get("artifacts")),
            }
        )
        if len(examples) >= max(0, int(max_examples)):
            break
    return examples


def build_conclusion(benchmark_reports: list[dict[str, Any]]) -> str:
    """Return a small rule-based conclusion for paper notes."""

    available_reports = [report for report in benchmark_reports if report.get("reobserve_metrics_available")]
    if not available_reports:
        return (
            "Re-observation policy metrics were not available in the included benchmark summaries. "
            "Rerun the fusion benchmarks after the policy patch to measure trigger rates and reason counts."
        )

    closed_loop_reports = [
        report
        for report in available_reports
        if _as_float(report.get("closed_loop_execution_rate")) > 0.0
        and report.get("initial_reobserve_trigger_rate") is not None
        and report.get("final_reobserve_trigger_rate") is not None
    ]
    if closed_loop_reports:
        initial_rate = _mean(_as_float(report.get("initial_reobserve_trigger_rate")) for report in closed_loop_reports)
        final_rate = _mean(_as_float(report.get("final_reobserve_trigger_rate")) for report in closed_loop_reports)
        execution_rate = _mean(_as_float(report.get("closed_loop_execution_rate")) for report in closed_loop_reports)
        if final_rate < initial_rate:
            return (
                "Closed-loop re-observation produced measurable diagnostic signal and reduced the mean policy "
                f"trigger rate from {initial_rate:.4f} to {final_rate:.4f} across included closed-loop benchmarks "
                f"(execution rate {execution_rate:.4f})."
            )
        if final_rate == initial_rate:
            return (
                "Closed-loop re-observation executed in the included benchmarks, but did not reduce the mean policy "
                f"trigger rate ({initial_rate:.4f} before and {final_rate:.4f} after; execution rate "
                f"{execution_rate:.4f}). This suggests the added views did not resolve the dominant uncertainty."
            )
        return (
            "Closed-loop re-observation executed in the included benchmarks, but the mean policy trigger rate "
            f"increased from {initial_rate:.4f} to {final_rate:.4f} (execution rate {execution_rate:.4f}). "
            "Inspect per-query artifacts before treating the extra-view policy as beneficial."
        )

    max_rate = max((_as_float(report.get("reobserve_trigger_rate")) for report in available_reports), default=0.0)
    all_reasons: dict[str, int] = {}
    for report in available_reports:
        for reason, count in report["reobserve_reason_counts"].items():
            if reason in NON_TRIGGER_REASONS:
                continue
            all_reasons[reason] = all_reasons.get(reason, 0) + int(count)

    if max_rate == 0.0:
        return (
            "The current policy did not request additional views in the included benchmarks. "
            "This suggests either confident fused selections or policy thresholds that are too conservative."
        )
    if all_reasons:
        top_reason = sorted(all_reasons.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return (
            "The re-observation policy produced measurable diagnostic signal. "
            f"The most common triggered reason was `{top_reason}`, which should guide the next closed-loop experiment."
        )
    return "The policy trigger rate is nonzero, but reason counts were unavailable in the included summaries."


def render_markdown(report: dict[str, Any]) -> str:
    """Render the policy report as Markdown."""

    lines = [
        "# Re-Observation Policy Diagnostics",
        "",
        "This report summarizes the rule-based policy that decides whether another view would be useful.",
        "",
        "## Benchmarks",
        "",
        "| label | runs | view_preset | selected_frac | mean_confidence | reobserve_metrics | reobserve_trigger_rate | initial_trigger_rate | final_trigger_rate | closed_loop_execution_rate | reason_counts |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["benchmarks"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(row["label"]),
                    str(row["total_runs"]),
                    _escape(row["view_preset"]),
                    _format_float(row["fraction_with_selected_object"]),
                    _format_float(row["mean_selected_overall_confidence"]),
                    "yes" if row["reobserve_metrics_available"] else "no",
                    _format_optional_float(row["reobserve_trigger_rate"]),
                    _format_optional_float(row["initial_reobserve_trigger_rate"]),
                    _format_optional_float(row["final_reobserve_trigger_rate"]),
                    _format_optional_float(row["closed_loop_execution_rate"]),
                    _escape(format_reason_counts(row["reobserve_reason_counts"])),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Per-Query Breakdown", ""])
    for benchmark in report["benchmarks"]:
        lines.extend(
            [
                f"### {benchmark['label']}",
                "",
                "| query | runs | selected_frac | mean_confidence | reobserve_metrics | reobserve_trigger_rate | initial_trigger_rate | final_trigger_rate | closed_loop_execution_rate | reason_counts |",
                "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        if not benchmark["per_query"]:
            lines.append("| n/a | 0 | 0.0000 | 0.0000 | no | n/a | n/a | n/a | n/a | n/a |")
        for row in benchmark["per_query"]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape(row["query"]),
                        str(row["total_runs"]),
                        _format_float(row["fraction_with_selected_object"]),
                        _format_float(row["mean_selected_overall_confidence"]),
                        "yes" if row["reobserve_metrics_available"] else "no",
                        _format_optional_float(row["reobserve_trigger_rate"]),
                        _format_optional_float(row["initial_reobserve_trigger_rate"]),
                        _format_optional_float(row["final_reobserve_trigger_rate"]),
                        _format_optional_float(row["closed_loop_execution_rate"]),
                        _escape(format_reason_counts(row["reobserve_reason_counts"])),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(
        [
            "## Trigger Examples",
            "",
            "| benchmark | query | seed | reason | confidence | artifacts |",
            "| --- | --- | ---: | --- | ---: | --- |",
        ]
    )
    had_example = False
    for benchmark in report["benchmarks"]:
        for example in benchmark["trigger_examples"]:
            had_example = True
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape(benchmark["label"]),
                        _escape(example["query"]),
                        str(example["seed"]),
                        _escape(example["reason"]),
                        _format_float(example["selected_overall_confidence"]),
                        f"`{_escape(example['artifacts'])}`",
                    ]
                )
                + " |"
            )
    if not had_example:
        lines.append("| n/a | n/a | 0 | none | 0.0000 | n/a |")

    lines.extend(["", "## Conclusion", "", report["conclusion"], ""])
    return "\n".join(lines)


def load_json_dict(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def load_optional_json_list(path: Path) -> list[dict[str, Any]]:
    """Load an optional JSON list of objects."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("aggregate_metrics")
    return metrics if isinstance(metrics, dict) else {}


def _reason_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, count in value.items():
        counts[str(key or "none")] = _as_int(count)
    return counts


def format_reason_counts(counts: dict[str, int]) -> str:
    """Render reason counts as a compact table cell."""

    if not counts:
        return "n/a"
    parts = [f"{reason}: {count}" for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    return "; ".join(parts)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _value(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)


def _format_float(value: Any) -> str:
    return f"{_as_float(value):.4f}"


def _format_optional_float(value: Any) -> str:
    parsed = _optional_float(value)
    return "n/a" if parsed is None else f"{parsed:.4f}"


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _mean(values: Any) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


if __name__ == "__main__":
    raise SystemExit(main())
