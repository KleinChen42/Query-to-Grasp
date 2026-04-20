"""Generate a compact markdown report from benchmark outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import aggregate_runs_by_query  # noqa: E402
from src.io.export_utils import write_json  # noqa: E402

METRIC_KEYS = [
    "total_runs",
    "mean_raw_num_detections",
    "mean_num_detections",
    "mean_num_ranked_candidates",
    "mean_num_3d_points",
    "fraction_with_3d_target",
    "pick_success_rate",
    "fraction_top1_changed_by_rerank",
    "mean_runtime_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a markdown report from benchmark summary outputs.")
    parser.add_argument("--benchmark-dir", type=Path, required=True, help="Directory containing benchmark_rows.json and benchmark_summary.json.")
    parser.add_argument("--compare-benchmark-dir", type=Path, default=None, help="Optional second benchmark directory to compare against.")
    parser.add_argument("--output-md", type=Path, default=None, help="Output markdown path. Defaults to <benchmark-dir>/report.md.")
    parser.add_argument("--output-json", type=Path, default=None, help="Output JSON path. Defaults to <benchmark-dir>/report_summary.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_md = args.output_md or args.benchmark_dir / "report.md"
    output_json = args.output_json or args.benchmark_dir / "report_summary.json"
    generate_report(
        benchmark_dir=args.benchmark_dir,
        compare_benchmark_dir=args.compare_benchmark_dir,
        output_md=output_md,
        output_json=output_json,
    )
    print(f"Wrote benchmark report: {output_md}")
    print(f"Wrote report summary:   {output_json}")


def generate_report(
    benchmark_dir: str | Path,
    compare_benchmark_dir: str | Path | None = None,
    output_md: str | Path | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Generate ``report.md`` and ``report_summary.json`` from benchmark artifacts."""

    primary_dir = Path(benchmark_dir)
    secondary_dir = Path(compare_benchmark_dir) if compare_benchmark_dir is not None else None
    output_md_path = Path(output_md) if output_md is not None else primary_dir / "report.md"
    output_json_path = Path(output_json) if output_json is not None else primary_dir / "report_summary.json"

    primary_rows = load_benchmark_rows(primary_dir)
    primary_summary = load_benchmark_summary(primary_dir)
    secondary_summary = load_benchmark_summary(secondary_dir) if secondary_dir is not None else None
    comparison_metrics = compute_comparison_metrics(primary_summary, secondary_summary) if secondary_summary is not None else None

    markdown = render_markdown_report(
        benchmark_dir=primary_dir,
        rows=primary_rows,
        primary_summary=primary_summary,
        compare_benchmark_dir=secondary_dir,
        secondary_summary=secondary_summary,
        comparison_metrics=comparison_metrics,
    )
    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(markdown, encoding="utf-8")

    report_summary = {
        "benchmark_dir": str(primary_dir),
        "primary_summary": primary_summary,
    }
    if secondary_dir is not None:
        report_summary["compare_benchmark_dir"] = str(secondary_dir)
        report_summary["secondary_summary"] = secondary_summary
        report_summary["comparison_metrics"] = comparison_metrics
    write_json(report_summary, output_json_path)
    return report_summary


def load_benchmark_rows(benchmark_dir: str | Path) -> list[dict[str, Any]]:
    """Load ``benchmark_rows.json`` from a benchmark directory."""

    path = Path(benchmark_dir) / "benchmark_rows.json"
    data = _load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}.")
    return [row if isinstance(row, dict) else {"value": row} for row in data]


def load_benchmark_summary(benchmark_dir: str | Path | None) -> dict[str, Any]:
    """Load ``benchmark_summary.json`` from a benchmark directory."""

    if benchmark_dir is None:
        raise ValueError("benchmark_dir cannot be None.")
    path = Path(benchmark_dir) / "benchmark_summary.json"
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}, got {type(data).__name__}.")
    return data


def compute_comparison_metrics(primary_summary: dict[str, Any], secondary_summary: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Compute primary, secondary, and delta values for report metrics."""

    primary_metrics = _metrics(primary_summary)
    secondary_metrics = _metrics(secondary_summary)
    comparison: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        primary_value = _as_float(primary_metrics.get(key))
        secondary_value = _as_float(secondary_metrics.get(key))
        comparison[key] = {
            "primary": primary_value,
            "secondary": secondary_value,
            "delta": primary_value - secondary_value,
        }
    return comparison


def render_markdown_report(
    benchmark_dir: Path,
    rows: list[dict[str, Any]],
    primary_summary: dict[str, Any],
    compare_benchmark_dir: Path | None = None,
    secondary_summary: dict[str, Any] | None = None,
    comparison_metrics: dict[str, dict[str, float]] | None = None,
) -> str:
    """Render a compact markdown report."""

    metrics = _metrics(primary_summary)
    per_query_metrics = _per_query_metrics(primary_summary, rows)
    lines = [
        "# Benchmark Report",
        "",
        f"- Benchmark directory: `{benchmark_dir}`",
        f"- Timestamp: {_value(primary_summary.get('timestamp'))}",
        f"- Total runs: {_value(primary_summary.get('total_runs', len(rows)))}",
        f"- Unique queries: {_format_queries(primary_summary.get('unique_queries'))}",
        f"- Detector backend: {_value(primary_summary.get('detector_backend'))}",
        f"- Skip CLIP: {_value(primary_summary.get('skip_clip'))}",
        f"- Depth scale: {_value(primary_summary.get('depth_scale'))}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in METRIC_KEYS:
        lines.append(f"| {key} | {_format_number(metrics.get(key))} |")
    lines.append(f"| pick_stage_counts | `{json.dumps(metrics.get('pick_stage_counts', {}), sort_keys=True)}` |")

    lines.extend(
        [
            "",
            "## Ambiguity Conclusion",
            "",
            _ambiguity_conclusion(metrics),
        ]
    )

    lines.extend(
        [
            "",
            "## Per-Query Breakdown",
            "",
            "| Query | total_runs | mean_raw_num_detections | mean_num_detections | mean_num_ranked_candidates | mean_num_3d_points | fraction_with_3d_target | pick_success_rate | fraction_top1_changed_by_rerank | mean_runtime_seconds |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for query, query_metrics in per_query_metrics.items():
        lines.append(
            f"| {_escape_table_cell(query)} | "
            f"{_format_number(query_metrics.get('total_runs'))} | "
            f"{_format_number(query_metrics.get('mean_raw_num_detections'))} | "
            f"{_format_number(query_metrics.get('mean_num_detections'))} | "
            f"{_format_number(query_metrics.get('mean_num_ranked_candidates'))} | "
            f"{_format_number(query_metrics.get('mean_num_3d_points'))} | "
            f"{_format_number(query_metrics.get('fraction_with_3d_target'))} | "
            f"{_format_number(query_metrics.get('pick_success_rate'))} | "
            f"{_format_number(query_metrics.get('fraction_top1_changed_by_rerank'))} | "
            f"{_format_number(query_metrics.get('mean_runtime_seconds'))} |"
        )

    if compare_benchmark_dir is not None and secondary_summary is not None and comparison_metrics is not None:
        lines.extend(
            [
                "",
                "## Comparison",
                "",
                f"- Compare benchmark directory: `{compare_benchmark_dir}`",
                "",
                "| Metric | Primary | Secondary | Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for key in METRIC_KEYS:
            values = comparison_metrics[key]
            lines.append(
                f"| {key} | {_format_number(values['primary'])} | "
                f"{_format_number(values['secondary'])} | {_format_number(values['delta'])} |"
            )

    lines.append("")
    return "\n".join(lines)


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required benchmark artifact not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("aggregate_metrics")
    if isinstance(metrics, dict):
        return metrics
    return {}


def _per_query_metrics(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics = summary.get("per_query_metrics")
    if isinstance(metrics, dict):
        return {
            str(query): query_metrics if isinstance(query_metrics, dict) else {}
            for query, query_metrics in sorted(metrics.items())
        }
    return aggregate_runs_by_query(rows)


def _format_queries(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) if value else "unknown"
    return _value(value)


def _value(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)


def _format_number(value: Any) -> str:
    number = _as_float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:.4f}"


def _as_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _ambiguity_conclusion(metrics: dict[str, Any]) -> str:
    mean_raw = _as_float(metrics.get("mean_raw_num_detections"))
    top1_changed = _as_float(metrics.get("fraction_top1_changed_by_rerank"))
    if mean_raw <= 1.2 and top1_changed == 0.0:
        return "Current ambiguity benchmark still does not provide useful reranking headroom."
    if mean_raw > 1.0 and top1_changed > 0.0:
        return "Reranking has measurable opportunity in this benchmark setting."
    return "Current ambiguity benchmark provides limited or inconclusive reranking headroom."


def _escape_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
