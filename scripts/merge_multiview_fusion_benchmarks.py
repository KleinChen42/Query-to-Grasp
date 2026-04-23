"""Merge multiple multi-view fusion benchmark directories into one summary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multiview_fusion_benchmark import (  # noqa: E402
    aggregate_rows,
    aggregate_rows_by_query,
    write_rows_csv,
)
from src.io.export_utils import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multi-view fusion benchmark rows and summaries.")
    parser.add_argument(
        "--benchmark-dir",
        action="append",
        default=[],
        help="Input benchmark directory containing benchmark_rows.json and benchmark_summary.json. Repeatable.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Merged benchmark output directory.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing input directories instead of failing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        merged = merge_benchmark_dirs(
            benchmark_dirs=[Path(path) for path in args.benchmark_dir],
            output_dir=args.output_dir,
            skip_missing=args.skip_missing,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    metrics = merged["aggregate_metrics"]
    print("Merged multi-view fusion benchmarks")
    print(f"  Runs:          {merged['total_runs']}")
    print(f"  Sources:       {len(merged['source_benchmark_dirs'])}")
    print(f"  Selected frac: {metrics['fraction_with_selected_object']:.3f}")
    print(f"  Objects/run:   {metrics['mean_num_memory_objects']:.3f}")
    print(f"  Confidence:    {metrics['mean_selected_overall_confidence']:.3f}")
    print(f"  Reobserve:     {metrics['reobserve_trigger_rate']:.3f}")
    print(f"  Artifacts:     {args.output_dir}")
    return 0


def merge_benchmark_dirs(
    benchmark_dirs: list[Path],
    output_dir: Path,
    skip_missing: bool = False,
) -> dict[str, Any]:
    """Merge rows and summary metadata from benchmark directories."""

    if not benchmark_dirs:
        raise ValueError("Provide at least one --benchmark-dir.")

    loaded_inputs: list[dict[str, Any]] = []
    missing: list[Path] = []
    for benchmark_dir in benchmark_dirs:
        rows_path = benchmark_dir / "benchmark_rows.json"
        summary_path = benchmark_dir / "benchmark_summary.json"
        if not rows_path.exists() or not summary_path.exists():
            missing.append(benchmark_dir)
            continue
        loaded_inputs.append(
            {
                "benchmark_dir": benchmark_dir,
                "rows": load_json_list(rows_path),
                "summary": load_json_dict(summary_path),
            }
        )

    if missing and not skip_missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing benchmark rows or summaries:\n{missing_text}")
    if not loaded_inputs:
        raise ValueError("No benchmark inputs were available to merge.")

    rows: list[dict[str, Any]] = []
    for item in loaded_inputs:
        source_dir = str(item["benchmark_dir"])
        for row in item["rows"]:
            merged_row = dict(row)
            merged_row["source_benchmark_dir"] = source_dir
            rows.append(merged_row)

    summaries = [item["summary"] for item in loaded_inputs]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(rows, output_dir / "benchmark_rows.json")
    write_rows_csv(rows, output_dir / "benchmark_rows.csv")
    merged_summary = build_merged_summary(rows=rows, summaries=summaries, source_dirs=[item["benchmark_dir"] for item in loaded_inputs])
    write_json(merged_summary, output_dir / "benchmark_summary.json")
    return merged_summary


def build_merged_summary(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    source_dirs: list[Path],
) -> dict[str, Any]:
    """Build a benchmark_summary-compatible merged summary."""

    queries = sorted({str(row.get("query") or "") for row in rows if str(row.get("query") or "")})
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "merged_from": [str(path) for path in source_dirs],
        "source_benchmark_dirs": [str(path) for path in source_dirs],
        "total_runs": len(rows),
        "unique_queries": queries,
        "view_ids": _consistent_value(summaries, "view_ids", default=[]),
        "camera_name": _consistent_value(summaries, "camera_name"),
        "view_preset": _consistent_value(summaries, "view_preset"),
        "detector_backend": _consistent_value(summaries, "detector_backend"),
        "skip_clip": _consistent_value(summaries, "skip_clip"),
        "depth_scale": _consistent_value(summaries, "depth_scale"),
        "merge_distance": _consistent_value(summaries, "merge_distance"),
        "aggregate_metrics": aggregate_rows(rows),
        "per_query_metrics": aggregate_rows_by_query(rows),
    }


def load_json_dict(path: Path) -> dict[str, Any]:
    """Load a JSON object."""

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def load_json_list(path: Path) -> list[dict[str, Any]]:
    """Load a JSON list of objects."""

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}, got {type(data).__name__}.")
    return [item for item in data if isinstance(item, dict)]


def _consistent_value(summaries: list[dict[str, Any]], key: str, default: Any = "mixed") -> Any:
    values = [summary.get(key) for summary in summaries if key in summary]
    if not values:
        return default
    first = values[0]
    return first if all(value == first for value in values) else "mixed"


if __name__ == "__main__":
    raise SystemExit(main())
