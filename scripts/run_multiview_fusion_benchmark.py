"""Batch benchmark wrapper for multi-view fusion debug runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_single_view_pick_benchmark import find_newest_run_dir, load_queries  # noqa: E402
from src.io.export_utils import write_json  # noqa: E402

LOGGER = logging.getLogger(__name__)

CSV_COLUMNS = [
    "query",
    "seed",
    "num_views",
    "num_memory_objects",
    "num_observations_added",
    "has_selected_object",
    "selected_object_id",
    "selected_top_label",
    "selected_overall_confidence",
    "selection_label",
    "should_reobserve",
    "reobserve_reason",
    "initial_should_reobserve",
    "initial_reobserve_reason",
    "final_should_reobserve",
    "final_reobserve_reason",
    "closed_loop_reobserve_enabled",
    "closed_loop_reobserve_executed",
    "closed_loop_reobserve_view_ids",
    "closed_loop_delta_num_views",
    "closed_loop_delta_num_memory_objects",
    "closed_loop_delta_num_observations_added",
    "closed_loop_delta_selected_overall_confidence",
    "closed_loop_delta_selected_num_views",
    "closed_loop_delta_selected_num_observations",
    "closed_loop_selected_object_changed",
    "closed_loop_reobserve_reason_changed",
    "closed_loop_reobserve_resolved",
    "closed_loop_reobserve_still_needed",
    "runtime_seconds",
    "detector_backend",
    "skip_clip",
    "view_preset",
    "camera_name",
    "artifacts",
    "run_failed",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a batch benchmark for multi-view fusion debug outputs.")
    parser.add_argument("--queries-file", type=Path, default=None, help="Text file with one query per line.")
    parser.add_argument("--queries", nargs="*", default=[], help="Inline queries.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Integer seeds. Defaults to range(num-runs).")
    parser.add_argument("--num-runs", type=int, default=1, help="Fallback number of seeds when --seeds is omitted.")
    parser.add_argument("--view-ids", nargs="*", default=[], help="Optional camera keys forwarded to the debug runner.")
    parser.add_argument("--view-preset", default="none", help="Optional virtual camera pose preset forwarded to the debug runner.")
    parser.add_argument("--camera-name", default=None, help="Fallback camera key when --view-ids is omitted.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "multiview_fusion_benchmark")
    parser.add_argument(
        "--detector-backend",
        default="mock",
        choices=["auto", "hf", "transformers", "groundingdino", "original", "mock"],
    )
    parser.add_argument("--mock-box-position", default="center", choices=["center", "left", "right", "all"])
    parser.add_argument("--skip-clip", dest="skip_clip", action="store_true", default=False)
    parser.add_argument("--use-clip", dest="skip_clip", action="store_false", help="Run CLIP reranking. This is the default unless --skip-clip is set.")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", default="rgbd")
    parser.add_argument("--merge-distance", type=float, default=0.08)
    parser.add_argument("--enable-closed-loop-reobserve", action="store_true")
    parser.add_argument("--closed-loop-max-extra-views", type=int, default=1)
    parser.add_argument("--fail-on-child-error", action="store_true", help="Exit nonzero if any child debug run fails.")
    parser.add_argument("--log-level", default="INFO", help="Benchmark logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")

    queries = load_queries(args.queries_file, args.queries)
    seeds = args.seeds if args.seeds else list(range(args.num_runs))
    if args.num_runs < 1 and not args.seeds:
        raise ValueError("--num-runs must be at least 1 when --seeds is not supplied.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    child_runs_dir = args.output_dir / "runs"
    child_runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    run_index = 0
    for query in queries:
        for seed in seeds:
            run_index += 1
            rows.append(
                run_one_child(
                    args=args,
                    query=query,
                    seed=seed,
                    run_index=run_index,
                    child_runs_dir=child_runs_dir,
                )
            )

    write_json(rows, args.output_dir / "benchmark_rows.json")
    write_rows_csv(rows, args.output_dir / "benchmark_rows.csv")
    benchmark_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(rows),
        "unique_queries": sorted(set(queries)),
        "view_ids": [view_id for view_id in args.view_ids if view_id],
        "camera_name": args.camera_name,
        "view_preset": args.view_preset,
        "detector_backend": args.detector_backend,
        "skip_clip": bool(args.skip_clip),
        "depth_scale": float(args.depth_scale),
        "merge_distance": float(args.merge_distance),
        "closed_loop_reobserve_enabled": bool(args.enable_closed_loop_reobserve),
        "closed_loop_max_extra_views": int(args.closed_loop_max_extra_views),
        "aggregate_metrics": aggregate_rows(rows),
        "per_query_metrics": aggregate_rows_by_query(rows),
    }
    write_json(benchmark_summary, args.output_dir / "benchmark_summary.json")
    print_benchmark_summary(benchmark_summary, args.output_dir)
    if args.fail_on_child_error and benchmark_summary["aggregate_metrics"]["failed_runs"]:
        raise SystemExit(1)


def run_one_child(
    args: argparse.Namespace,
    query: str,
    seed: int,
    run_index: int,
    child_runs_dir: Path,
) -> dict[str, Any]:
    """Run one debug child process and return a flat benchmark row."""

    invocation_root = child_runs_dir / f"run_{run_index:04d}_seed_{seed}"
    invocation_root.mkdir(parents=True, exist_ok=True)
    command = build_child_command(args=args, query=query, seed=seed, output_dir=invocation_root)
    LOGGER.info("Running fusion child %s: query=%r seed=%s", run_index, query, seed)
    start_time = time.time()
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    except Exception as exc:
        return failed_row(
            query=query,
            seed=seed,
            message=f"subprocess failed to start: {exc}",
            artifacts=str(invocation_root),
            args=args,
        )

    if completed.returncode != 0:
        LOGGER.warning("Fusion child failed with return code %s for query=%r seed=%s", completed.returncode, query, seed)
        return failed_row(
            query=query,
            seed=seed,
            message=f"subprocess returned {completed.returncode}",
            artifacts=str(invocation_root),
            args=args,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    run_dir = find_newest_run_dir(invocation_root, start_time=start_time)
    if run_dir is None:
        return failed_row(
            query=query,
            seed=seed,
            message="child completed but no run directory with summary.json was found",
            artifacts=str(invocation_root),
            args=args,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    try:
        summary = load_json_dict(run_dir / "summary.json")
        row = summarize_fusion_run(summary)
        row["query"] = row.get("query") or query
        row["seed"] = int(seed)
        row["run_failed"] = False
        return row
    except Exception as exc:
        return failed_row(
            query=query,
            seed=seed,
            message=f"failed to read child outputs: {exc}",
            artifacts=str(run_dir),
            args=args,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def build_child_command(args: argparse.Namespace, query: str, seed: int, output_dir: Path) -> list[str]:
    """Build the subprocess command for one multi-view fusion debug run."""

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_multiview_fusion_debug.py"),
        "--query",
        query,
        "--seed",
        str(seed),
        "--env-id",
        args.env_id,
        "--obs-mode",
        args.obs_mode,
        "--detector-backend",
        args.detector_backend,
        "--mock-box-position",
        args.mock_box_position,
        "--depth-scale",
        str(args.depth_scale),
        "--merge-distance",
        str(args.merge_distance),
        "--output-dir",
        str(output_dir),
    ]
    if args.camera_name:
        command.extend(["--camera-name", args.camera_name])
    if args.view_preset and args.view_preset != "none":
        command.extend(["--view-preset", args.view_preset])
    if args.view_ids:
        command.append("--view-ids")
        command.extend(str(view_id) for view_id in args.view_ids if str(view_id).strip())
    if args.skip_clip:
        command.append("--skip-clip")
    else:
        command.append("--use-clip")
    if args.enable_closed_loop_reobserve:
        command.append("--enable-closed-loop-reobserve")
        command.extend(["--closed-loop-max-extra-views", str(args.closed_loop_max_extra_views)])
    return command


def summarize_fusion_run(summary: dict[str, Any]) -> dict[str, Any]:
    """Convert one debug summary into a flat benchmark row."""

    selected_object_id = _optional_str(summary.get("selected_object_id"))
    return {
        "query": str(summary.get("query") or ""),
        "num_views": _as_int(summary.get("num_views"), 0),
        "num_memory_objects": _as_int(summary.get("num_memory_objects"), 0),
        "num_observations_added": _as_int(summary.get("num_observations_added"), 0),
        "has_selected_object": selected_object_id is not None,
        "selected_object_id": selected_object_id,
        "selected_top_label": _optional_str(summary.get("selected_top_label")),
        "selected_overall_confidence": _as_float(summary.get("selected_overall_confidence"), 0.0),
        "selection_label": _optional_str(summary.get("selection_label")),
        "should_reobserve": _as_bool(summary.get("should_reobserve")),
        "reobserve_reason": _optional_str(summary.get("reobserve_reason")),
        "initial_should_reobserve": _as_bool(summary.get("initial_should_reobserve", summary.get("should_reobserve"))),
        "initial_reobserve_reason": _optional_str(summary.get("initial_reobserve_reason", summary.get("reobserve_reason"))),
        "final_should_reobserve": _as_bool(summary.get("final_should_reobserve", summary.get("should_reobserve"))),
        "final_reobserve_reason": _optional_str(summary.get("final_reobserve_reason", summary.get("reobserve_reason"))),
        "closed_loop_reobserve_enabled": _as_bool(summary.get("closed_loop_reobserve_enabled")),
        "closed_loop_reobserve_executed": _as_bool(summary.get("closed_loop_reobserve_executed")),
        "closed_loop_reobserve_view_ids": " ".join(str(item) for item in summary.get("closed_loop_reobserve_view_ids", [])),
        "closed_loop_delta_num_views": _as_int(summary.get("closed_loop_delta_num_views"), 0),
        "closed_loop_delta_num_memory_objects": _as_int(summary.get("closed_loop_delta_num_memory_objects"), 0),
        "closed_loop_delta_num_observations_added": _as_int(
            summary.get("closed_loop_delta_num_observations_added"),
            0,
        ),
        "closed_loop_delta_selected_overall_confidence": _as_float(
            summary.get("closed_loop_delta_selected_overall_confidence"),
            0.0,
        ),
        "closed_loop_delta_selected_num_views": _as_int(summary.get("closed_loop_delta_selected_num_views"), 0),
        "closed_loop_delta_selected_num_observations": _as_int(
            summary.get("closed_loop_delta_selected_num_observations"),
            0,
        ),
        "closed_loop_selected_object_changed": _as_bool(summary.get("closed_loop_selected_object_changed")),
        "closed_loop_reobserve_reason_changed": _as_bool(summary.get("closed_loop_reobserve_reason_changed")),
        "closed_loop_reobserve_resolved": _as_bool(summary.get("closed_loop_reobserve_resolved")),
        "closed_loop_reobserve_still_needed": _as_bool(summary.get("closed_loop_reobserve_still_needed")),
        "runtime_seconds": _as_float(summary.get("runtime_seconds"), 0.0),
        "detector_backend": str(summary.get("detector_backend") or ""),
        "skip_clip": _as_bool(summary.get("skip_clip")),
        "view_preset": str(summary.get("view_preset") or "none"),
        "camera_name": _optional_str(summary.get("camera_name")),
        "artifacts": str(summary.get("artifacts") or ""),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multi-view fusion benchmark rows."""

    total_runs = len(rows)
    if total_runs == 0:
        return {
            "total_runs": 0,
            "failed_runs": 0,
            "fraction_run_failed": 0.0,
            "mean_num_views": 0.0,
            "mean_num_memory_objects": 0.0,
            "mean_num_observations_added": 0.0,
            "fraction_with_selected_object": 0.0,
            "reobserve_trigger_rate": 0.0,
            "initial_reobserve_trigger_rate": 0.0,
            "final_reobserve_trigger_rate": 0.0,
            "closed_loop_execution_rate": 0.0,
            "closed_loop_resolution_rate": 0.0,
            "closed_loop_still_needed_rate": 0.0,
            "closed_loop_selected_object_change_rate": 0.0,
            "closed_loop_reobserve_reason_change_rate": 0.0,
            "mean_closed_loop_delta_num_views": 0.0,
            "mean_closed_loop_delta_num_memory_objects": 0.0,
            "mean_closed_loop_delta_num_observations_added": 0.0,
            "mean_closed_loop_delta_selected_overall_confidence": 0.0,
            "mean_closed_loop_delta_selected_num_views": 0.0,
            "mean_closed_loop_delta_selected_num_observations": 0.0,
            "reobserve_reason_counts": {},
            "initial_reobserve_reason_counts": {},
            "final_reobserve_reason_counts": {},
            "mean_selected_overall_confidence": 0.0,
            "mean_runtime_seconds": 0.0,
        }

    failed_runs = sum(1 for row in rows if _as_bool(row.get("run_failed")))
    return {
        "total_runs": total_runs,
        "failed_runs": failed_runs,
        "fraction_run_failed": failed_runs / total_runs,
        "mean_num_views": _mean(_as_int(row.get("num_views"), 0) for row in rows),
        "mean_num_memory_objects": _mean(_as_int(row.get("num_memory_objects"), 0) for row in rows),
        "mean_num_observations_added": _mean(_as_int(row.get("num_observations_added"), 0) for row in rows),
        "fraction_with_selected_object": _mean(1 if _as_bool(row.get("has_selected_object")) else 0 for row in rows),
        "reobserve_trigger_rate": _mean(1 if _as_bool(row.get("should_reobserve")) else 0 for row in rows),
        "initial_reobserve_trigger_rate": _mean(1 if _as_bool(row.get("initial_should_reobserve", row.get("should_reobserve"))) else 0 for row in rows),
        "final_reobserve_trigger_rate": _mean(1 if _as_bool(row.get("final_should_reobserve", row.get("should_reobserve"))) else 0 for row in rows),
        "closed_loop_execution_rate": _mean(1 if _as_bool(row.get("closed_loop_reobserve_executed")) else 0 for row in rows),
        "closed_loop_resolution_rate": _mean(1 if _as_bool(row.get("closed_loop_reobserve_resolved")) else 0 for row in rows),
        "closed_loop_still_needed_rate": _mean(1 if _as_bool(row.get("closed_loop_reobserve_still_needed")) else 0 for row in rows),
        "closed_loop_selected_object_change_rate": _mean(
            1 if _as_bool(row.get("closed_loop_selected_object_changed")) else 0 for row in rows
        ),
        "closed_loop_reobserve_reason_change_rate": _mean(
            1 if _as_bool(row.get("closed_loop_reobserve_reason_changed")) else 0 for row in rows
        ),
        "mean_closed_loop_delta_num_views": _mean(
            _as_int(row.get("closed_loop_delta_num_views"), 0) for row in rows
        ),
        "mean_closed_loop_delta_num_memory_objects": _mean(
            _as_int(row.get("closed_loop_delta_num_memory_objects"), 0) for row in rows
        ),
        "mean_closed_loop_delta_num_observations_added": _mean(
            _as_int(row.get("closed_loop_delta_num_observations_added"), 0) for row in rows
        ),
        "mean_closed_loop_delta_selected_overall_confidence": _mean(
            _as_float(row.get("closed_loop_delta_selected_overall_confidence"), 0.0) for row in rows
        ),
        "mean_closed_loop_delta_selected_num_views": _mean(
            _as_int(row.get("closed_loop_delta_selected_num_views"), 0) for row in rows
        ),
        "mean_closed_loop_delta_selected_num_observations": _mean(
            _as_int(row.get("closed_loop_delta_selected_num_observations"), 0) for row in rows
        ),
        "reobserve_reason_counts": count_values(
            row.get("reobserve_reason") or "none"
            for row in rows
        ),
        "initial_reobserve_reason_counts": count_values(
            row.get("initial_reobserve_reason") or row.get("reobserve_reason") or "none"
            for row in rows
        ),
        "final_reobserve_reason_counts": count_values(
            row.get("final_reobserve_reason") or row.get("reobserve_reason") or "none"
            for row in rows
        ),
        "mean_selected_overall_confidence": _mean(_as_float(row.get("selected_overall_confidence"), 0.0) for row in rows),
        "mean_runtime_seconds": _mean(_as_float(row.get("runtime_seconds"), 0.0) for row in rows),
    }


def aggregate_rows_by_query(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate rows separately for each query."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("query") or ""), []).append(row)
    return {query: aggregate_rows(query_rows) for query, query_rows in sorted(grouped.items())}


def failed_row(
    query: str,
    seed: int,
    message: str,
    artifacts: str,
    args: argparse.Namespace | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> dict[str, Any]:
    """Build a row for a failed child invocation."""

    return {
        "query": query,
        "seed": int(seed),
        "num_views": 0,
        "num_memory_objects": 0,
        "num_observations_added": 0,
        "has_selected_object": False,
        "selected_object_id": None,
        "selected_top_label": None,
        "selected_overall_confidence": 0.0,
        "selection_label": None,
        "should_reobserve": False,
        "reobserve_reason": None,
        "initial_should_reobserve": False,
        "initial_reobserve_reason": None,
        "final_should_reobserve": False,
        "final_reobserve_reason": None,
        "closed_loop_reobserve_enabled": False if args is None else bool(getattr(args, "enable_closed_loop_reobserve", False)),
        "closed_loop_reobserve_executed": False,
        "closed_loop_reobserve_view_ids": "",
        "closed_loop_delta_num_views": 0,
        "closed_loop_delta_num_memory_objects": 0,
        "closed_loop_delta_num_observations_added": 0,
        "closed_loop_delta_selected_overall_confidence": 0.0,
        "closed_loop_delta_selected_num_views": 0,
        "closed_loop_delta_selected_num_observations": 0,
        "closed_loop_selected_object_changed": False,
        "closed_loop_reobserve_reason_changed": False,
        "closed_loop_reobserve_resolved": False,
        "closed_loop_reobserve_still_needed": False,
        "runtime_seconds": 0.0,
        "detector_backend": "" if args is None else str(args.detector_backend),
        "skip_clip": False if args is None else bool(args.skip_clip),
        "view_preset": "none" if args is None else str(args.view_preset or "none"),
        "camera_name": None if args is None else _optional_str(args.camera_name),
        "artifacts": artifacts,
        "run_failed": True,
        "error_message": message,
        "stdout": stdout or "",
        "stderr": stderr or "",
    }


def load_json_dict(path: str | Path) -> dict[str, Any]:
    """Load a JSON object."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write one benchmark CSV row per child run."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_benchmark_summary(benchmark_summary: dict[str, Any], output_dir: Path) -> None:
    """Print a short benchmark summary."""

    metrics = benchmark_summary["aggregate_metrics"]
    print("Multi-view fusion benchmark complete")
    print(f"  Runs:          {benchmark_summary['total_runs']}")
    print(f"  Queries:       {', '.join(benchmark_summary['unique_queries'])}")
    print(f"  Selected frac: {metrics['fraction_with_selected_object']:.3f}")
    print(f"  Objects/run:   {metrics['mean_num_memory_objects']:.3f}")
    print(f"  Confidence:    {metrics['mean_selected_overall_confidence']:.3f}")
    print(f"  Reobserve:     {metrics['reobserve_trigger_rate']:.3f}")
    print(f"  Runtime:       {metrics['mean_runtime_seconds']:.3f}s")
    print(f"  Artifacts:     {output_dir}")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _mean(values: Any) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def count_values(values: Any) -> dict[str, int]:
    """Count stringified values."""

    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "none")
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    main()
