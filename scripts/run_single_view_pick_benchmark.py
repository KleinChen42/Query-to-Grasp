"""Batch benchmark wrapper for the single-view placeholder-pick pipeline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import logging
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import aggregate_runs, aggregate_runs_by_query, load_pick_result, load_summary, summarize_run  # noqa: E402
from src.io.export_utils import write_json  # noqa: E402

LOGGER = logging.getLogger(__name__)

CSV_COLUMNS = [
    "query",
    "seed",
    "raw_num_detections",
    "num_detections",
    "num_ranked_candidates",
    "top1_changed_by_rerank",
    "detector_top_phrase",
    "final_top_phrase",
    "has_3d_target",
    "num_3d_points",
    "pick_success",
    "pick_stage",
    "runtime_seconds",
    "artifacts",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal batch benchmark for single-view placeholder pick.")
    parser.add_argument("--queries-file", type=Path, default=None, help="Text file with one query per line.")
    parser.add_argument("--queries", nargs="*", default=[], help="Inline queries.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Integer seeds. Defaults to range(num-runs).")
    parser.add_argument("--num-runs", type=int, default=1, help="Fallback number of seeds when --seeds is omitted.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "benchmark_mock_pick")
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
            row = run_one_child(args=args, query=query, seed=seed, run_index=run_index, child_runs_dir=child_runs_dir)
            rows.append(row)

    write_json(rows, args.output_dir / "benchmark_rows.json")
    write_rows_csv(rows, args.output_dir / "benchmark_rows.csv")
    benchmark_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(rows),
        "unique_queries": sorted(set(query for query in queries)),
        "detector_backend": args.detector_backend,
        "skip_clip": bool(args.skip_clip),
        "depth_scale": float(args.depth_scale),
        "aggregate_metrics": aggregate_runs(rows),
        "per_query_metrics": aggregate_runs_by_query(rows),
    }
    write_json(benchmark_summary, args.output_dir / "benchmark_summary.json")
    print_benchmark_summary(benchmark_summary, args.output_dir)


def load_queries(queries_file: Path | None, inline_queries: list[str]) -> list[str]:
    """Load queries from a text file and inline CLI values."""

    queries: list[str] = []
    if queries_file is not None:
        with queries_file.open("r", encoding="utf-8") as file:
            queries.extend(_clean_query_line(line) for line in file if _clean_query_line(line) is not None)
    queries.extend(query.strip() for query in inline_queries if query.strip())
    if not queries:
        raise ValueError("Provide at least one query via --queries-file or --queries.")
    return queries


def _clean_query_line(line: str) -> str | None:
    """Return a query line, skipping blanks and whole-line comments."""

    query = line.strip()
    if not query or query.startswith("#"):
        return None
    return query


def run_one_child(
    args: argparse.Namespace,
    query: str,
    seed: int,
    run_index: int,
    child_runs_dir: Path,
) -> dict[str, Any]:
    """Run one child pipeline process and return a flat benchmark row."""

    invocation_root = child_runs_dir / f"run_{run_index:04d}_seed_{seed}"
    invocation_root.mkdir(parents=True, exist_ok=True)
    command = build_child_command(args=args, query=query, seed=seed, output_dir=invocation_root)
    LOGGER.info("Running benchmark child %s: query=%r seed=%s", run_index, query, seed)
    start_time = time.time()
    try:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    except Exception as exc:
        return failed_row(query=query, seed=seed, message=f"subprocess failed to start: {exc}", artifacts=str(invocation_root))

    if completed.returncode != 0:
        LOGGER.warning("Child run failed with return code %s for query=%r seed=%s", completed.returncode, query, seed)
        return failed_row(
            query=query,
            seed=seed,
            message=f"subprocess returned {completed.returncode}",
            artifacts=str(invocation_root),
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
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    try:
        summary = load_summary(run_dir / "summary.json")
        pick_result = load_pick_result(run_dir / "pick_result.json")
        row = summarize_run(summary, pick_result)
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
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def build_child_command(args: argparse.Namespace, query: str, seed: int, output_dir: Path) -> list[str]:
    """Build the subprocess command for one pick-pipeline run."""

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_single_view_pick.py"),
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
        "--output-dir",
        str(output_dir),
    ]
    if args.skip_clip:
        command.append("--skip-clip")
    return command


def find_newest_run_dir(output_dir: Path, start_time: float) -> Path | None:
    """Find the newest child run directory that contains ``summary.json``."""

    if not output_dir.exists():
        return None
    candidates = [
        path
        for path in output_dir.iterdir()
        if path.is_dir() and (path / "summary.json").exists() and path.stat().st_mtime >= start_time - 1.0
    ]
    if not candidates:
        candidates = [path for path in output_dir.iterdir() if path.is_dir() and (path / "summary.json").exists()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def failed_row(
    query: str,
    seed: int,
    message: str,
    artifacts: str,
    stdout: str | None = None,
    stderr: str | None = None,
) -> dict[str, Any]:
    """Build a row for a failed child invocation."""

    return {
        "query": query,
        "seed": int(seed),
        "raw_num_detections": 0,
        "num_detections": 0,
        "num_ranked_candidates": 0,
        "top1_changed_by_rerank": False,
        "detector_top_phrase": None,
        "final_top_phrase": None,
        "has_3d_target": False,
        "num_3d_points": 0,
        "pick_success": False,
        "pick_stage": "run_failed",
        "runtime_seconds": 0.0,
        "artifacts": artifacts,
        "run_failed": True,
        "error_message": message,
        "stdout": stdout or "",
        "stderr": stderr or "",
    }


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
    print("Single-view pick benchmark complete")
    print(f"  Runs:        {benchmark_summary['total_runs']}")
    print(f"  Queries:     {', '.join(benchmark_summary['unique_queries'])}")
    print(f"  3D target:   {metrics['fraction_with_3d_target']:.3f}")
    print(f"  Pick rate:   {metrics['pick_success_rate']:.3f}")
    print(f"  Runtime:     {metrics.get('mean_runtime_seconds', 0.0):.3f}s")
    print(f"  Artifacts:   {output_dir}")


if __name__ == "__main__":
    main()
