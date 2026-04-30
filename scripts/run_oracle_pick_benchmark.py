"""Benchmark simulated picks from privileged oracle object poses."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_oracle_pick_smoke import find_oracle_target_xyz  # noqa: E402
from src.env.maniskill_env import ManiSkillScene  # noqa: E402
from src.eval.metrics import aggregate_runs  # noqa: E402
from src.io.export_utils import write_json  # noqa: E402


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
    "grasp_attempted",
    "pick_success",
    "task_success",
    "is_grasped",
    "pick_stage",
    "pick_target_xyz",
    "pick_target_source",
    "runtime_seconds",
    "artifacts",
    "run_failed",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run oracle-target simulated pick benchmarks.")
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", default="rgbd")
    parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Integer seeds. Defaults to range(num-runs).")
    parser.add_argument("--num-runs", type=int, default=1, help="Fallback number of seeds when --seeds is omitted.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "oracle_pick_benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds else list(range(args.num_runs))
    if args.num_runs < 1 and not args.seeds:
        raise ValueError("--num-runs must be at least 1 when --seeds is not supplied.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    scene = ManiSkillScene(env_name=args.env_id, obs_mode=args.obs_mode, control_mode=args.control_mode)
    try:
        for run_index, seed in enumerate(seeds, start=1):
            rows.append(run_one_seed(scene=scene, args=args, seed=seed, run_index=run_index, runs_dir=runs_dir))
    finally:
        scene.close()

    aggregate_metrics = aggregate_oracle_rows(rows)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_runs": len(rows),
        "unique_queries": ["oracle_object_pose"],
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "detector_backend": "oracle",
        "skip_clip": True,
        "pick_executor": "sim_topdown",
        "grasp_target_mode": "oracle",
        "pick_target_source": "oracle_object_pose",
        "aggregate_metrics": aggregate_metrics,
        "per_query_metrics": {"oracle_object_pose": aggregate_metrics},
    }
    write_json(rows, args.output_dir / "benchmark_rows.json")
    write_rows_csv(rows, args.output_dir / "benchmark_rows.csv")
    write_json(summary, args.output_dir / "benchmark_summary.json")
    print_summary(summary, args.output_dir)


def run_one_seed(
    scene: ManiSkillScene,
    args: argparse.Namespace,
    seed: int,
    run_index: int,
    runs_dir: Path,
) -> dict[str, Any]:
    """Reset one seed, pick from the oracle object pose, and return a benchmark row."""

    run_dir = runs_dir / f"run_{run_index:04d}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    try:
        scene.reset(seed=seed)
        target_xyz = find_oracle_target_xyz(scene.env)
        if target_xyz is None:
            raise RuntimeError("could not discover oracle object pose")
        target_xyz = np.asarray(target_xyz, dtype=np.float32).reshape(3)
        result = scene.execute_pick(target_xyz, executor="sim_topdown")
        runtime_seconds = time.perf_counter() - start_time
        summary = build_child_summary(
            args=args,
            seed=seed,
            run_dir=run_dir,
            target_xyz=target_xyz,
            pick_result=result,
            runtime_seconds=runtime_seconds,
        )
        write_json(result, run_dir / "pick_result.json")
        write_json(summary, run_dir / "summary.json")
        return row_from_summary(summary)
    except Exception as exc:
        runtime_seconds = time.perf_counter() - start_time
        row = failed_row(seed=seed, message=str(exc), artifacts=run_dir, runtime_seconds=runtime_seconds)
        write_json(row, run_dir / "summary.json")
        return row


def build_child_summary(
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
    target_xyz: np.ndarray,
    pick_result: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    """Return a per-run summary matching the benchmark row contract."""

    return {
        "query": "oracle_object_pose",
        "seed": int(seed),
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "detector_backend": "oracle",
        "skip_clip": True,
        "pick_executor": "sim_topdown",
        "grasp_target_mode": "oracle",
        "target_xyz": target_xyz.astype(float).tolist(),
        "pick_target_xyz": target_xyz.astype(float).tolist(),
        "pick_target_source": "oracle_object_pose",
        "raw_num_detections": 0,
        "num_detections": 0,
        "num_ranked_candidates": 0,
        "top1_changed_by_rerank": False,
        "detector_top_phrase": None,
        "final_top_phrase": None,
        "has_3d_target": True,
        "num_3d_points": 0,
        "grasp_attempted": bool(pick_result.get("grasp_attempted", False)),
        "pick_success": bool(pick_result.get("pick_success", pick_result.get("success", False))),
        "task_success": pick_result.get("task_success"),
        "is_grasped": pick_result.get("is_grasped"),
        "pick_stage": pick_result.get("stage", "unknown"),
        "runtime_seconds": float(runtime_seconds),
        "artifacts": str(run_dir),
        "run_failed": False,
        "error_message": "",
    }


def row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten a child summary into the oracle benchmark CSV/JSON row."""

    return {column: summary.get(column) for column in CSV_COLUMNS}


def failed_row(seed: int, message: str, artifacts: Path, runtime_seconds: float = 0.0) -> dict[str, Any]:
    """Return a benchmark row for a failed oracle child run."""

    return {
        "query": "oracle_object_pose",
        "seed": int(seed),
        "raw_num_detections": 0,
        "num_detections": 0,
        "num_ranked_candidates": 0,
        "top1_changed_by_rerank": False,
        "detector_top_phrase": None,
        "final_top_phrase": None,
        "has_3d_target": False,
        "num_3d_points": 0,
        "grasp_attempted": False,
        "pick_success": False,
        "task_success": False,
        "is_grasped": False,
        "pick_stage": "run_failed",
        "pick_target_xyz": None,
        "pick_target_source": "oracle_object_pose",
        "runtime_seconds": float(runtime_seconds),
        "artifacts": str(artifacts),
        "run_failed": True,
        "error_message": message,
    }


def aggregate_oracle_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate oracle rows with the standard grasp metrics plus failure count."""

    metrics = aggregate_runs(rows)
    metrics["failed_runs"] = sum(1 for row in rows if _as_bool(row.get("run_failed")))
    metrics["pick_target_source_counts"] = _count_values(row.get("pick_target_source") for row in rows)
    return metrics


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write one oracle benchmark CSV row per seed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    """Print a compact oracle benchmark summary."""

    metrics = summary["aggregate_metrics"]
    print("Oracle pick benchmark complete")
    print(f"  Env:         {summary['env_id']}")
    print(f"  Runs:        {summary['total_runs']}")
    print(f"  Failed:      {metrics['failed_runs']}")
    print(f"  Pick rate:   {metrics['pick_success_rate']:.3f}")
    print(f"  Task rate:   {metrics['task_success_rate']:.3f}")
    print(f"  Artifacts:   {output_dir}")


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


if __name__ == "__main__":
    main()
