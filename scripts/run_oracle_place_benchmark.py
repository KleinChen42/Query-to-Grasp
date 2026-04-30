"""Benchmark oracle StackCube pick-place attempts from privileged object poses."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.maniskill_env import ManiSkillScene  # noqa: E402
from src.eval.metrics import aggregate_runs  # noqa: E402
from src.io.export_utils import write_json  # noqa: E402
from src.manipulation.pick_executor import SimulatedPickPlaceExecutor  # noqa: E402


CSV_COLUMNS = [
    "benchmark_type",
    "query",
    "seed",
    "env_id",
    "obs_mode",
    "control_mode",
    "pick_executor",
    "grasp_target_mode",
    "detector_backend",
    "pick_target_source",
    "place_target_source",
    "pick_xyz",
    "place_xyz",
    "grasp_attempted",
    "place_attempted",
    "pick_success",
    "place_success",
    "task_success",
    "is_grasped",
    "pick_stage",
    "stage",
    "runtime_seconds",
    "artifacts",
    "run_failed",
    "error_message",
]


@dataclass(frozen=True)
class OraclePlaceTargets:
    """Privileged StackCube pick/place targets."""

    pick_xyz: np.ndarray
    place_xyz: np.ndarray
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run oracle StackCube simulated pick-place benchmarks.")
    parser.add_argument("--env-id", default="StackCube-v1")
    parser.add_argument("--obs-mode", default="rgbd")
    parser.add_argument("--control-mode", default="pd_ee_delta_pos")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Integer seeds. Defaults to range(num-runs).")
    parser.add_argument("--num-runs", type=int, default=1, help="Fallback number of seeds when --seeds is omitted.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "oracle_place_benchmark")
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

    aggregate_metrics = aggregate_oracle_place_rows(rows)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": "oracle_place",
        "total_runs": len(rows),
        "unique_queries": ["oracle_stackcube_place"],
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "detector_backend": "oracle",
        "skip_clip": True,
        "pick_executor": "sim_pick_place",
        "grasp_target_mode": "oracle_place",
        "pick_target_source": "oracle_cubeA_pose",
        "place_target_source": "oracle_cubeB_pose",
        "aggregate_metrics": aggregate_metrics,
        "per_query_metrics": {"oracle_stackcube_place": aggregate_metrics},
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
    """Reset one seed, use privileged StackCube poses, and return a row."""

    run_dir = runs_dir / f"run_{run_index:04d}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    try:
        scene.reset(seed=seed)
        targets = find_stackcube_oracle_place_targets(scene.env)
        result = SimulatedPickPlaceExecutor(env=scene.env).execute(
            pick_xyz=targets.pick_xyz,
            place_xyz=targets.place_xyz,
        )
        runtime_seconds = time.perf_counter() - start_time
        summary = build_child_summary(
            args=args,
            seed=seed,
            run_dir=run_dir,
            targets=targets,
            place_result=result,
            runtime_seconds=runtime_seconds,
        )
        write_json(result, run_dir / "place_result.json")
        write_json(summary, run_dir / "summary.json")
        return row_from_summary(summary)
    except Exception as exc:
        runtime_seconds = time.perf_counter() - start_time
        row = failed_row(args=args, seed=seed, message=str(exc), artifacts=run_dir, runtime_seconds=runtime_seconds)
        write_json(row, run_dir / "summary.json")
        return row


def find_stackcube_oracle_place_targets(env: Any) -> OraclePlaceTargets:
    """Return privileged cubeA pick and cubeB place targets for StackCube."""

    unwrapped = getattr(env, "unwrapped", env)
    cube_a = _find_named_actor(unwrapped, ["cubeA", "cube_a", "cubeA_actor", "cube_a_actor"])
    cube_b = _find_named_actor(unwrapped, ["cubeB", "cube_b", "cubeB_actor", "cube_b_actor"])
    cube_a_xyz = _pose_xyz(cube_a)
    cube_b_xyz = _pose_xyz(cube_b)
    if cube_a_xyz is None or cube_b_xyz is None:
        raise RuntimeError(
            "Could not discover StackCube oracle cubeA/cubeB poses. "
            f"Available object-like attributes: {_object_attribute_dump(unwrapped)}"
        )
    return OraclePlaceTargets(
        pick_xyz=cube_a_xyz,
        place_xyz=cube_b_xyz,
        metadata={
            "cubeA_attribute_type": type(cube_a).__name__,
            "cubeB_attribute_type": type(cube_b).__name__,
            "cube_half_size": _json_safe_value(getattr(unwrapped, "cube_half_size", None)),
        },
    )


def build_child_summary(
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
    targets: OraclePlaceTargets,
    place_result: dict[str, Any],
    runtime_seconds: float,
) -> dict[str, Any]:
    """Return a per-run summary matching the oracle-place row contract."""

    return {
        "benchmark_type": "oracle_place",
        "query": "oracle_stackcube_place",
        "seed": int(seed),
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "pick_executor": "sim_pick_place",
        "grasp_target_mode": "oracle_place",
        "detector_backend": "oracle",
        "skip_clip": True,
        "pick_target_source": "oracle_cubeA_pose",
        "place_target_source": "oracle_cubeB_pose",
        "pick_xyz": targets.pick_xyz.astype(float).tolist(),
        "place_xyz": targets.place_xyz.astype(float).tolist(),
        "oracle_metadata": targets.metadata,
        "grasp_attempted": bool(place_result.get("grasp_attempted", False)),
        "place_attempted": bool(place_result.get("place_attempted", False)),
        "pick_success": bool(place_result.get("pick_success", place_result.get("success", False))),
        "place_success": bool(place_result.get("place_success", False)),
        "task_success": bool(place_result.get("task_success", False)),
        "is_grasped": bool(place_result.get("is_grasped", False)),
        "pick_stage": place_result.get("stage", "unknown"),
        "stage": place_result.get("stage", "unknown"),
        "runtime_seconds": float(runtime_seconds),
        "artifacts": str(run_dir),
        "run_failed": False,
        "error_message": "",
    }


def row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Flatten a child summary into the oracle placement benchmark row."""

    return {column: summary.get(column) for column in CSV_COLUMNS}


def failed_row(
    args: argparse.Namespace,
    seed: int,
    message: str,
    artifacts: Path,
    runtime_seconds: float = 0.0,
) -> dict[str, Any]:
    """Return a benchmark row for a failed oracle placement run."""

    return {
        "benchmark_type": "oracle_place",
        "query": "oracle_stackcube_place",
        "seed": int(seed),
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "pick_executor": "sim_pick_place",
        "grasp_target_mode": "oracle_place",
        "detector_backend": "oracle",
        "pick_target_source": "oracle_cubeA_pose",
        "place_target_source": "oracle_cubeB_pose",
        "pick_xyz": None,
        "place_xyz": None,
        "grasp_attempted": False,
        "place_attempted": False,
        "pick_success": False,
        "place_success": False,
        "task_success": False,
        "is_grasped": False,
        "pick_stage": "run_failed",
        "stage": "run_failed",
        "runtime_seconds": float(runtime_seconds),
        "artifacts": str(artifacts),
        "run_failed": True,
        "error_message": message,
    }


def aggregate_oracle_place_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate oracle placement rows with pick, place, and failure metrics."""

    metrics = aggregate_runs(rows)
    total_runs = len(rows)
    metrics["failed_runs"] = sum(1 for row in rows if _as_bool(row.get("run_failed")))
    metrics["place_attempted_rate"] = _mean(1 if _as_bool(row.get("place_attempted")) else 0 for row in rows)
    metrics["place_success_rate"] = _mean(1 if _as_bool(row.get("place_success")) else 0 for row in rows)
    metrics["place_stage_counts"] = _count_values(row.get("stage") for row in rows)
    metrics["pick_target_source_counts"] = _count_values(row.get("pick_target_source") for row in rows)
    metrics["place_target_source_counts"] = _count_values(row.get("place_target_source") for row in rows)
    metrics["total_runs"] = total_runs
    return metrics


def write_rows_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write one oracle placement benchmark CSV row per seed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    """Print a compact oracle placement benchmark summary."""

    metrics = summary["aggregate_metrics"]
    print("Oracle place benchmark complete")
    print(f"  Env:          {summary['env_id']}")
    print(f"  Runs:         {summary['total_runs']}")
    print(f"  Failed:       {metrics['failed_runs']}")
    print(f"  Pick rate:    {metrics['pick_success_rate']:.3f}")
    print(f"  Place rate:   {metrics['place_success_rate']:.3f}")
    print(f"  Task rate:    {metrics['task_success_rate']:.3f}")
    print(f"  Artifacts:    {output_dir}")


def _find_named_actor(unwrapped: Any, names: list[str]) -> Any:
    for name in names:
        value = getattr(unwrapped, name, None)
        if value is not None:
            return value
    return None


def _pose_xyz(value: Any) -> np.ndarray | None:
    pose = getattr(value, "pose", None)
    position = getattr(pose, "p", None) if pose is not None else None
    if position is None:
        return None
    try:
        if hasattr(position, "detach"):
            position = position.detach()
        if hasattr(position, "cpu"):
            position = position.cpu()
        if hasattr(position, "numpy"):
            position = position.numpy()
        array = np.asarray(position, dtype=np.float32).reshape(-1, 3)
    except Exception:
        return None
    if array.size == 0 or not np.all(np.isfinite(array[0])):
        return None
    return array[0]


def _object_attribute_dump(unwrapped: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in dir(unwrapped):
        if not any(token in name.lower() for token in ("cube", "obj", "goal")):
            continue
        try:
            value = getattr(unwrapped, name)
        except Exception as exc:
            rows.append({"name": name, "type": "error", "value": str(exc)})
            continue
        rows.append({"name": name, "type": type(value).__name__, "has_pose": str(_pose_xyz(value) is not None)})
    return rows


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _mean(values: Any) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _json_safe_value(value: Any) -> Any:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value)
        if array.shape == ():
            return array.item()
        return array.tolist()
    except Exception:
        return None if value is None else str(value)


if __name__ == "__main__":
    main()
