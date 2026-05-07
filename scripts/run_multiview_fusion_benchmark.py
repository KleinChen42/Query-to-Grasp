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
    "selected_world_xyz",
    "selected_grasp_world_xyz",
    "selected_semantic_to_grasp_xy_distance",
    "selected_semantic_to_grasp_z_delta",
    "selected_grasp_observation_count",
    "selected_grasp_observation_xy_spread",
    "selected_grasp_observation_z_spread",
    "selected_grasp_observation_max_distance_to_fused",
    "selected_grasp_observation_history_json",
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
    "closed_loop_before_selected_present_after",
    "closed_loop_before_selected_still_selected",
    "closed_loop_before_selected_received_observation",
    "closed_loop_before_selected_gained_view_support",
    "closed_loop_before_selected_merged_extra_view_ids",
    "closed_loop_before_selected_delta_num_observations",
    "closed_loop_before_selected_delta_num_views",
    "closed_loop_extra_view_absorber_object_ids",
    "closed_loop_extra_view_absorber_count",
    "closed_loop_final_selected_absorbed_extra_view",
    "closed_loop_extra_view_third_object_ids",
    "closed_loop_extra_view_third_object_involved",
    "closed_loop_selected_object_continuity_enabled",
    "closed_loop_preferred_merge_count",
    "closed_loop_preferred_merge_rate",
    "closed_loop_post_selection_continuity_enabled",
    "closed_loop_post_selection_continuity_eligible",
    "closed_loop_post_selection_continuity_applied",
    "closed_loop_post_selection_continuity_reason",
    "grasp_attempted",
    "pick_success",
    "task_success",
    "is_grasped",
    "pick_stage",
    "pick_target_xyz",
    "pick_target_source",
    "place_attempted",
    "place_success",
    "place_target_xyz",
    "place_target_source",
    "place_query",
    "place_selection_reason",
    "place_pick_xy_distance",
    "task_grasp_target_guard_applied",
    "task_grasp_target_guard_reason",
    "execution_video_path",
    "execution_video_status",
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
    parser.add_argument("--start-seed", type=int, default=None, help="Convenience: generate seeds from start-seed to start-seed+num-seeds-1.")
    parser.add_argument("--num-seeds", type=int, default=None, help="Convenience: number of seeds starting from --start-seed.")
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
    parser.add_argument("--control-mode", default=None)
    parser.add_argument("--pick-executor", default="placeholder", choices=["placeholder", "sim_topdown", "sim_pick_place"])
    parser.add_argument("--grasp-target-mode", default="semantic", choices=["semantic", "refined"])
    parser.add_argument("--place-target-source", default="none", choices=["none", "oracle_cubeB_pose", "predicted_place_object"])
    parser.add_argument("--place-query", default="cube")
    parser.add_argument("--place-min-distance-from-pick", type=float, default=0.05)
    parser.add_argument("--place-target-z", type=float, default=0.02)
    parser.add_argument("--sensor-width", type=int, default=None)
    parser.add_argument("--sensor-height", type=int, default=None)
    parser.add_argument("--merge-distance", type=float, default=0.08)
    parser.add_argument("--enable-closed-loop-reobserve", action="store_true")
    parser.add_argument("--closed-loop-max-extra-views", type=int, default=1)
    parser.add_argument("--enable-selected-object-continuity", action="store_true")
    parser.add_argument("--selected-object-continuity-distance-scale", type=float, default=1.0)
    parser.add_argument("--enable-post-reobserve-selection-continuity", action="store_true")
    parser.add_argument("--post-reobserve-selection-margin", type=float, default=0.03)
    parser.add_argument("--capture-execution-video", action="store_true", help="Forward opt-in execution video capture to child runs.")
    parser.add_argument("--execution-video-fps", type=float, default=24.0)
    parser.add_argument("--execution-video-camera-name", default="base_camera")
    parser.add_argument("--execution-video-every-n-steps", type=int, default=1)
    parser.add_argument("--execution-video-width", type=int, default=None)
    parser.add_argument("--execution-video-height", type=int, default=None)
    parser.add_argument("--fail-on-child-error", action="store_true", help="Exit nonzero if any child debug run fails.")
    parser.add_argument("--log-level", default="INFO", help="Benchmark logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pick_executor in {"sim_topdown", "sim_pick_place"} and args.control_mode is None:
        args.control_mode = "pd_ee_delta_pos"
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")

    queries = load_queries(args.queries_file, args.queries)
    if (args.start_seed is None) != (args.num_seeds is None):
        raise ValueError("--start-seed and --num-seeds must be supplied together.")
    if args.num_seeds is not None and args.num_seeds < 1:
        raise ValueError("--num-seeds must be at least 1.")
    if args.start_seed is not None and args.num_seeds is not None:
        seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    elif args.seeds:
        seeds = args.seeds
    else:
        seeds = list(range(args.num_runs))
    if not seeds:
        raise ValueError("No seeds specified. Use --seeds, --start-seed/--num-seeds, or --num-runs.")

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
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "view_ids": [view_id for view_id in args.view_ids if view_id],
        "camera_name": args.camera_name,
        "view_preset": args.view_preset,
        "control_mode": args.control_mode,
        "pick_executor": args.pick_executor,
        "grasp_target_mode": args.grasp_target_mode,
        "place_target_source": args.place_target_source,
        "place_query": args.place_query,
        "place_min_distance_from_pick": float(args.place_min_distance_from_pick),
        "place_target_z": float(args.place_target_z),
        "detector_backend": args.detector_backend,
        "skip_clip": bool(args.skip_clip),
        "depth_scale": float(args.depth_scale),
        "merge_distance": float(args.merge_distance),
        "closed_loop_reobserve_enabled": bool(args.enable_closed_loop_reobserve),
        "closed_loop_max_extra_views": int(args.closed_loop_max_extra_views),
        "closed_loop_selected_object_continuity_enabled": bool(args.enable_selected_object_continuity),
        "selected_object_continuity_distance_scale": float(args.selected_object_continuity_distance_scale),
        "closed_loop_post_selection_continuity_enabled": bool(args.enable_post_reobserve_selection_continuity),
        "post_reobserve_selection_margin": float(args.post_reobserve_selection_margin),
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
        "--pick-executor",
        args.pick_executor,
        "--grasp-target-mode",
        args.grasp_target_mode,
        "--place-target-source",
        args.place_target_source,
        "--place-query",
        args.place_query,
        "--place-min-distance-from-pick",
        str(args.place_min_distance_from_pick),
        "--place-target-z",
        str(args.place_target_z),
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
    if args.control_mode:
        command.extend(["--control-mode", args.control_mode])
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
    if getattr(args, "sensor_width", None) is not None:
        command.extend(["--sensor-width", str(args.sensor_width)])
    if getattr(args, "sensor_height", None) is not None:
        command.extend(["--sensor-height", str(args.sensor_height)])
    if args.enable_closed_loop_reobserve:
        command.append("--enable-closed-loop-reobserve")
        command.extend(["--closed-loop-max-extra-views", str(args.closed_loop_max_extra_views)])
    if args.enable_selected_object_continuity:
        command.append("--enable-selected-object-continuity")
        command.extend(
            [
                "--selected-object-continuity-distance-scale",
                str(args.selected_object_continuity_distance_scale),
            ]
        )
    if args.enable_post_reobserve_selection_continuity:
        command.append("--enable-post-reobserve-selection-continuity")
        command.extend(["--post-reobserve-selection-margin", str(args.post_reobserve_selection_margin)])
    if getattr(args, "capture_execution_video", False):
        command.append("--capture-execution-video")
        command.extend(["--execution-video-fps", str(getattr(args, "execution_video_fps", 24.0))])
        command.extend(["--execution-video-camera-name", getattr(args, "execution_video_camera_name", "base_camera")])
        command.extend(["--execution-video-every-n-steps", str(getattr(args, "execution_video_every_n_steps", 1))])
        if getattr(args, "execution_video_width", None) is not None:
            command.extend(["--execution-video-width", str(args.execution_video_width)])
        if getattr(args, "execution_video_height", None) is not None:
            command.extend(["--execution-video-height", str(args.execution_video_height)])
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
        "selected_world_xyz": _join_sequence(summary.get("selected_world_xyz")),
        "selected_grasp_world_xyz": _join_sequence(summary.get("selected_grasp_world_xyz")),
        "selected_semantic_to_grasp_xy_distance": _as_float(
            summary.get("selected_semantic_to_grasp_xy_distance"),
            0.0,
        ),
        "selected_semantic_to_grasp_z_delta": _as_float(
            summary.get("selected_semantic_to_grasp_z_delta"),
            0.0,
        ),
        "selected_grasp_observation_count": _as_int(summary.get("selected_grasp_observation_count"), 0),
        "selected_grasp_observation_xy_spread": _as_float(
            summary.get("selected_grasp_observation_xy_spread"),
            0.0,
        ),
        "selected_grasp_observation_z_spread": _as_float(
            summary.get("selected_grasp_observation_z_spread"),
            0.0,
        ),
        "selected_grasp_observation_max_distance_to_fused": _as_float(
            summary.get("selected_grasp_observation_max_distance_to_fused"),
            0.0,
        ),
        "selected_grasp_observation_history_json": _json_dumps_compact(
            summary.get("selected_grasp_observation_history", [])
        ),
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
        "closed_loop_before_selected_present_after": _as_bool(
            summary.get("closed_loop_before_selected_present_after")
        ),
        "closed_loop_before_selected_still_selected": _as_bool(
            summary.get("closed_loop_before_selected_still_selected")
        ),
        "closed_loop_before_selected_received_observation": _as_bool(
            summary.get("closed_loop_before_selected_received_observation")
        ),
        "closed_loop_before_selected_gained_view_support": _as_bool(
            summary.get("closed_loop_before_selected_gained_view_support")
        ),
        "closed_loop_before_selected_merged_extra_view_ids": " ".join(
            str(item) for item in summary.get("closed_loop_before_selected_merged_extra_view_ids", [])
        ),
        "closed_loop_before_selected_delta_num_observations": _as_int(
            summary.get("closed_loop_before_selected_delta_num_observations"),
            0,
        ),
        "closed_loop_before_selected_delta_num_views": _as_int(
            summary.get("closed_loop_before_selected_delta_num_views"),
            0,
        ),
        "closed_loop_extra_view_absorber_object_ids": " ".join(
            str(item) for item in summary.get("closed_loop_extra_view_absorber_object_ids", [])
        ),
        "closed_loop_extra_view_absorber_count": _as_int(
            summary.get("closed_loop_extra_view_absorber_count"),
            0,
        ),
        "closed_loop_final_selected_absorbed_extra_view": _as_bool(
            summary.get("closed_loop_final_selected_absorbed_extra_view")
        ),
        "closed_loop_extra_view_third_object_ids": " ".join(
            str(item) for item in summary.get("closed_loop_extra_view_third_object_ids", [])
        ),
        "closed_loop_extra_view_third_object_involved": _as_bool(
            summary.get("closed_loop_extra_view_third_object_involved")
        ),
        "closed_loop_selected_object_continuity_enabled": _as_bool(
            summary.get("closed_loop_selected_object_continuity_enabled")
        ),
        "closed_loop_preferred_merge_count": _as_int(
            summary.get("closed_loop_preferred_merge_count"),
            0,
        ),
        "closed_loop_preferred_merge_rate": _as_float(
            summary.get("closed_loop_preferred_merge_rate"),
            0.0,
        ),
        "closed_loop_post_selection_continuity_enabled": _as_bool(
            summary.get("closed_loop_post_selection_continuity_enabled")
        ),
        "closed_loop_post_selection_continuity_eligible": _as_bool(
            summary.get("closed_loop_post_selection_continuity_eligible")
        ),
        "closed_loop_post_selection_continuity_applied": _as_bool(
            summary.get("closed_loop_post_selection_continuity_applied")
        ),
        "closed_loop_post_selection_continuity_reason": _optional_str(
            summary.get("closed_loop_post_selection_continuity_reason")
        ),
        "grasp_attempted": _as_bool(summary.get("grasp_attempted")),
        "pick_success": _as_bool(summary.get("pick_success")),
        "task_success": _as_bool(summary.get("task_success")),
        "is_grasped": _as_bool(summary.get("is_grasped")),
        "pick_stage": _optional_str(summary.get("pick_stage")) or "not_attempted",
        "pick_target_xyz": _join_sequence(summary.get("pick_target_xyz")),
        "pick_target_source": _optional_str(summary.get("pick_target_source")),
        "place_attempted": _as_bool(summary.get("place_attempted")),
        "place_success": _as_bool(summary.get("place_success")),
        "place_target_xyz": _join_sequence(summary.get("place_target_xyz")),
        "place_target_source": _optional_str(summary.get("place_target_source")),
        "place_query": _optional_str(summary.get("place_query")),
        "place_selection_reason": _optional_str(summary.get("place_selection_reason")),
        "place_pick_xy_distance": _as_float(summary.get("place_pick_xy_distance"), 0.0),
        "task_grasp_target_guard_applied": _as_bool(summary.get("task_grasp_target_guard_applied")),
        "task_grasp_target_guard_reason": _optional_str(summary.get("task_grasp_target_guard_reason")),
        "execution_video_path": _execution_video_path(summary),
        "execution_video_status": _execution_video_status(summary),
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
            "closed_loop_before_selected_still_selected_rate": 0.0,
            "closed_loop_before_selected_received_observation_rate": 0.0,
            "closed_loop_before_selected_gained_view_support_rate": 0.0,
            "closed_loop_final_selected_absorbed_extra_view_rate": 0.0,
            "closed_loop_extra_view_third_object_involved_rate": 0.0,
            "mean_closed_loop_preferred_merge_count": 0.0,
            "mean_closed_loop_preferred_merge_rate": 0.0,
            "closed_loop_post_selection_continuity_eligibility_rate": 0.0,
            "closed_loop_post_selection_continuity_apply_rate": 0.0,
            "mean_closed_loop_delta_num_views": 0.0,
            "mean_closed_loop_delta_num_memory_objects": 0.0,
            "mean_closed_loop_delta_num_observations_added": 0.0,
            "mean_closed_loop_delta_selected_overall_confidence": 0.0,
            "mean_closed_loop_delta_selected_num_views": 0.0,
            "mean_closed_loop_delta_selected_num_observations": 0.0,
            "mean_closed_loop_before_selected_delta_num_observations": 0.0,
            "mean_closed_loop_before_selected_delta_num_views": 0.0,
            "mean_closed_loop_extra_view_absorber_count": 0.0,
            "reobserve_reason_counts": {},
            "initial_reobserve_reason_counts": {},
            "final_reobserve_reason_counts": {},
            "closed_loop_post_selection_continuity_reason_counts": {},
            "grasp_attempted_rate": 0.0,
            "pick_success_rate": 0.0,
            "task_success_rate": 0.0,
            "place_attempted_rate": 0.0,
            "place_success_rate": 0.0,
            "is_grasped_rate": 0.0,
            "pick_stage_counts": {},
            "place_stage_counts": {},
            "mean_selected_semantic_to_grasp_xy_distance": 0.0,
            "mean_selected_grasp_observation_count": 0.0,
            "mean_selected_grasp_observation_xy_spread": 0.0,
            "mean_selected_grasp_observation_max_distance_to_fused": 0.0,
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
        "closed_loop_before_selected_still_selected_rate": _mean(
            1 if _as_bool(row.get("closed_loop_before_selected_still_selected")) else 0 for row in rows
        ),
        "closed_loop_before_selected_received_observation_rate": _mean(
            1 if _as_bool(row.get("closed_loop_before_selected_received_observation")) else 0 for row in rows
        ),
        "closed_loop_before_selected_gained_view_support_rate": _mean(
            1 if _as_bool(row.get("closed_loop_before_selected_gained_view_support")) else 0 for row in rows
        ),
        "closed_loop_final_selected_absorbed_extra_view_rate": _mean(
            1 if _as_bool(row.get("closed_loop_final_selected_absorbed_extra_view")) else 0 for row in rows
        ),
        "closed_loop_extra_view_third_object_involved_rate": _mean(
            1 if _as_bool(row.get("closed_loop_extra_view_third_object_involved")) else 0 for row in rows
        ),
        "mean_closed_loop_preferred_merge_count": _mean(
            _as_int(row.get("closed_loop_preferred_merge_count"), 0) for row in rows
        ),
        "mean_closed_loop_preferred_merge_rate": _mean(
            _as_float(row.get("closed_loop_preferred_merge_rate"), 0.0) for row in rows
        ),
        "closed_loop_post_selection_continuity_eligibility_rate": _mean(
            1 if _as_bool(row.get("closed_loop_post_selection_continuity_eligible")) else 0 for row in rows
        ),
        "closed_loop_post_selection_continuity_apply_rate": _mean(
            1 if _as_bool(row.get("closed_loop_post_selection_continuity_applied")) else 0 for row in rows
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
        "mean_closed_loop_before_selected_delta_num_observations": _mean(
            _as_int(row.get("closed_loop_before_selected_delta_num_observations"), 0) for row in rows
        ),
        "mean_closed_loop_before_selected_delta_num_views": _mean(
            _as_int(row.get("closed_loop_before_selected_delta_num_views"), 0) for row in rows
        ),
        "mean_closed_loop_extra_view_absorber_count": _mean(
            _as_int(row.get("closed_loop_extra_view_absorber_count"), 0) for row in rows
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
        "closed_loop_post_selection_continuity_reason_counts": count_values(
            row.get("closed_loop_post_selection_continuity_reason") or "none"
            for row in rows
        ),
        "grasp_attempted_rate": _mean(1 if _as_bool(row.get("grasp_attempted")) else 0 for row in rows),
        "pick_success_rate": _mean(1 if _as_bool(row.get("pick_success")) else 0 for row in rows),
        "place_attempted_rate": _mean(1 if _as_bool(row.get("place_attempted")) else 0 for row in rows),
        "place_success_rate": _mean(1 if _as_bool(row.get("place_success")) else 0 for row in rows),
        "task_success_rate": _mean(1 if _as_bool(row.get("task_success")) else 0 for row in rows),
        "is_grasped_rate": _mean(1 if _as_bool(row.get("is_grasped")) else 0 for row in rows),
        "pick_stage_counts": count_values(row.get("pick_stage") or "unknown" for row in rows),
        "place_stage_counts": count_values(
            row.get("pick_stage") or "unknown" for row in rows if _as_bool(row.get("place_attempted"))
        ),
        "mean_selected_semantic_to_grasp_xy_distance": _mean(
            _as_float(row.get("selected_semantic_to_grasp_xy_distance"), 0.0) for row in rows
        ),
        "mean_selected_grasp_observation_count": _mean(
            _as_int(row.get("selected_grasp_observation_count"), 0) for row in rows
        ),
        "mean_selected_grasp_observation_xy_spread": _mean(
            _as_float(row.get("selected_grasp_observation_xy_spread"), 0.0) for row in rows
        ),
        "mean_selected_grasp_observation_max_distance_to_fused": _mean(
            _as_float(row.get("selected_grasp_observation_max_distance_to_fused"), 0.0) for row in rows
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
        "selected_world_xyz": "",
        "selected_grasp_world_xyz": "",
        "selected_semantic_to_grasp_xy_distance": 0.0,
        "selected_semantic_to_grasp_z_delta": 0.0,
        "selected_grasp_observation_count": 0,
        "selected_grasp_observation_xy_spread": 0.0,
        "selected_grasp_observation_z_spread": 0.0,
        "selected_grasp_observation_max_distance_to_fused": 0.0,
        "selected_grasp_observation_history_json": "[]",
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
        "closed_loop_before_selected_present_after": False,
        "closed_loop_before_selected_still_selected": False,
        "closed_loop_before_selected_received_observation": False,
        "closed_loop_before_selected_gained_view_support": False,
        "closed_loop_before_selected_merged_extra_view_ids": "",
        "closed_loop_before_selected_delta_num_observations": 0,
        "closed_loop_before_selected_delta_num_views": 0,
        "closed_loop_extra_view_absorber_object_ids": "",
        "closed_loop_extra_view_absorber_count": 0,
        "closed_loop_final_selected_absorbed_extra_view": False,
        "closed_loop_extra_view_third_object_ids": "",
        "closed_loop_extra_view_third_object_involved": False,
        "closed_loop_selected_object_continuity_enabled": False,
        "closed_loop_preferred_merge_count": 0,
        "closed_loop_preferred_merge_rate": 0.0,
        "closed_loop_post_selection_continuity_enabled": False if args is None else bool(getattr(args, "enable_post_reobserve_selection_continuity", False)),
        "closed_loop_post_selection_continuity_eligible": False,
        "closed_loop_post_selection_continuity_applied": False,
        "closed_loop_post_selection_continuity_reason": None,
        "grasp_attempted": False,
        "pick_success": False,
        "task_success": False,
        "is_grasped": False,
        "pick_stage": "run_failed",
        "pick_target_xyz": "",
        "pick_target_source": None,
        "place_attempted": False,
        "place_success": False,
        "place_target_xyz": "",
        "place_target_source": None,
        "place_query": None,
        "place_selection_reason": None,
        "place_pick_xy_distance": 0.0,
        "task_grasp_target_guard_applied": False,
        "task_grasp_target_guard_reason": None,
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
    print(f"  Pick success:  {metrics.get('pick_success_rate', 0.0):.3f}")
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


def _join_sequence(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def _json_dumps_compact(value: Any) -> str:
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    except TypeError:
        return "[]"


def _execution_video_path(summary: dict[str, Any]) -> str | None:
    execution_video = summary.get("execution_video")
    if isinstance(execution_video, dict):
        return _optional_str(execution_video.get("video_path"))
    return None


def _execution_video_status(summary: dict[str, Any]) -> str | None:
    execution_video = summary.get("execution_video")
    if isinstance(execution_video, dict):
        return _optional_str(execution_video.get("status"))
    return None


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
