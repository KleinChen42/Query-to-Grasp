"""Minimal metrics for single-view placeholder-pick benchmark outputs."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_summary(path: str | Path) -> dict[str, Any]:
    """Load a per-run ``summary.json`` file."""

    return _load_json_dict(path)


def load_pick_result(path: str | Path) -> dict[str, Any]:
    """Load a per-run ``pick_result.json`` file."""

    return _load_json_dict(path)


def summarize_run(summary: dict[str, Any], pick_result: dict[str, Any]) -> dict[str, Any]:
    """Convert current pipeline outputs into one flat benchmark row."""

    num_3d_points = _as_int(summary.get("num_3d_points"), default=0)
    pick_success = _as_bool(summary.get("pick_success", pick_result.get("success", False)))
    pick_stage = str(summary.get("pick_stage") or pick_result.get("stage") or "unknown")
    grasp_attempted = _as_bool(summary.get("grasp_attempted", pick_result.get("grasp_attempted", False)))
    task_success = _as_bool(summary.get("task_success", pick_result.get("task_success", False)))
    place_attempted = _as_bool(summary.get("place_attempted", pick_result.get("place_attempted", False)))
    place_success = _as_bool(summary.get("place_success", pick_result.get("place_success", False)))
    is_grasped = _as_bool(summary.get("is_grasped", pick_result.get("is_grasped", False)))
    runtime_seconds = _as_float(summary.get("runtime_seconds"), default=0.0)
    num_detections = _as_int(summary.get("num_detections"), default=0)
    raw_num_detections = _as_int(summary.get("raw_num_detections"), default=num_detections)
    return {
        "query": str(summary.get("query") or ""),
        "raw_num_detections": raw_num_detections,
        "num_detections": num_detections,
        "num_ranked_candidates": _as_int(summary.get("num_ranked_candidates"), default=0),
        "top1_changed_by_rerank": _as_bool(summary.get("top1_changed_by_rerank", False)),
        "detector_top_phrase": _as_optional_str(summary.get("detector_top_phrase")),
        "final_top_phrase": _as_optional_str(summary.get("final_top_phrase")),
        "has_3d_target": _has_3d_target(summary, pick_result, num_3d_points),
        "num_3d_points": num_3d_points,
        "grasp_attempted": grasp_attempted,
        "pick_success": pick_success,
        "place_attempted": place_attempted,
        "place_success": place_success,
        "place_target_xyz": summary.get("place_target_xyz", pick_result.get("place_xyz")),
        "place_target_source": summary.get("place_target_source"),
        "place_query": summary.get("place_query"),
        "place_selection_reason": summary.get("place_selection_reason"),
        "place_pick_xy_distance": summary.get("place_pick_xy_distance"),
        "task_success": task_success,
        "is_grasped": is_grasped,
        "pick_stage": pick_stage,
        "execution_video_path": _execution_video_path(summary),
        "execution_video_status": _execution_video_status(summary),
        "runtime_seconds": runtime_seconds,
        "depth_noise_std_m": _as_float(summary.get("depth_noise_std_m"), default=0.0),
        "depth_dropout_prob": _as_float(summary.get("depth_dropout_prob"), default=0.0),
        "valid_depth_pixels_before": _as_int(summary.get("valid_depth_pixels_before"), default=0),
        "valid_depth_pixels_after": _as_int(summary.get("valid_depth_pixels_after"), default=0),
        "dropped_depth_pixels": _as_int(summary.get("dropped_depth_pixels"), default=0),
        "sensor_stress_applied": _as_bool(summary.get("sensor_stress_applied", False)),
        "artifacts": str(summary.get("artifacts") or ""),
    }


def aggregate_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate flat benchmark rows into simple mean/rate metrics."""

    total_runs = len(rows)
    if total_runs == 0:
        return {
            "total_runs": 0,
            "failed_runs": 0,
            "mean_raw_num_detections": 0.0,
            "mean_num_detections": 0.0,
            "mean_num_ranked_candidates": 0.0,
            "mean_num_3d_points": 0.0,
            "fraction_with_3d_target": 0.0,
            "grasp_attempted_rate": 0.0,
            "pick_success_rate": 0.0,
            "place_attempted_rate": 0.0,
            "place_success_rate": 0.0,
            "task_success_rate": 0.0,
            "is_grasped_rate": 0.0,
            "fraction_top1_changed_by_rerank": 0.0,
            "mean_runtime_seconds": 0.0,
            "pick_stage_counts": {},
            "place_stage_counts": {},
        }

    stage_counts = Counter(str(row.get("pick_stage") or "unknown") for row in rows)
    place_stage_counts = Counter(str(row.get("pick_stage") or "unknown") for row in rows if _as_bool(row.get("place_attempted")))
    return {
        "total_runs": total_runs,
        "failed_runs": sum(1 for row in rows if _as_bool(row.get("run_failed"))),
        "mean_raw_num_detections": _mean(_row_raw_num_detections(row) for row in rows),
        "mean_num_detections": _mean(_as_int(row.get("num_detections"), 0) for row in rows),
        "mean_num_ranked_candidates": _mean(_as_int(row.get("num_ranked_candidates"), 0) for row in rows),
        "mean_num_3d_points": _mean(_as_int(row.get("num_3d_points"), 0) for row in rows),
        "fraction_with_3d_target": _mean(1 if _as_bool(row.get("has_3d_target")) else 0 for row in rows),
        "grasp_attempted_rate": _mean(1 if _as_bool(row.get("grasp_attempted")) else 0 for row in rows),
        "pick_success_rate": _mean(1 if _as_bool(row.get("pick_success")) else 0 for row in rows),
        "place_attempted_rate": _mean(1 if _as_bool(row.get("place_attempted")) else 0 for row in rows),
        "place_success_rate": _mean(1 if _as_bool(row.get("place_success")) else 0 for row in rows),
        "task_success_rate": _mean(1 if _as_bool(row.get("task_success")) else 0 for row in rows),
        "is_grasped_rate": _mean(1 if _as_bool(row.get("is_grasped")) else 0 for row in rows),
        "fraction_top1_changed_by_rerank": _mean(1 if _as_bool(row.get("top1_changed_by_rerank")) else 0 for row in rows),
        "mean_runtime_seconds": _mean(_as_float(row.get("runtime_seconds"), 0.0) for row in rows),
        "pick_stage_counts": dict(sorted(stage_counts.items())),
        "place_stage_counts": dict(sorted(place_stage_counts.items())),
    }


def aggregate_runs_by_query(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate benchmark rows separately for each query string."""

    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        query = str(row.get("query") or "")
        grouped_rows.setdefault(query, []).append(row)
    return {query: aggregate_runs(query_rows) for query, query_rows in sorted(grouped_rows.items())}


def _load_json_dict(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}.")
    return data


def _has_3d_target(summary: dict[str, Any], pick_result: dict[str, Any], num_3d_points: int) -> bool:
    if _looks_like_xyz(summary.get("world_xyz")) or _looks_like_xyz(summary.get("camera_xyz")):
        return True
    if num_3d_points > 0:
        return True
    return _looks_like_xyz(pick_result.get("target_xyz"))


def _looks_like_xyz(value: Any) -> bool:
    try:
        array = np.asarray(value, dtype=np.float32)
    except Exception:
        return False
    return array.shape == (3,) and bool(np.all(np.isfinite(array)))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_raw_num_detections(row: dict[str, Any]) -> int:
    return _as_int(row.get("raw_num_detections"), _as_int(row.get("num_detections"), 0))


def _execution_video_path(summary: dict[str, Any]) -> str | None:
    execution_video = summary.get("execution_video")
    if isinstance(execution_video, dict):
        return _as_optional_str(execution_video.get("video_path"))
    return None


def _execution_video_status(summary: dict[str, Any]) -> str | None:
    execution_video = summary.get("execution_video")
    if isinstance(execution_video, dict):
        return _as_optional_str(execution_video.get("status"))
    return None


def _mean(values: Any) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(values_list) / len(values_list))
