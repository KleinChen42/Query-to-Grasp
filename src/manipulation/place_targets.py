"""Perception-derived place target helpers for simulated StackCube baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class PredictedPlaceTarget:
    """A non-oracle place target selected from perception outputs."""

    place_xyz: np.ndarray
    source: str = "predicted_place_object"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        xyz = np.asarray(self.place_xyz, dtype=np.float32).reshape(-1)
        if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
            raise ValueError("PredictedPlaceTarget.place_xyz must be finite xyz.")
        object.__setattr__(self, "place_xyz", xyz)


def select_candidate_place_target(
    candidates: Sequence[Any],
    pick_xyz: Any,
    min_xy_distance: float = 0.05,
    place_query: str | None = None,
    place_target_z: float | None = None,
) -> PredictedPlaceTarget | None:
    """Select the first ranked 3D candidate far enough from the pick target."""

    pick = _valid_xyz(pick_xyz)
    if pick is None:
        return None
    min_distance = float(min_xy_distance)
    eligible: list[tuple[int, np.ndarray, float, Any]] = []
    close_count = 0
    invalid_count = 0
    for index, candidate in enumerate(candidates):
        xyz = _representative_place_xyz(candidate, place_target_z=place_target_z)
        if xyz is None:
            invalid_count += 1
            continue
        xy_distance = float(np.linalg.norm(xyz[:2] - pick[:2]))
        if xy_distance < min_distance:
            close_count += 1
            continue
        eligible.append((index, xyz, xy_distance, candidate))
    if not eligible:
        return None
    index, xyz, xy_distance, candidate = eligible[0]
    return PredictedPlaceTarget(
        place_xyz=xyz,
        metadata={
            "place_query": place_query,
            "selection_reason": "first_ranked_candidate_far_from_pick",
            "selected_candidate_index": int(index),
            "num_candidates": int(len(candidates)),
            "num_eligible_candidates": int(len(eligible)),
            "num_close_to_pick_rejected": int(close_count),
            "num_invalid_candidates": int(invalid_count),
            "place_pick_xy_distance": xy_distance,
            "selected_world_xyz": xyz.tolist(),
            "raw_semantic_world_xyz": _optional_xyz_list(getattr(candidate, "world_xyz", None)),
            "selected_grasp_world_xyz": _optional_xyz_list(getattr(candidate, "grasp_world_xyz", None)),
            "place_target_z": None if place_target_z is None else float(place_target_z),
            "selected_num_points": int(getattr(candidate, "num_points", 0) or 0),
            "selected_depth_valid_ratio": float(getattr(candidate, "depth_valid_ratio", 0.0) or 0.0),
        },
    )


def select_memory_place_target(
    objects: Sequence[Any],
    pick_xyz: Any,
    selected_pick_object_id: str | None = None,
    min_xy_distance: float = 0.05,
    place_query: str | None = None,
    place_target_z: float | None = None,
) -> PredictedPlaceTarget | None:
    """Select the highest-confidence memory object far enough from the pick target."""

    pick = _valid_xyz(pick_xyz)
    if pick is None:
        return None
    min_distance = float(min_xy_distance)
    eligible: list[tuple[float, int, str, np.ndarray, float, Any]] = []
    close_count = 0
    same_object_count = 0
    invalid_count = 0
    for obj in objects:
        object_id = str(getattr(obj, "object_id", "") or "")
        if selected_pick_object_id is not None and object_id == selected_pick_object_id:
            same_object_count += 1
            continue
        xyz = _representative_place_xyz(obj, place_target_z=place_target_z)
        if xyz is None:
            invalid_count += 1
            continue
        xy_distance = float(np.linalg.norm(xyz[:2] - pick[:2]))
        if xy_distance < min_distance:
            close_count += 1
            continue
        eligible.append(
            (
                -float(getattr(obj, "overall_confidence", 0.0) or 0.0),
                -len(getattr(obj, "view_ids", []) or []),
                object_id,
                xyz,
                xy_distance,
                obj,
            )
        )
    if not eligible:
        return None
    _, _, _, xyz, xy_distance, obj = sorted(eligible, key=lambda item: item[:3])[0]
    return PredictedPlaceTarget(
        place_xyz=xyz,
        metadata={
            "place_query": place_query,
            "selection_reason": "highest_confidence_memory_object_far_from_pick",
            "selected_object_id": getattr(obj, "object_id", None),
            "selected_top_label": getattr(obj, "top_label", None),
            "selected_overall_confidence": float(getattr(obj, "overall_confidence", 0.0) or 0.0),
            "selected_num_views": len(getattr(obj, "view_ids", []) or []),
            "selected_num_observations": int(getattr(obj, "num_observations", 0) or 0),
            "num_memory_objects": int(len(objects)),
            "num_eligible_memory_objects": int(len(eligible)),
            "num_same_pick_object_rejected": int(same_object_count),
            "num_close_to_pick_rejected": int(close_count),
            "num_invalid_objects": int(invalid_count),
            "place_pick_xy_distance": xy_distance,
            "selected_world_xyz": xyz.tolist(),
            "raw_semantic_world_xyz": _optional_xyz_list(getattr(obj, "world_xyz", None)),
            "selected_grasp_world_xyz": _optional_xyz_list(getattr(obj, "grasp_world_xyz", None)),
            "place_target_z": None if place_target_z is None else float(place_target_z),
        },
    )


def _representative_place_xyz(value: Any, place_target_z: float | None) -> np.ndarray | None:
    """Use refined grasp/reference geometry for XY when available, with optional fixed Z."""

    xyz = _valid_xyz(getattr(value, "grasp_world_xyz", None))
    if xyz is None:
        xyz = _valid_xyz(getattr(value, "world_xyz", None))
    if xyz is None:
        return None
    if place_target_z is None:
        return xyz
    return np.asarray([float(xyz[0]), float(xyz[1]), float(place_target_z)], dtype=np.float32)


def _valid_xyz(value: Any) -> np.ndarray | None:
    try:
        xyz = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        return None
    return xyz


def _optional_xyz_list(value: Any) -> list[float] | None:
    xyz = _valid_xyz(value)
    if xyz is None:
        return None
    return xyz.astype(float).tolist()
