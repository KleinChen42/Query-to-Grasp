"""Privileged ManiSkill target discovery helpers for oracle baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OraclePlaceTargets:
    """Privileged pick/place targets."""

    pick_xyz: np.ndarray
    place_xyz: np.ndarray
    metadata: dict[str, Any]


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


def find_stackcube_oracle_place_xyz(env: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the privileged cubeB place target and diagnostic metadata."""

    targets = find_stackcube_oracle_place_targets(env)
    return targets.place_xyz, targets.metadata


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
