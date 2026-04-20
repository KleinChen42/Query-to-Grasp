"""Observation parsing and camera utility helpers.

The functions in this module intentionally avoid depending on ManiSkill at
import time. ManiSkill observation dictionaries can vary by version, task,
observation mode, and vectorization settings, so the extraction logic is
keyword-based and defensive rather than tied to one exact schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class CameraInfo:
    """Camera calibration and pose data when available."""

    camera_name: str | None = None
    intrinsic: np.ndarray | None = None
    extrinsic: np.ndarray | None = None
    intrinsic_key: str | None = None
    extrinsic_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "camera_name": self.camera_name,
            "intrinsic": _array_to_list(self.intrinsic),
            "extrinsic": _array_to_list(self.extrinsic),
            "intrinsic_key": self.intrinsic_key,
            "extrinsic_key": self.extrinsic_key,
            "metadata": self.metadata,
        }


@dataclass
class ObservationFrame:
    """Normalized camera frame extracted from a raw observation dictionary."""

    rgb: np.ndarray | None
    depth: np.ndarray | None
    segmentation: np.ndarray | None
    camera_info: CameraInfo
    observation_keys: list[str]
    observation_summary: dict[str, dict[str, Any]]
    source_keys: dict[str, str | None]


def to_numpy(value: Any) -> np.ndarray:
    """Convert common array/tensor values to ``numpy.ndarray``.

    Supports NumPy arrays, PyTorch-like tensors, and array-like Python values.
    The conversion is deliberately local so the project does not need to import
    PyTorch just to parse an observation.
    """

    if isinstance(value, np.ndarray):
        return value

    array_value = value
    if hasattr(array_value, "detach"):
        array_value = array_value.detach()
    if hasattr(array_value, "cpu"):
        array_value = array_value.cpu()
    if hasattr(array_value, "numpy"):
        array_value = array_value.numpy()

    return np.asarray(array_value)


def normalize_rgb(value: Any) -> np.ndarray:
    """Normalize an RGB-like array to ``uint8`` with shape ``(H, W, 3)``."""

    array = _squeeze_vectorized_axis(to_numpy(value))

    if array.ndim == 2:
        array = np.repeat(array[..., None], repeats=3, axis=-1)
    elif array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)

    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"RGB array must have shape (H, W, 3/4), got {array.shape}")

    if array.shape[-1] == 4:
        array = array[..., :3]

    if np.issubdtype(array.dtype, np.floating):
        finite = array[np.isfinite(array)]
        max_value = float(finite.max()) if finite.size else 0.0
        if max_value <= 1.0:
            array = array * 255.0

    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


def normalize_depth(value: Any) -> np.ndarray:
    """Normalize a depth-like array to ``float32`` with shape ``(H, W)``."""

    array = _squeeze_vectorized_axis(to_numpy(value))
    array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError(f"Depth array must squeeze to shape (H, W), got {array.shape}")

    return array.astype(np.float32, copy=False)


def normalize_segmentation(value: Any) -> np.ndarray:
    """Normalize a segmentation-like array to shape ``(H, W)`` when possible."""

    array = _squeeze_vectorized_axis(to_numpy(value))
    array = np.squeeze(array)

    if array.ndim != 2:
        raise ValueError(f"Segmentation array must squeeze to shape (H, W), got {array.shape}")

    return array


def flatten_observation_keys(observation: Mapping[str, Any]) -> list[str]:
    """Return dotted leaf-key paths for a nested observation mapping."""

    return [path for path, _ in _iter_leaf_items(observation)]


def summarize_observation(observation: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return shape, dtype, and type information for every observation leaf."""

    summary: dict[str, dict[str, Any]] = {}
    for path, value in _iter_leaf_items(observation):
        item: dict[str, Any] = {"python_type": type(value).__name__}
        try:
            array = to_numpy(value)
            item["shape"] = list(array.shape)
            item["dtype"] = str(array.dtype)
        except Exception as exc:  # pragma: no cover - defensive metadata only
            item["conversion_error"] = str(exc)
        summary[path] = item
    return summary


def extract_camera_info(
    observation: Mapping[str, Any],
    camera_name: str | None = None,
) -> CameraInfo:
    """Extract camera intrinsics/extrinsics from a nested observation if present."""

    intrinsic, intrinsic_key = _find_matrix(
        observation,
        aliases=("intrinsic", "intrinsics", "intrinsic_cv", "camera_matrix"),
        valid_shapes=((3, 3),),
        camera_name=camera_name,
        allow_leaf_key_k=True,
    )
    extrinsic, extrinsic_key = _find_matrix(
        observation,
        aliases=("extrinsic", "extrinsics", "cam2world", "camera_to_world", "c2w"),
        valid_shapes=((4, 4),),
        camera_name=camera_name,
        allow_camera_pose_alias=True,
    )

    return CameraInfo(
        camera_name=camera_name,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        intrinsic_key=intrinsic_key,
        extrinsic_key=extrinsic_key,
    )


def extract_observation_frame(
    observation: Mapping[str, Any],
    camera_name: str | None = None,
) -> ObservationFrame:
    """Extract RGB, depth, segmentation, and camera info from a raw observation."""

    rgb, rgb_key = _find_array(
        observation,
        aliases=("rgb", "color", "colour"),
        normalizer=normalize_rgb,
        camera_name=camera_name,
    )
    depth, depth_key = _find_array(
        observation,
        aliases=("depth",),
        normalizer=normalize_depth,
        camera_name=camera_name,
    )
    segmentation, segmentation_key = _find_array(
        observation,
        aliases=("segmentation", "seg", "actor_seg", "object_seg", "label"),
        normalizer=normalize_segmentation,
        camera_name=camera_name,
    )

    camera_info = extract_camera_info(observation, camera_name=camera_name)
    return ObservationFrame(
        rgb=rgb,
        depth=depth,
        segmentation=segmentation,
        camera_info=camera_info,
        observation_keys=flatten_observation_keys(observation),
        observation_summary=summarize_observation(observation),
        source_keys={
            "rgb": rgb_key,
            "depth": depth_key,
            "segmentation": segmentation_key,
            "intrinsic": camera_info.intrinsic_key,
            "extrinsic": camera_info.extrinsic_key,
        },
    )


def validate_rgb_depth_consistency(rgb: np.ndarray, depth: np.ndarray) -> None:
    """Raise ``ValueError`` if RGB and depth spatial dimensions do not match."""

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB shape (H, W, 3), got {rgb.shape}")
    if depth.ndim != 2:
        raise ValueError(f"Expected depth shape (H, W), got {depth.shape}")
    if rgb.shape[:2] != depth.shape:
        raise ValueError(
            "RGB and depth spatial shapes differ: "
            f"rgb={rgb.shape[:2]}, depth={depth.shape}"
        )


def _iter_leaf_items(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_leaf_items(nested_value, child_prefix)
        return

    if isinstance(value, (list, tuple)) and not _looks_like_array_leaf(value):
        for index, nested_value in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            yield from _iter_leaf_items(nested_value, child_prefix)
        return

    yield prefix, value


def _looks_like_array_leaf(value: Sequence[Any]) -> bool:
    if not value:
        return True
    if len(value) > 16:
        return True
    return not any(isinstance(item, Mapping) for item in value)


def _find_array(
    observation: Mapping[str, Any],
    aliases: Sequence[str],
    normalizer: Any,
    camera_name: str | None,
) -> tuple[np.ndarray | None, str | None]:
    candidates: list[tuple[int, str, np.ndarray]] = []
    for path, value in _iter_leaf_items(observation):
        path_lower = path.lower()
        if not any(alias in path_lower for alias in aliases):
            continue
        try:
            array = normalizer(value)
        except Exception as exc:
            LOGGER.debug("Skipping observation leaf %s: %s", path, exc)
            continue
        candidates.append((_path_score(path, aliases, camera_name), path, array))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    score, path, array = candidates[0]
    LOGGER.debug("Selected observation leaf %s with score %s", path, score)
    return array, path


def _find_matrix(
    observation: Mapping[str, Any],
    aliases: Sequence[str],
    valid_shapes: Sequence[tuple[int, int]],
    camera_name: str | None,
    allow_leaf_key_k: bool = False,
    allow_camera_pose_alias: bool = False,
) -> tuple[np.ndarray | None, str | None]:
    candidates: list[tuple[int, str, np.ndarray]] = []
    for path, value in _iter_leaf_items(observation):
        path_lower = path.lower()
        leaf_key = path_lower.split(".")[-1]
        has_alias = any(alias in path_lower for alias in aliases)
        if allow_leaf_key_k and leaf_key == "k":
            has_alias = True
        if allow_camera_pose_alias and camera_name and leaf_key == "pose":
            has_alias = True
        if not has_alias:
            continue

        try:
            array = np.squeeze(to_numpy(value))
        except Exception as exc:
            LOGGER.debug("Skipping camera matrix leaf %s: %s", path, exc)
            continue

        if tuple(array.shape) not in valid_shapes:
            continue
        candidates.append((_path_score(path, aliases, camera_name), path, array.astype(np.float32)))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    _, path, matrix = candidates[0]
    return matrix, path


def _path_score(path: str, aliases: Sequence[str], camera_name: str | None) -> int:
    path_lower = path.lower()
    score = 0
    if camera_name and camera_name.lower() in path_lower:
        score += 100
    leaf_key = path_lower.split(".")[-1]
    if leaf_key in aliases:
        score += 25
    for alias in aliases:
        if alias in path_lower:
            score += 10
    if "sensor" in path_lower or "camera" in path_lower:
        score += 5
    return score


def _squeeze_vectorized_axis(array: np.ndarray) -> np.ndarray:
    """Remove a leading vectorized-environment axis when it has length one."""

    if array.ndim >= 4 and array.shape[0] == 1:
        return np.squeeze(array, axis=0)
    return array


def _array_to_list(array: np.ndarray | None) -> list[Any] | None:
    return array.tolist() if array is not None else None

