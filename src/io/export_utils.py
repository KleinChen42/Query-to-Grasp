"""Utilities for saving Phase 1 observation artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.env.camera_utils import ObservationFrame

LOGGER = logging.getLogger(__name__)


def export_observation_frame(
    frame: ObservationFrame,
    output_dir: str | Path,
    env_name: str,
    step_name: str = "reset",
) -> dict[str, Path]:
    """Save one normalized observation frame and metadata to disk.

    The function writes files only for modalities that are present. Metadata is
    always written and records which modalities were missing.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    if frame.rgb is not None:
        rgb_path = output_path / "rgb.png"
        save_rgb_png(frame.rgb, rgb_path)
        saved_paths["rgb"] = rgb_path

    if frame.depth is not None:
        depth_path = output_path / "depth.npy"
        np.save(depth_path, frame.depth)
        saved_paths["depth"] = depth_path

    if frame.segmentation is not None:
        segmentation_path = output_path / "segmentation.npy"
        np.save(segmentation_path, frame.segmentation)
        saved_paths["segmentation"] = segmentation_path

    metadata_path = output_path / "metadata.json"
    metadata = build_metadata(frame=frame, env_name=env_name, step_name=step_name, saved_paths=saved_paths)
    write_json(metadata, metadata_path)
    saved_paths["metadata"] = metadata_path

    LOGGER.info("Saved observation artifacts to %s", output_path)
    return saved_paths


def build_metadata(
    frame: ObservationFrame,
    env_name: str,
    step_name: str,
    saved_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Build JSON metadata for one exported observation frame."""

    saved_paths = saved_paths or {}
    arrays = {
        "rgb": _array_metadata(frame.rgb),
        "depth": _array_metadata(frame.depth),
        "segmentation": _array_metadata(frame.segmentation),
    }
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment_name": env_name,
        "step_name": step_name,
        "modalities_present": {name: metadata is not None for name, metadata in arrays.items()},
        "arrays": arrays,
        "source_keys": frame.source_keys,
        "observation_keys_present": frame.observation_keys,
        "observation_summary": frame.observation_summary,
        "camera_info": frame.camera_info.to_json_dict(),
        "saved_files": {name: str(path) for name, path in saved_paths.items()},
    }


def save_rgb_png(rgb: np.ndarray, path: str | Path) -> None:
    """Save a ``uint8`` RGB array as a PNG."""

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save rgb.png. Install it with `pip install pillow`.") from exc

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {array.shape}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8, copy=False), mode="RGB").save(path)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write JSON with stable indentation."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(data), file, indent=2, sort_keys=True)


def _array_metadata(array: np.ndarray | None) -> dict[str, Any] | None:
    if array is None:
        return None
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
