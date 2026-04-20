"""Lift 2D candidate boxes into 3D using depth and camera calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.env.camera_utils import normalize_depth, normalize_rgb, validate_rgb_depth_consistency
from src.geometry.rgbd_to_pointcloud import (
    create_open3d_point_cloud,
    fallback_pinhole_intrinsics,
    save_point_cloud,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class Candidate3D:
    """3D estimate for a lifted 2D candidate."""

    box_xyxy: np.ndarray
    camera_xyz: np.ndarray | None
    world_xyz: np.ndarray | None
    num_points: int
    depth_valid_ratio: float
    point_cloud_path: str | None = None
    segmentation_id: int | None = None
    box_area: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "box_xyxy": np.asarray(self.box_xyxy, dtype=float).tolist(),
            "camera_xyz": None if self.camera_xyz is None else np.asarray(self.camera_xyz, dtype=float).tolist(),
            "world_xyz": None if self.world_xyz is None else np.asarray(self.world_xyz, dtype=float).tolist(),
            "num_points": int(self.num_points),
            "depth_valid_ratio": float(self.depth_valid_ratio),
            "point_cloud_path": self.point_cloud_path,
            "segmentation_id": self.segmentation_id,
            "box_area": int(self.box_area),
            "metadata": self.metadata,
        }


def lift_box_to_3d(
    rgb: np.ndarray,
    depth: np.ndarray,
    box_xyxy: np.ndarray | Sequence[float],
    intrinsic: np.ndarray | None = None,
    extrinsic: np.ndarray | None = None,
    segmentation: np.ndarray | None = None,
    segmentation_id: int | None = None,
    use_segmentation: bool = False,
    output_point_cloud_path: str | Path | None = None,
    depth_scale: float = 1.0,
    min_depth: float = 1e-6,
    max_depth: float | None = None,
    fallback_fov_degrees: float = 60.0,
    center_strategy: str = "median",
    background_segmentation_ids: Sequence[int] = (0, -1),
) -> Candidate3D:
    """Lift a candidate box to camera/world coordinates using valid depth pixels."""

    rgb_array = normalize_rgb(rgb)
    depth_array = normalize_depth(depth) / float(depth_scale)
    validate_rgb_depth_consistency(rgb_array, depth_array)

    height, width = depth_array.shape
    box = np.asarray(box_xyxy, dtype=np.float32)
    x0, y0, x1, y1 = clip_box_to_bounds(box, width=width, height=height)
    box_area = max(0, x1 - x0) * max(0, y1 - y0)
    if box_area == 0:
        return Candidate3D(
            box_xyxy=box,
            camera_xyz=None,
            world_xyz=None,
            num_points=0,
            depth_valid_ratio=0.0,
            box_area=0,
            metadata={"reason": "empty_box"},
        )

    depth_crop = depth_array[y0:y1, x0:x1]
    valid_mask = np.isfinite(depth_crop) & (depth_crop > min_depth)
    if max_depth is not None:
        valid_mask &= depth_crop <= max_depth

    chosen_segmentation_id = segmentation_id
    if use_segmentation and segmentation is not None:
        segmentation_array = np.squeeze(np.asarray(segmentation))
        if segmentation_array.shape != depth_array.shape:
            raise ValueError(
                "Segmentation shape must match depth shape when use_segmentation=True: "
                f"segmentation={segmentation_array.shape}, depth={depth_array.shape}"
            )
        segmentation_crop = segmentation_array[y0:y1, x0:x1]
        if chosen_segmentation_id is None:
            chosen_segmentation_id = _choose_segmentation_id(
                segmentation_crop,
                valid_mask=valid_mask,
                background_ids=background_segmentation_ids,
            )
        if chosen_segmentation_id is not None:
            valid_mask &= segmentation_crop == chosen_segmentation_id

    num_points = int(np.count_nonzero(valid_mask))
    depth_valid_ratio = float(num_points / box_area) if box_area else 0.0
    if num_points == 0:
        return Candidate3D(
            box_xyxy=box,
            camera_xyz=None,
            world_xyz=None,
            num_points=0,
            depth_valid_ratio=depth_valid_ratio,
            segmentation_id=chosen_segmentation_id,
            box_area=box_area,
            metadata={"reason": "no_valid_depth"},
        )

    intrinsic_used = (
        np.asarray(intrinsic, dtype=np.float32)
        if intrinsic is not None
        else fallback_pinhole_intrinsics(width=width, height=height, fov_degrees=fallback_fov_degrees)
    )
    camera_points, colors = _project_crop_to_camera_points(
        rgb=rgb_array,
        depth=depth_array,
        valid_mask=valid_mask,
        box_bounds=(x0, y0, x1, y1),
        intrinsic=intrinsic_used,
    )
    world_points = transform_camera_points(camera_points, extrinsic) if extrinsic is not None else None
    camera_center = _estimate_center(camera_points, strategy=center_strategy)
    world_center = _estimate_center(world_points, strategy=center_strategy) if world_points is not None else None

    point_cloud_path = None
    if output_point_cloud_path is not None:
        points_for_cloud = world_points if world_points is not None else camera_points
        point_cloud = create_open3d_point_cloud(points_for_cloud, colors)
        point_cloud_path = str(save_point_cloud(point_cloud, output_point_cloud_path))

    return Candidate3D(
        box_xyxy=box,
        camera_xyz=camera_center,
        world_xyz=world_center,
        num_points=num_points,
        depth_valid_ratio=depth_valid_ratio,
        point_cloud_path=point_cloud_path,
        segmentation_id=chosen_segmentation_id,
        box_area=box_area,
        metadata={
            "box_bounds_xyxy_exclusive": [x0, y0, x1, y1],
            "intrinsics_source": "provided" if intrinsic is not None else "fallback",
            "extrinsics_source": "provided" if extrinsic is not None else "missing",
            "center_strategy": center_strategy,
        },
    )


def clip_box_to_bounds(box_xyxy: np.ndarray | Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Convert an ``xyxy`` box to integer image bounds with exclusive max edges."""

    box = np.asarray(box_xyxy, dtype=float)
    if box.shape != (4,):
        raise ValueError(f"Box must have shape (4,), got {box.shape}")
    x0 = int(np.floor(np.clip(min(box[0], box[2]), 0, width)))
    y0 = int(np.floor(np.clip(min(box[1], box[3]), 0, height)))
    x1 = int(np.ceil(np.clip(max(box[0], box[2]), 0, width)))
    y1 = int(np.ceil(np.clip(max(box[1], box[3]), 0, height)))
    return x0, y0, x1, y1


def transform_camera_points(points: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Apply a camera-to-world transform to camera-frame points."""

    extrinsic_array = np.asarray(extrinsic, dtype=np.float32)
    if extrinsic_array.shape != (4, 4):
        raise ValueError(f"Extrinsic matrix must have shape (4, 4), got {extrinsic_array.shape}")
    homogeneous = np.concatenate([points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    return (homogeneous @ extrinsic_array.T)[:, :3].astype(np.float32)


def _project_crop_to_camera_points(
    rgb: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    box_bounds: tuple[int, int, int, int],
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = box_bounds
    intrinsic_array = np.asarray(intrinsic, dtype=np.float32)
    if intrinsic_array.shape != (3, 3):
        raise ValueError(f"Intrinsic matrix must have shape (3, 3), got {intrinsic_array.shape}")

    v_coords, u_coords = np.indices((y1 - y0, x1 - x0))
    u_coords = u_coords + x0
    v_coords = v_coords + y0
    z_values = depth[y0:y1, x0:x1][valid_mask]
    x_values = (u_coords[valid_mask].astype(np.float32) - intrinsic_array[0, 2]) * z_values / intrinsic_array[0, 0]
    y_values = (v_coords[valid_mask].astype(np.float32) - intrinsic_array[1, 2]) * z_values / intrinsic_array[1, 1]
    points = np.stack([x_values, y_values, z_values], axis=1).astype(np.float32)
    colors = rgb[y0:y1, x0:x1][valid_mask].astype(np.float32) / 255.0
    return points, colors


def _estimate_center(points: np.ndarray | None, strategy: str) -> np.ndarray | None:
    if points is None or points.size == 0:
        return None
    if strategy == "median":
        return np.median(points, axis=0).astype(np.float32)
    if strategy == "mean":
        return np.mean(points, axis=0).astype(np.float32)
    raise ValueError(f"Unknown center strategy: {strategy}")


def _choose_segmentation_id(
    segmentation_crop: np.ndarray,
    valid_mask: np.ndarray,
    background_ids: Sequence[int],
) -> int | None:
    ids = segmentation_crop[valid_mask]
    if ids.size == 0:
        return None
    background = set(int(item) for item in background_ids)
    foreground_ids = np.asarray([item for item in ids.reshape(-1) if int(item) not in background])
    ids_to_count = foreground_ids if foreground_ids.size else ids.reshape(-1)
    values, counts = np.unique(ids_to_count, return_counts=True)
    return int(values[np.argmax(counts)]) if values.size else None

