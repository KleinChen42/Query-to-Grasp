"""RGB-D to colored point cloud conversion utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.env.camera_utils import normalize_depth, normalize_rgb, validate_rgb_depth_consistency

LOGGER = logging.getLogger(__name__)


def fallback_pinhole_intrinsics(width: int, height: int, fov_degrees: float = 60.0) -> np.ndarray:
    """Create a simple pinhole intrinsic matrix when calibration is unavailable.

    Assumption: square pixels, centered principal point, and a horizontal field
    of view of ``fov_degrees``. This is only a debug fallback; real experiments
    should use simulator-provided intrinsics.
    """

    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got width={width}, height={height}")

    fov_radians = np.deg2rad(fov_degrees)
    focal = 0.5 * float(width) / np.tan(0.5 * fov_radians)
    intrinsic = np.array(
        [
            [focal, 0.0, (width - 1.0) / 2.0],
            [0.0, focal, (height - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    LOGGER.warning(
        "Camera intrinsics unavailable; using fallback pinhole intrinsics with %.1f deg FOV.",
        fov_degrees,
    )
    return intrinsic


def project_depth_to_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray | None = None,
    depth_scale: float = 1.0,
    min_depth: float = 1e-6,
    max_depth: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a depth image to 3D points.

    Args:
        depth: Depth image with shape ``(H, W)``.
        intrinsic: Camera intrinsic matrix with shape ``(3, 3)``.
        extrinsic: Optional camera-to-world transform with shape ``(4, 4)``.
            If provided, output points are transformed to world coordinates.
        depth_scale: Incoming depth is divided by this value. Keep at ``1.0``
            for depth already stored in meters.
        min_depth: Minimum positive depth to keep.
        max_depth: Optional maximum depth to keep.

    Returns:
        A tuple ``(points, valid_mask)`` where points has shape ``(N, 3)`` and
        valid_mask has shape ``(H, W)``.
    """

    depth_array = normalize_depth(depth) / float(depth_scale)
    intrinsic_array = np.asarray(intrinsic, dtype=np.float32)
    if intrinsic_array.shape != (3, 3):
        raise ValueError(f"Intrinsic matrix must have shape (3, 3), got {intrinsic_array.shape}")

    valid_mask = np.isfinite(depth_array) & (depth_array > min_depth)
    if max_depth is not None:
        valid_mask &= depth_array <= max_depth

    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float32), valid_mask

    v_coords, u_coords = np.indices(depth_array.shape)
    z_values = depth_array[valid_mask]
    x_values = (u_coords[valid_mask].astype(np.float32) - intrinsic_array[0, 2]) * z_values / intrinsic_array[0, 0]
    y_values = (v_coords[valid_mask].astype(np.float32) - intrinsic_array[1, 2]) * z_values / intrinsic_array[1, 1]
    points = np.stack([x_values, y_values, z_values], axis=-1).astype(np.float32)

    if extrinsic is not None:
        extrinsic_array = np.asarray(extrinsic, dtype=np.float32)
        if extrinsic_array.shape != (4, 4):
            raise ValueError(f"Extrinsic matrix must have shape (4, 4), got {extrinsic_array.shape}")
        homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
        points = (homogeneous @ extrinsic_array.T)[:, :3].astype(np.float32)

    return points, valid_mask


def rgbd_to_pointcloud_arrays(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray | None = None,
    extrinsic: np.ndarray | None = None,
    depth_scale: float = 1.0,
    fallback_fov_degrees: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert RGB-D arrays to point and color arrays.

    Returns:
        ``(points, colors, intrinsic_used)``. Colors are float values in
        ``[0, 1]`` with shape ``(N, 3)``.
    """

    rgb_array = normalize_rgb(rgb)
    depth_array = normalize_depth(depth)
    validate_rgb_depth_consistency(rgb_array, depth_array)

    height, width = depth_array.shape
    intrinsic_used = intrinsic
    if intrinsic_used is None:
        intrinsic_used = fallback_pinhole_intrinsics(width=width, height=height, fov_degrees=fallback_fov_degrees)

    points, valid_mask = project_depth_to_points(
        depth=depth_array,
        intrinsic=intrinsic_used,
        extrinsic=extrinsic,
        depth_scale=depth_scale,
    )
    colors = rgb_array[valid_mask].astype(np.float32) / 255.0
    return points, colors, np.asarray(intrinsic_used, dtype=np.float32)


def create_open3d_point_cloud(points: np.ndarray, colors: np.ndarray) -> Any:
    """Create an Open3D point cloud from point and color arrays."""

    o3d = _require_open3d()
    point_array = np.asarray(points, dtype=np.float64)
    color_array = np.asarray(colors, dtype=np.float64)

    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {point_array.shape}")
    if color_array.shape != point_array.shape:
        raise ValueError(f"Colors must match points shape {point_array.shape}, got {color_array.shape}")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(point_array)
    point_cloud.colors = o3d.utility.Vector3dVector(np.clip(color_array, 0.0, 1.0))
    return point_cloud


def save_point_cloud(point_cloud: Any, path: str | Path) -> Path:
    """Save an Open3D point cloud to disk."""

    o3d = _require_open3d()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = bool(o3d.io.write_point_cloud(str(output_path), point_cloud))
    if not success:
        raise RuntimeError(f"Open3D failed to write point cloud: {output_path}")
    LOGGER.info("Saved point cloud to %s", output_path)
    return output_path


def generate_and_save_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    output_path: str | Path,
    intrinsic: np.ndarray | None = None,
    extrinsic: np.ndarray | None = None,
    depth_scale: float = 1.0,
    fallback_fov_degrees: float = 60.0,
) -> Path:
    """Generate a colored Open3D point cloud from RGB-D data and save it."""

    points, colors, _ = rgbd_to_pointcloud_arrays(
        rgb=rgb,
        depth=depth,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        depth_scale=depth_scale,
        fallback_fov_degrees=fallback_fov_degrees,
    )
    if points.size == 0:
        raise ValueError("Depth image contained no valid points; point cloud was not written.")

    point_cloud = create_open3d_point_cloud(points, colors)
    return save_point_cloud(point_cloud, output_path)


def _require_open3d() -> Any:
    try:
        import open3d as o3d  # type: ignore

        return o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required for point cloud export. Install it with `pip install open3d`.") from exc

