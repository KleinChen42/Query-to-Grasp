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

OPENCV_TO_OPENGL_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
DEFAULT_GRASP_WORKSPACE_XY_LIMIT = 0.20
DEFAULT_GRASP_WORKSPACE_Z_MIN = -0.02
DEFAULT_GRASP_WORKSPACE_Z_MAX = 0.08
DEFAULT_MIN_GRASP_WORKSPACE_POINTS = 20
DEFAULT_GRASP_SUPPORT_Z_PERCENTILE = 10.0
DEFAULT_GRASP_ELEVATED_Z_OFFSET = 0.015
DEFAULT_MIN_GRASP_ELEVATED_POINTS = 12
DEFAULT_GRASP_LOCAL_SUPPORT_RADIUS = 0.04
DEFAULT_GRASP_LOCAL_SUPPORT_Z_MARGIN = 0.012
DEFAULT_GRASP_COMPONENT_RADIUS = 0.03
DEFAULT_GRASP_FALLBACK_BOX_Y_SHIFT_FRACTION = 0.25


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
    grasp_world_xyz: np.ndarray | None = None
    grasp_camera_xyz: np.ndarray | None = None
    grasp_num_points: int = 0
    grasp_metadata: dict[str, Any] = field(default_factory=dict)

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
            "grasp_world_xyz": (
                None if self.grasp_world_xyz is None else np.asarray(self.grasp_world_xyz, dtype=float).tolist()
            ),
            "grasp_camera_xyz": (
                None if self.grasp_camera_xyz is None else np.asarray(self.grasp_camera_xyz, dtype=float).tolist()
            ),
            "grasp_num_points": int(self.grasp_num_points),
            "grasp_metadata": self.grasp_metadata,
        }


def lift_box_to_3d(
    rgb: np.ndarray,
    depth: np.ndarray,
    box_xyxy: np.ndarray | Sequence[float],
    intrinsic: np.ndarray | None = None,
    extrinsic: np.ndarray | None = None,
    extrinsic_source: str | None = None,
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
    min_grasp_workspace_points: int = DEFAULT_MIN_GRASP_WORKSPACE_POINTS,
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
    world_points = (
        transform_camera_points(camera_points, extrinsic, extrinsic_source=extrinsic_source)
        if extrinsic is not None
        else None
    )
    camera_center = _estimate_center(camera_points, strategy=center_strategy)
    world_center = _estimate_center(world_points, strategy=center_strategy) if world_points is not None else None
    grasp_candidate = estimate_workspace_low_z_grasp_candidate(
        camera_points=camera_points,
        world_points=world_points,
        min_points=min_grasp_workspace_points,
    )
    shifted_grasp_candidate = _estimate_shifted_grasp_candidate(
        rgb=rgb_array,
        depth=depth_array,
        original_box_bounds=(x0, y0, x1, y1),
        intrinsic=intrinsic_used,
        extrinsic=extrinsic,
        extrinsic_source=extrinsic_source,
        min_depth=min_depth,
        max_depth=max_depth,
        segmentation=segmentation,
        use_segmentation=use_segmentation,
        segmentation_id=chosen_segmentation_id,
        background_ids=background_segmentation_ids,
        min_grasp_workspace_points=min_grasp_workspace_points,
        original_grasp_candidate=grasp_candidate,
    )
    if shifted_grasp_candidate is not None:
        grasp_candidate = shifted_grasp_candidate

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
            "extrinsics_source": extrinsic_source or ("provided" if extrinsic is not None else "missing"),
            "camera_frame_conversion": camera_frame_conversion_for_source(extrinsic_source),
            "center_strategy": center_strategy,
        },
        grasp_world_xyz=grasp_candidate["grasp_world_xyz"],
        grasp_camera_xyz=grasp_candidate["grasp_camera_xyz"],
        grasp_num_points=grasp_candidate["grasp_num_points"],
        grasp_metadata=grasp_candidate["grasp_metadata"],
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


def transform_camera_points(
    points: np.ndarray,
    extrinsic: np.ndarray,
    extrinsic_source: str | None = None,
) -> np.ndarray:
    """Apply a camera-to-world transform to camera-frame points."""

    extrinsic_array = normalize_camera_to_world_extrinsic(extrinsic, extrinsic_source=extrinsic_source)
    homogeneous = np.concatenate([points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    return (homogeneous @ extrinsic_array.T)[:, :3].astype(np.float32)


def normalize_camera_to_world_extrinsic(
    extrinsic: np.ndarray,
    extrinsic_source: str | None = None,
) -> np.ndarray:
    """Normalize known ManiSkill camera-to-world matrices to OpenCV point inputs.

    RGB-D lifting projects pixels into OpenCV-style camera coordinates
    (x right, y down, z forward). ManiSkill's ``cam2world_gl`` matrix expects
    OpenGL-style camera coordinates (x right, y up, z backward), so a fixed
    camera-frame conversion is needed before applying that transform.
    """

    extrinsic_array = np.asarray(extrinsic, dtype=np.float32)
    if extrinsic_array.shape != (4, 4):
        raise ValueError(f"Extrinsic matrix must have shape (4, 4), got {extrinsic_array.shape}")
    if camera_frame_conversion_for_source(extrinsic_source) == "opencv_to_opengl":
        return (extrinsic_array @ OPENCV_TO_OPENGL_CAMERA).astype(np.float32)
    return extrinsic_array


def camera_frame_conversion_for_source(extrinsic_source: str | None) -> str:
    """Return the camera-frame conversion implied by an extrinsic source key."""

    if extrinsic_source and "cam2world_gl" in extrinsic_source.lower():
        return "opencv_to_opengl"
    return "none"


def estimate_workspace_low_z_grasp_candidate(
    camera_points: np.ndarray,
    world_points: np.ndarray | None,
    min_points: int = DEFAULT_MIN_GRASP_WORKSPACE_POINTS,
    xy_limit: float = DEFAULT_GRASP_WORKSPACE_XY_LIMIT,
    z_min: float = DEFAULT_GRASP_WORKSPACE_Z_MIN,
    z_max: float = DEFAULT_GRASP_WORKSPACE_Z_MAX,
    support_z_percentile: float = DEFAULT_GRASP_SUPPORT_Z_PERCENTILE,
    elevated_z_offset: float = DEFAULT_GRASP_ELEVATED_Z_OFFSET,
    min_elevated_points: int = DEFAULT_MIN_GRASP_ELEVATED_POINTS,
    local_support_radius: float = DEFAULT_GRASP_LOCAL_SUPPORT_RADIUS,
    local_support_z_margin: float = DEFAULT_GRASP_LOCAL_SUPPORT_Z_MARGIN,
    component_radius: float = DEFAULT_GRASP_COMPONENT_RADIUS,
) -> dict[str, Any]:
    """Estimate a conservative grasp point from object-like geometry in the workspace."""

    metadata = {
        "strategy": "workspace_elevated_xy_low_z",
        "workspace_bounds": {
            "abs_x_max": float(xy_limit),
            "abs_y_max": float(xy_limit),
            "z_min": float(z_min),
            "z_max": float(z_max),
        },
        "min_points": int(min_points),
        "support_z_percentile": float(support_z_percentile),
        "elevated_z_offset": float(elevated_z_offset),
        "min_elevated_points": int(min_elevated_points),
        "local_support_radius": float(local_support_radius),
        "local_support_z_margin": float(local_support_z_margin),
        "component_radius": float(component_radius),
        "applied": False,
        "reason": None,
    }
    if world_points is None or world_points.size == 0:
        metadata["reason"] = "missing_world_points"
        return {
            "grasp_world_xyz": None,
            "grasp_camera_xyz": None,
            "grasp_num_points": 0,
            "grasp_metadata": metadata,
        }

    world_array = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
    camera_array = np.asarray(camera_points, dtype=np.float32).reshape(-1, 3)
    mask = (
        (np.abs(world_array[:, 0]) <= float(xy_limit))
        & (np.abs(world_array[:, 1]) <= float(xy_limit))
        & (world_array[:, 2] >= float(z_min))
        & (world_array[:, 2] <= float(z_max))
    )
    num_workspace_points = int(np.count_nonzero(mask))
    metadata["num_workspace_points"] = num_workspace_points
    if num_workspace_points < int(min_points):
        metadata["reason"] = "too_few_workspace_points"
        return {
            "grasp_world_xyz": None,
            "grasp_camera_xyz": None,
            "grasp_num_points": num_workspace_points,
            "grasp_metadata": metadata,
        }

    workspace_world = world_array[mask]
    workspace_camera = camera_array[mask]
    support_z = float(np.percentile(workspace_world[:, 2], float(support_z_percentile)))
    elevated_mask = workspace_world[:, 2] >= support_z + float(elevated_z_offset)
    elevated_count = int(np.count_nonzero(elevated_mask))
    metadata["support_z"] = support_z
    metadata["elevated_point_count"] = elevated_count
    if elevated_count >= int(min_elevated_points):
        elevated_world = workspace_world[elevated_mask]
        components = _cluster_xy_radius_components(elevated_world[:, :2], radius=float(component_radius))
        component_stats = _summarize_xy_components(elevated_world[:, :2], components)
        selected_component = _select_xy_component(component_stats, min_size=int(min_elevated_points))
        metadata["component_count"] = len(components)
        metadata["component_sizes"] = [int(stat["size"]) for stat in component_stats[:8]]
        if selected_component is not None:
            component_indices = components[int(selected_component["component_index"])]
            selected_world = elevated_world[component_indices]
            elevated_xy = np.median(selected_world[:, :2], axis=0).astype(np.float32)
            selected_size = int(selected_component["size"])
            metadata["component_selection_strategy"] = "largest_component_then_low_spread"
            metadata["component_fallback_reason"] = None
            metadata["selected_component_size"] = selected_size
            metadata["selected_component_xy"] = [float(elevated_xy[0]), float(elevated_xy[1])]
            metadata["selected_component_spread"] = float(selected_component["spread"])
        else:
            elevated_xy = np.median(elevated_world[:, :2], axis=0).astype(np.float32)
            selected_size = elevated_count
            metadata["component_selection_strategy"] = "global_elevated_median"
            metadata["component_fallback_reason"] = "no_component_met_min_size"
            metadata["selected_component_size"] = 0
            metadata["selected_component_xy"] = [float(elevated_xy[0]), float(elevated_xy[1])]
            metadata["selected_component_spread"] = None
        xy_delta = workspace_world[:, :2] - elevated_xy.reshape(1, 2)
        local_support_mask = (
            (np.linalg.norm(xy_delta, axis=1) <= float(local_support_radius))
            & (workspace_world[:, 2] <= support_z + float(local_support_z_margin))
        )
        local_support_count = int(np.count_nonzero(local_support_mask))
        metadata["local_support_point_count"] = local_support_count
        grasp_z = (
            float(np.median(workspace_world[local_support_mask, 2]))
            if local_support_count > 0
            else support_z
        )
        metadata["applied"] = True
        metadata["reason"] = "component_elevated_xy_with_local_support_z"
        metadata["fallback_reason"] = None if local_support_count > 0 else "no_local_support_points"
        grasp_world = np.asarray([elevated_xy[0], elevated_xy[1], grasp_z], dtype=np.float32)
        return {
            "grasp_world_xyz": grasp_world,
            "grasp_camera_xyz": _nearest_camera_point_for_world_point(
                grasp_world=grasp_world,
                world_points=workspace_world,
                camera_points=workspace_camera,
            ),
            "grasp_num_points": selected_size,
            "grasp_metadata": metadata,
        }

    metadata["applied"] = True
    metadata["reason"] = "workspace_filter_passed"
    metadata["fallback_reason"] = "too_few_elevated_points"
    metadata["local_support_point_count"] = 0
    workspace_components = _cluster_xy_radius_components(workspace_world[:, :2], radius=float(component_radius))
    workspace_component_stats = _summarize_xy_components(workspace_world[:, :2], workspace_components)
    selected_workspace_component = _select_xy_component(workspace_component_stats, min_size=int(min_points))
    metadata["component_count"] = len(workspace_components)
    metadata["component_sizes"] = [int(stat["size"]) for stat in workspace_component_stats[:8]]
    if selected_workspace_component is not None:
        component_indices = workspace_components[int(selected_workspace_component["component_index"])]
        selected_world = workspace_world[component_indices]
        selected_camera = workspace_camera[component_indices]
        grasp_world = np.median(selected_world, axis=0).astype(np.float32)
        grasp_camera = np.median(selected_camera, axis=0).astype(np.float32)
        metadata["component_selection_strategy"] = "largest_workspace_component_then_low_spread"
        metadata["component_fallback_reason"] = "too_few_elevated_points"
        metadata["selected_component_size"] = int(selected_workspace_component["size"])
        metadata["selected_component_xy"] = [float(grasp_world[0]), float(grasp_world[1])]
        metadata["selected_component_spread"] = float(selected_workspace_component["spread"])
        return {
            "grasp_world_xyz": grasp_world,
            "grasp_camera_xyz": grasp_camera,
            "grasp_num_points": int(selected_workspace_component["size"]),
            "grasp_metadata": metadata,
        }
    metadata["component_selection_strategy"] = "global_workspace_median"
    metadata["component_fallback_reason"] = "no_workspace_component_met_min_size"
    metadata["selected_component_size"] = 0
    metadata["selected_component_xy"] = None
    metadata["selected_component_spread"] = None
    return {
        "grasp_world_xyz": np.median(workspace_world, axis=0).astype(np.float32),
        "grasp_camera_xyz": np.median(workspace_camera, axis=0).astype(np.float32),
        "grasp_num_points": num_workspace_points,
        "grasp_metadata": metadata,
    }


def _estimate_shifted_grasp_candidate(
    rgb: np.ndarray,
    depth: np.ndarray,
    original_box_bounds: tuple[int, int, int, int],
    intrinsic: np.ndarray,
    extrinsic: np.ndarray | None,
    extrinsic_source: str | None,
    min_depth: float,
    max_depth: float | None,
    segmentation: np.ndarray | None,
    use_segmentation: bool,
    segmentation_id: int | None,
    background_ids: Sequence[int],
    min_grasp_workspace_points: int,
    original_grasp_candidate: dict[str, Any],
    y_shift_fraction: float = DEFAULT_GRASP_FALLBACK_BOX_Y_SHIFT_FRACTION,
) -> dict[str, Any] | None:
    metadata = dict(original_grasp_candidate.get("grasp_metadata") or {})
    if metadata.get("elevated_point_count", 0) >= DEFAULT_MIN_GRASP_ELEVATED_POINTS:
        return None
    if extrinsic is None:
        return None

    x0, y0, x1, y1 = original_box_bounds
    height, width = depth.shape
    box_height = max(1, y1 - y0)
    shift = int(round(float(y_shift_fraction) * box_height))
    if shift <= 0:
        return None
    shifted_bounds = (x0, min(height, y0 + shift), x1, min(height, y1 + shift))
    if shifted_bounds == original_box_bounds or shifted_bounds[3] <= shifted_bounds[1]:
        return None

    shifted_valid_mask = _valid_depth_mask_for_bounds(
        depth=depth,
        bounds=shifted_bounds,
        min_depth=min_depth,
        max_depth=max_depth,
        segmentation=segmentation,
        use_segmentation=use_segmentation,
        segmentation_id=segmentation_id,
        background_ids=background_ids,
    )
    if int(np.count_nonzero(shifted_valid_mask)) == 0:
        return None
    shifted_camera_points, _ = _project_crop_to_camera_points(
        rgb=rgb,
        depth=depth,
        valid_mask=shifted_valid_mask,
        box_bounds=shifted_bounds,
        intrinsic=intrinsic,
    )
    shifted_world_points = transform_camera_points(
        shifted_camera_points,
        extrinsic,
        extrinsic_source=extrinsic_source,
    )
    shifted_candidate = estimate_workspace_low_z_grasp_candidate(
        camera_points=shifted_camera_points,
        world_points=shifted_world_points,
        min_points=min_grasp_workspace_points,
    )
    shifted_metadata = dict(shifted_candidate.get("grasp_metadata") or {})
    if shifted_candidate.get("grasp_world_xyz") is None:
        return None
    if shifted_metadata.get("elevated_point_count", 0) < DEFAULT_MIN_GRASP_ELEVATED_POINTS:
        return None

    shifted_metadata["source_box"] = "y_shifted_grasp_fallback"
    shifted_metadata["source_box_y_shift_fraction"] = float(y_shift_fraction)
    shifted_metadata["source_box_bounds_xyxy_exclusive"] = list(shifted_bounds)
    shifted_metadata["original_box_bounds_xyxy_exclusive"] = [x0, y0, x1, y1]
    shifted_metadata["original_grasp_reason"] = metadata.get("reason")
    shifted_metadata["original_elevated_point_count"] = metadata.get("elevated_point_count")
    shifted_metadata["original_num_workspace_points"] = metadata.get("num_workspace_points")
    shifted_candidate["grasp_metadata"] = shifted_metadata
    return shifted_candidate


def _valid_depth_mask_for_bounds(
    depth: np.ndarray,
    bounds: tuple[int, int, int, int],
    min_depth: float,
    max_depth: float | None,
    segmentation: np.ndarray | None,
    use_segmentation: bool,
    segmentation_id: int | None,
    background_ids: Sequence[int],
) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    depth_crop = depth[y0:y1, x0:x1]
    valid_mask = np.isfinite(depth_crop) & (depth_crop > min_depth)
    if max_depth is not None:
        valid_mask &= depth_crop <= max_depth
    if use_segmentation and segmentation is not None:
        segmentation_array = np.squeeze(np.asarray(segmentation))
        segmentation_crop = segmentation_array[y0:y1, x0:x1]
        chosen_id = segmentation_id
        if chosen_id is None:
            chosen_id = _choose_segmentation_id(
                segmentation_crop,
                valid_mask=valid_mask,
                background_ids=background_ids,
            )
        if chosen_id is not None:
            valid_mask &= segmentation_crop == chosen_id
    return valid_mask


def _nearest_camera_point_for_world_point(
    grasp_world: np.ndarray,
    world_points: np.ndarray,
    camera_points: np.ndarray,
) -> np.ndarray:
    distances = np.linalg.norm(world_points.astype(np.float32) - grasp_world.reshape(1, 3).astype(np.float32), axis=1)
    return np.asarray(camera_points[int(np.argmin(distances))], dtype=np.float32)


def _cluster_xy_radius_components(points_xy: np.ndarray, radius: float) -> list[np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.size == 0:
        return []
    radius = float(radius)
    if radius <= 0.0:
        return [np.asarray([index], dtype=np.int64) for index in range(points.shape[0])]

    cell_size = radius
    cells: dict[tuple[int, int], list[int]] = {}
    for index, point in enumerate(points):
        cell = (int(np.floor(float(point[0]) / cell_size)), int(np.floor(float(point[1]) / cell_size)))
        cells.setdefault(cell, []).append(index)

    visited = np.zeros(points.shape[0], dtype=bool)
    components: list[np.ndarray] = []
    radius_sq = radius * radius
    for start in range(points.shape[0]):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            cell = (
                int(np.floor(float(points[current, 0]) / cell_size)),
                int(np.floor(float(points[current, 1]) / cell_size)),
            )
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for neighbor in cells.get((cell[0] + dx, cell[1] + dy), []):
                        if visited[neighbor]:
                            continue
                        delta = points[neighbor] - points[current]
                        if float(np.dot(delta, delta)) <= radius_sq:
                            visited[neighbor] = True
                            stack.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    components.sort(key=lambda item: (-int(item.size), int(item[0]) if item.size else 0))
    return components


def _summarize_xy_components(points_xy: np.ndarray, components: list[np.ndarray]) -> list[dict[str, Any]]:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    summaries: list[dict[str, Any]] = []
    for component_index, indices in enumerate(components):
        component_points = points[indices]
        median_xy = np.median(component_points, axis=0).astype(np.float32)
        spread = float(np.percentile(np.linalg.norm(component_points - median_xy.reshape(1, 2), axis=1), 90.0))
        summaries.append(
            {
                "component_index": int(component_index),
                "size": int(indices.size),
                "median_xy": [float(median_xy[0]), float(median_xy[1])],
                "spread": spread,
            }
        )
    summaries.sort(key=lambda item: (-int(item["size"]), float(item["spread"]), int(item["component_index"])))
    return summaries


def _select_xy_component(component_stats: list[dict[str, Any]], min_size: int) -> dict[str, Any] | None:
    valid = [stat for stat in component_stats if int(stat["size"]) >= int(min_size)]
    if not valid:
        return None
    return valid[0]


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
