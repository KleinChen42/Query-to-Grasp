from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.geometry.rgbd_to_pointcloud import (
    fallback_pinhole_intrinsics,
    generate_and_save_pointcloud,
    rgbd_to_pointcloud_arrays,
)


def test_rgbd_to_pointcloud_arrays_uses_valid_depth_only() -> None:
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    depth = np.array([[1.0, 0.0], [2.0, np.nan]], dtype=np.float32)
    intrinsic = fallback_pinhole_intrinsics(width=2, height=2)

    points, colors, intrinsic_used = rgbd_to_pointcloud_arrays(rgb=rgb, depth=depth, intrinsic=intrinsic)

    assert points.shape == (2, 3)
    assert colors.shape == (2, 3)
    assert intrinsic_used.shape == (3, 3)
    np.testing.assert_allclose(colors[0], [1.0, 0.0, 0.0])


def test_pointcloud_file_creation() -> None:
    pytest.importorskip("open3d")
    rgb = np.full((2, 2, 3), 127, dtype=np.uint8)
    depth = np.ones((2, 2), dtype=np.float32)
    output_path = Path("outputs/test_pointcloud_generation/pointcloud.ply")

    saved_path = generate_and_save_pointcloud(rgb=rgb, depth=depth, output_path=output_path)

    assert saved_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
