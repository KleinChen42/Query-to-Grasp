from __future__ import annotations

import numpy as np
import pytest

from src.env.camera_utils import (
    extract_observation_frame,
    flatten_observation_keys,
    validate_rgb_depth_consistency,
)


def test_extracts_maniskill_like_observation() -> None:
    observation = {
        "sensor_data": {
            "base_camera": {
                "rgb": np.zeros((1, 4, 5, 3), dtype=np.uint8),
                "depth": np.ones((1, 4, 5, 1), dtype=np.float32),
                "segmentation": np.arange(20, dtype=np.int32).reshape(1, 4, 5, 1),
            }
        },
        "sensor_param": {
            "base_camera": {
                "intrinsic_cv": np.eye(3, dtype=np.float32),
                "cam2world_gl": np.eye(4, dtype=np.float32),
            }
        },
    }

    frame = extract_observation_frame(observation, camera_name="base_camera")

    assert frame.rgb is not None
    assert frame.depth is not None
    assert frame.segmentation is not None
    assert frame.rgb.shape == (4, 5, 3)
    assert frame.depth.shape == (4, 5)
    assert frame.segmentation.shape == (4, 5)
    assert frame.camera_info.intrinsic is not None
    assert frame.camera_info.extrinsic is not None
    validate_rgb_depth_consistency(frame.rgb, frame.depth)


def test_flatten_observation_keys_lists_nested_leaves() -> None:
    keys = flatten_observation_keys({"a": {"b": np.zeros(1)}, "c": 3})

    assert keys == ["a.b", "c"]


def test_rgb_depth_consistency_rejects_mismatched_shapes() -> None:
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="spatial shapes differ"):
        validate_rgb_depth_consistency(rgb, depth)

