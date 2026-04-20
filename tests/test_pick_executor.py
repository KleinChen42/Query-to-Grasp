from __future__ import annotations

import json

import numpy as np

from src.manipulation.pick_executor import SafePlaceholderPickExecutor


def test_placeholder_executor_accepts_valid_target() -> None:
    executor = SafePlaceholderPickExecutor(env=None)

    result = executor.execute(np.array([0.1, 0.2, 0.3], dtype=np.float32))

    assert result["success"] is False
    assert result["stage"] == "placeholder_not_executed"
    np.testing.assert_allclose(result["target_xyz"], [0.1, 0.2, 0.3])
    assert result["trajectory_summary"]["num_env_steps"] == 0
    assert result["metadata"]["low_level_control_verified"] is False


def test_placeholder_executor_rejects_invalid_shape() -> None:
    executor = SafePlaceholderPickExecutor(env=None)

    result = executor.execute(np.array([[0.1, 0.2, 0.3]], dtype=np.float32))

    assert result["success"] is False
    assert result["stage"] == "validation_failed"
    assert "shape (3,)" in result["message"]


def test_placeholder_executor_rejects_nan_target() -> None:
    executor = SafePlaceholderPickExecutor(env=None)

    result = executor.execute(np.array([0.1, np.nan, 0.3], dtype=np.float32))

    assert result["success"] is False
    assert result["stage"] == "validation_failed"
    assert "finite" in result["message"]


def test_pick_result_dict_is_json_serializable() -> None:
    executor = SafePlaceholderPickExecutor(env=None)
    result = executor.execute(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    encoded = json.dumps(result)

    assert "placeholder_not_executed" in encoded
