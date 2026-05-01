from __future__ import annotations

import json

import numpy as np

from src.manipulation.pick_executor import (
    SafePlaceholderPickExecutor,
    SimulatedPickPlaceExecutor,
    SimulatedTopDownPickExecutor,
)


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


def test_simulated_topdown_rejects_invalid_target_without_stepping() -> None:
    env = _FakeDeltaPosEnv()
    executor = SimulatedTopDownPickExecutor(env=env)

    result = executor.execute(np.array([0.1, np.nan, 0.3], dtype=np.float32))

    assert result["success"] is False
    assert result["stage"] == "validation_failed"
    assert result["grasp_attempted"] is False
    assert env.num_steps == 0


def test_simulated_topdown_builds_four_dimensional_actions() -> None:
    env = _FakeDeltaPosEnv()
    executor = SimulatedTopDownPickExecutor(
        env=env,
        move_above_steps=2,
        descend_steps=2,
        close_steps=2,
        lift_steps=2,
    )

    result = executor.execute(np.array([0.0, 0.0, 0.05], dtype=np.float32))

    assert result["grasp_attempted"] is True
    assert result["trajectory_summary"]["num_env_steps"] > 0
    assert result["trajectory_summary"]["executed_stages"] == [
        "move_above_target",
        "descend",
        "close_gripper",
        "lift",
    ]
    assert all(action.shape == (4,) for action in env.actions)
    assert any(float(action[3]) == -1.0 for action in env.actions)
    json.dumps(result)


def test_simulated_topdown_invokes_step_callback_without_changing_result() -> None:
    env = _FakeDeltaPosEnv()
    calls = []
    executor = SimulatedTopDownPickExecutor(
        env=env,
        step_callback=lambda **kwargs: calls.append(kwargs),
        move_above_steps=1,
        descend_steps=1,
        close_steps=1,
        lift_steps=1,
    )

    result = executor.execute(np.array([0.0, 0.0, 0.05], dtype=np.float32))

    assert result["grasp_attempted"] is True
    assert len(calls) == result["trajectory_summary"]["num_env_steps"]
    assert {call["stage"] for call in calls} == {"move_above_target", "descend", "close_gripper", "lift"}
    assert all(call["action"].shape == (4,) for call in calls)


def test_simulated_topdown_accepts_task_specific_grasp_flag() -> None:
    env = _FakeDeltaPosEnv(grasp_info_key="is_cubeA_grasped")
    executor = SimulatedTopDownPickExecutor(
        env=env,
        move_above_steps=2,
        descend_steps=2,
        close_steps=2,
        lift_steps=2,
    )

    result = executor.execute(np.array([0.0, 0.0, 0.05], dtype=np.float32))

    assert result["pick_success"] is True
    assert result["is_grasped"] is True
    assert result["stage"] == "success"
    assert result["trajectory_summary"]["grasp_info_keys"] == ["is_cubeA_grasped"]


def test_simulated_topdown_ignores_unrelated_info_flags() -> None:
    env = _FakeDeltaPosEnv(grasp_info_key=None)
    executor = SimulatedTopDownPickExecutor(
        env=env,
        move_above_steps=2,
        descend_steps=2,
        close_steps=2,
        lift_steps=2,
    )

    result = executor.execute(np.array([0.0, 0.0, 0.05], dtype=np.float32))

    assert result["pick_success"] is False
    assert result["is_grasped"] is False
    assert result["stage"] == "grasp_not_confirmed"
    assert result["trajectory_summary"]["grasp_info_keys"] == []


def test_simulated_pick_place_rejects_invalid_pick_without_stepping() -> None:
    env = _FakeDeltaPosEnv()
    executor = SimulatedPickPlaceExecutor(env=env)

    result = executor.execute(
        pick_xyz=np.array([0.1, np.nan, 0.3], dtype=np.float32),
        place_xyz=np.array([0.0, 0.0, 0.02], dtype=np.float32),
    )

    assert result["success"] is False
    assert result["stage"] == "validation_failed"
    assert result["grasp_attempted"] is False
    assert result["place_attempted"] is False
    assert env.num_steps == 0


def test_simulated_pick_place_rejects_invalid_place_without_stepping() -> None:
    env = _FakeDeltaPosEnv()
    executor = SimulatedPickPlaceExecutor(env=env)

    result = executor.execute(
        pick_xyz=np.array([0.0, 0.0, 0.02], dtype=np.float32),
        place_xyz=np.array([0.1, np.nan, 0.3], dtype=np.float32),
    )

    assert result["success"] is False
    assert result["stage"] == "validation_failed"
    assert result["place_attempted"] is False
    assert env.num_steps == 0


def test_simulated_pick_place_builds_actions_and_opens_at_place() -> None:
    env = _FakeDeltaPosEnv()
    executor = SimulatedPickPlaceExecutor(
        env=env,
        move_above_steps=2,
        descend_steps=2,
        close_steps=2,
        lift_steps=2,
        move_to_place_steps=2,
        place_descend_steps=2,
        open_steps=2,
        settle_steps=2,
    )

    result = executor.execute(
        pick_xyz=np.array([0.0, 0.0, 0.02], dtype=np.float32),
        place_xyz=np.array([0.08, 0.0, 0.02], dtype=np.float32),
    )

    assert result["grasp_attempted"] is True
    assert result["place_attempted"] is True
    assert result["pick_success"] is True
    assert result["place_success"] is True
    assert result["task_success"] is False
    assert result["stage"] == "placed_not_task_success"
    assert result["trajectory_summary"]["executed_stages"] == [
        "move_above_pick",
        "descend_to_pick",
        "close_gripper",
        "lift_from_pick",
        "move_above_place",
        "descend_to_place",
        "open_gripper",
        "settle",
    ]
    assert all(action.shape == (4,) for action in env.actions)
    assert any(float(action[3]) == -1.0 for action in env.actions)
    assert env.actions[-1][3] == 1.0
    json.dumps(result)


class _FakeActionSpace:
    shape = (4,)


class _FakeTcp:
    def __init__(self) -> None:
        self.pose = type("Pose", (), {"p": np.array([[0.0, 0.0, 0.15]], dtype=np.float32)})()


class _FakeAgent:
    def __init__(self) -> None:
        self.tcp = _FakeTcp()


class _FakeDeltaPosEnv:
    action_space = _FakeActionSpace()
    control_mode = "pd_ee_delta_pos"

    def __init__(self, grasp_info_key: str | None = "is_grasped") -> None:
        self.unwrapped = self
        self.agent = _FakeAgent()
        self.actions: list[np.ndarray] = []
        self.grasp_info_key = grasp_info_key
        self.num_steps = 0

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        self.actions.append(action.copy())
        self.num_steps += 1
        current = self.agent.tcp.pose.p.reshape(3)
        self.agent.tcp.pose.p = (current + action[:3] * 0.04).reshape(1, 3).astype(np.float32)
        info = {
            "success": False,
            "elapsed_steps": self.num_steps,
            "is_obj_placed": True,
            "is_robot_static": True,
        }
        if self.grasp_info_key is not None:
            info[self.grasp_info_key] = self.num_steps >= 5 and action[3] < 0
        observation = {
            "sensor_data": {
                "base_camera": {
                    "rgb": np.zeros((1, 4, 5, 3), dtype=np.uint8),
                    "depth": np.ones((1, 4, 5, 1), dtype=np.float32),
                }
            },
            "sensor_param": {
                "base_camera": {
                    "intrinsic_cv": np.eye(3, dtype=np.float32),
                    "cam2world_gl": np.eye(4, dtype=np.float32),
                }
            },
        }
        return observation, 0.0, False, False, info
