"""Minimal pick execution interface for the single-view baseline.

This module deliberately defaults to a safe placeholder executor. ManiSkill
control modes and robot action schemas vary by task/version, so the default
does not send low-level actions until that API is verified in a later step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class PickResult:
    """Structured result for a pick attempt."""

    success: bool
    stage: str
    target_xyz: list[float]
    message: str
    trajectory_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    grasp_attempted: bool = False
    pick_success: bool | None = None
    task_success: bool | None = None
    is_grasped: bool | None = None

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""

        pick_success = self.success if self.pick_success is None else bool(self.pick_success)
        return {
            "success": bool(self.success),
            "pick_success": pick_success,
            "grasp_attempted": bool(self.grasp_attempted),
            "task_success": self.task_success,
            "is_grasped": self.is_grasped,
            "stage": self.stage,
            "target_xyz": [float(value) for value in self.target_xyz],
            "message": self.message,
            "trajectory_summary": self.trajectory_summary,
            "metadata": self.metadata,
        }


class SafePlaceholderPickExecutor:
    """Validate a target and return a stable placeholder pick result."""

    def __init__(self, env: Any | None = None) -> None:
        self.env = env

    def execute(self, target_xyz: np.ndarray) -> dict[str, Any]:
        """Validate ``target_xyz`` and return a conservative pick result.

        No simulator actions are sent. This is the integration point for a
        future verified ManiSkill action sequence.
        """

        validation = validate_target_xyz(target_xyz)
        if validation is not None:
            LOGGER.warning("Rejecting pick target: %s", validation.message)
            return validation.to_json_dict()

        target = np.asarray(target_xyz, dtype=np.float32).reshape(3)
        LOGGER.info("Validated pick target %s; placeholder executor will not send robot actions.", target.tolist())
        return PickResult(
            success=False,
            stage="placeholder_not_executed",
            target_xyz=target.astype(float).tolist(),
            message=(
                "Target validated, but no low-level ManiSkill pick action was executed. "
                "Real action sequencing is intentionally left unverified in this placeholder."
            ),
            trajectory_summary={
                "planned_stages": ["move_above_target", "descend", "close_gripper", "lift"],
                "executed_stages": [],
                "num_env_steps": 0,
            },
            metadata={
                "executor": "SafePlaceholderPickExecutor",
                "low_level_control_verified": False,
                "env_type": type(self.env).__name__ if self.env is not None else None,
            },
            grasp_attempted=False,
            pick_success=False,
            task_success=None,
            is_grasped=None,
        ).to_json_dict()


class SimulatedTopDownPickExecutor:
    """Minimal top-down simulated grasp executor for ManiSkill smoke tests."""

    def __init__(
        self,
        env: Any,
        approach_height: float = 0.12,
        grasp_height: float = 0.025,
        lift_height: float = 0.18,
        position_scale: float = 0.04,
        move_above_steps: int = 40,
        descend_steps: int = 30,
        close_steps: int = 20,
        lift_steps: int = 40,
        goal_tolerance: float = 0.015,
    ) -> None:
        self.env = env
        self.approach_height = float(approach_height)
        self.grasp_height = float(grasp_height)
        self.lift_height = float(lift_height)
        self.position_scale = float(position_scale)
        self.move_above_steps = int(move_above_steps)
        self.descend_steps = int(descend_steps)
        self.close_steps = int(close_steps)
        self.lift_steps = int(lift_steps)
        self.goal_tolerance = float(goal_tolerance)

    def execute(self, target_xyz: np.ndarray) -> dict[str, Any]:
        """Run a simple top-down pick attempt from a world-frame target."""

        validation = validate_target_xyz(target_xyz)
        if validation is not None:
            return validation.to_json_dict()

        unsupported = self._validate_action_space()
        if unsupported is not None:
            return unsupported.to_json_dict()

        target = np.asarray(target_xyz, dtype=np.float32).reshape(3)
        above = np.array([target[0], target[1], target[2] + self.approach_height], dtype=np.float32)
        grasp = np.array([target[0], target[1], target[2] + self.grasp_height], dtype=np.float32)
        lift = np.array([target[0], target[1], target[2] + self.lift_height], dtype=np.float32)

        executed_stages: list[str] = []
        all_infos: list[dict[str, Any]] = []
        is_grasped = False
        task_success = False
        try:
            stage_infos = self._step_to_goal("move_above_target", above, gripper=1.0, max_steps=self.move_above_steps)
            executed_stages.append("move_above_target")
            all_infos.extend(stage_infos)

            stage_infos = self._step_to_goal("descend", grasp, gripper=1.0, max_steps=self.descend_steps)
            executed_stages.append("descend")
            all_infos.extend(stage_infos)

            stage_infos = self._hold_action("close_gripper", goal_xyz=grasp, gripper=-1.0, max_steps=self.close_steps)
            executed_stages.append("close_gripper")
            all_infos.extend(stage_infos)
            is_grasped = any(_info_any_grasped(info) for info in stage_infos)

            stage_infos = self._step_to_goal("lift", lift, gripper=-1.0, max_steps=self.lift_steps)
            executed_stages.append("lift")
            all_infos.extend(stage_infos)
            is_grasped = is_grasped or any(_info_any_grasped(info) for info in stage_infos)
            task_success = any(_info_bool(info, "success") for info in all_infos)
        except Exception as exc:
            LOGGER.exception("Simulated top-down pick failed during stage execution.")
            return PickResult(
                success=False,
                stage="execution_failed",
                target_xyz=target.astype(float).tolist(),
                message=f"Simulated top-down pick failed: {exc}",
                trajectory_summary={
                    "planned_stages": ["move_above_target", "descend", "close_gripper", "lift"],
                    "executed_stages": executed_stages,
                    "num_env_steps": len(all_infos),
                },
                metadata={"executor": "SimulatedTopDownPickExecutor", "env_type": type(self.env).__name__},
                grasp_attempted=True,
                pick_success=False,
                task_success=bool(task_success),
                is_grasped=bool(is_grasped),
            ).to_json_dict()

        final_tcp_xyz = self._current_tcp_xyz()
        pick_success = bool(is_grasped)
        return PickResult(
            success=pick_success,
            stage="success" if pick_success else "grasp_not_confirmed",
            target_xyz=target.astype(float).tolist(),
            message=(
                "Simulated top-down pick executed and grasp was detected."
                if pick_success
                else "Simulated top-down pick executed, but ManiSkill did not report a grasp."
            ),
            trajectory_summary={
                "planned_stages": ["move_above_target", "descend", "close_gripper", "lift"],
                "executed_stages": executed_stages,
                "num_env_steps": len(all_infos),
                "final_tcp_xyz": None if final_tcp_xyz is None else final_tcp_xyz.astype(float).tolist(),
                "final_info": all_infos[-1] if all_infos else {},
                "stage_step_counts": {
                    "move_above_target": self.move_above_steps,
                    "descend": self.descend_steps,
                    "close_gripper": self.close_steps,
                    "lift": self.lift_steps,
                },
                "grasp_info_keys": _grasp_info_keys(all_infos),
            },
            metadata={
                "executor": "SimulatedTopDownPickExecutor",
                "control_mode": getattr(self._unwrapped_env(), "control_mode", None),
                "action_convention": "[dx, dy, dz, gripper]; gripper +1=open, -1=close",
                "env_type": type(self.env).__name__,
            },
            grasp_attempted=True,
            pick_success=pick_success,
            task_success=bool(task_success),
            is_grasped=bool(is_grasped),
        ).to_json_dict()

    def _validate_action_space(self) -> PickResult | None:
        action_space = getattr(self.env, "action_space", None)
        shape = getattr(action_space, "shape", None)
        if tuple(shape or ()) != (4,):
            return PickResult(
                success=False,
                stage="unsupported_control_mode",
                target_xyz=[],
                message=(
                    "SimulatedTopDownPickExecutor requires ManiSkill control_mode=pd_ee_delta_pos "
                    f"with action shape (4,), got {shape}."
                ),
                metadata={
                    "executor": "SimulatedTopDownPickExecutor",
                    "action_space": repr(action_space),
                    "control_mode": getattr(self._unwrapped_env(), "control_mode", None),
                },
                grasp_attempted=False,
                pick_success=False,
                task_success=None,
                is_grasped=None,
            )
        return None

    def _step_to_goal(self, stage: str, goal_xyz: np.ndarray, gripper: float, max_steps: int) -> list[dict[str, Any]]:
        infos: list[dict[str, Any]] = []
        for _ in range(max_steps):
            current = self._current_tcp_xyz()
            if current is None:
                raise RuntimeError("Could not read env.unwrapped.agent.tcp.pose.p.")
            delta = goal_xyz.astype(np.float32) - current.astype(np.float32)
            action = self._build_action(delta, gripper=gripper)
            info = self._step(action)
            info["stage"] = stage
            infos.append(info)
            if float(np.linalg.norm(delta)) <= self.goal_tolerance:
                break
        return infos

    def _hold_action(self, stage: str, goal_xyz: np.ndarray, gripper: float, max_steps: int) -> list[dict[str, Any]]:
        infos: list[dict[str, Any]] = []
        for _ in range(max_steps):
            current = self._current_tcp_xyz()
            if current is None:
                raise RuntimeError("Could not read env.unwrapped.agent.tcp.pose.p.")
            delta = goal_xyz.astype(np.float32) - current.astype(np.float32)
            action = self._build_action(delta, gripper=gripper)
            info = self._step(action)
            info["stage"] = stage
            infos.append(info)
        return infos

    def _build_action(self, delta_xyz: np.ndarray, gripper: float) -> np.ndarray:
        scaled_delta = np.clip(delta_xyz / self.position_scale, -1.0, 1.0)
        return np.asarray([scaled_delta[0], scaled_delta[1], scaled_delta[2], gripper], dtype=np.float32)

    def _step(self, action: np.ndarray) -> dict[str, Any]:
        result = self.env.step(action)
        if not isinstance(result, tuple):
            raise RuntimeError(f"env.step returned non-tuple result {type(result).__name__}.")
        if len(result) == 5:
            _, _, _, _, info = result
        elif len(result) == 4:
            _, _, _, info = result
        else:
            raise RuntimeError(f"env.step returned {len(result)} values; expected 4 or 5.")
        return _json_safe_info(info)

    def _current_tcp_xyz(self) -> np.ndarray | None:
        env = self._unwrapped_env()
        agent = getattr(env, "agent", None)
        tcp = getattr(agent, "tcp", None) if agent is not None else None
        pose = getattr(tcp, "pose", None) if tcp is not None else None
        position = getattr(pose, "p", None) if pose is not None else None
        if position is None:
            return None
        array = _to_numpy(position)
        if array is None:
            return None
        return np.asarray(array, dtype=np.float32).reshape(-1, 3)[0]

    def _unwrapped_env(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)


def execute_pick_sim_topdown(env: Any, target_xyz: np.ndarray) -> dict[str, Any]:
    """Convenience helper for the simulated top-down executor."""

    return SimulatedTopDownPickExecutor(env=env).execute(target_xyz)


def execute_pick_placeholder(env: Any, target_xyz: np.ndarray) -> dict[str, Any]:
    """Convenience helper used by environment wrappers."""

    return SafePlaceholderPickExecutor(env=env).execute(target_xyz)


def validate_target_xyz(target_xyz: np.ndarray) -> PickResult | None:
    """Return a validation-error ``PickResult`` or ``None`` for a valid target."""

    try:
        target = np.asarray(target_xyz, dtype=np.float32)
    except Exception as exc:
        return PickResult(
            success=False,
            stage="validation_failed",
            target_xyz=[],
            message=f"target_xyz could not be converted to a numeric array: {exc}",
            metadata={"executor": "SafePlaceholderPickExecutor"},
        )

    if target.shape != (3,):
        return PickResult(
            success=False,
            stage="validation_failed",
            target_xyz=target.reshape(-1).astype(float).tolist() if target.size else [],
            message=f"target_xyz must have shape (3,), got {target.shape}.",
            metadata={"executor": "SafePlaceholderPickExecutor"},
        )

    if not np.all(np.isfinite(target)):
        return PickResult(
            success=False,
            stage="validation_failed",
            target_xyz=target.astype(float).tolist(),
            message="target_xyz must contain only finite values.",
            metadata={"executor": "SafePlaceholderPickExecutor"},
        )

    return None


def _json_safe_info(info: Any) -> dict[str, Any]:
    if not isinstance(info, dict):
        return {"value": _json_safe_value(info)}
    return {str(key): _json_safe_value(value) for key, value in info.items()}


def _json_safe_value(value: Any) -> Any:
    array = _to_numpy(value)
    if array is not None:
        if array.shape == ():
            return array.item()
        return array.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _to_numpy(value: Any) -> np.ndarray | None:
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)
    except Exception:
        return None


def _info_bool(info: dict[str, Any], key: str) -> bool:
    value = info.get(key)
    if isinstance(value, list):
        return any(_info_bool({"value": item}, "value") for item in value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _info_any_grasped(info: dict[str, Any]) -> bool:
    """Return true for standard or task-specific ManiSkill grasp flags."""

    return any(_info_bool(info, key) for key in _grasp_info_keys([info]))


def _grasp_info_keys(infos: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for info in infos:
        if "is_grasped" in info:
            keys.add("is_grasped")
        keys.update(
            key
            for key in info
            if key.startswith("is_") and key.endswith("_grasped") and key != "is_grasped"
        )
    return sorted(keys)
