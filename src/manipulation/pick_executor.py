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

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result dictionary."""

        return {
            "success": bool(self.success),
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
        ).to_json_dict()


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

