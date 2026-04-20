"""Minimal ManiSkill environment wrapper for Phase 1 debugging.

The wrapper keeps simulator-specific assumptions isolated. It uses Gymnasium
or Gym only when an environment is actually created, so importing this module
does not require ManiSkill to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Mapping, Sequence

import numpy as np

from src.env.camera_utils import ObservationFrame, extract_observation_frame

LOGGER = logging.getLogger(__name__)


@dataclass
class EnvStep:
    """Normalized step result across Gym and Gymnasium APIs."""

    observation: Mapping[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class ManiSkillScene:
    """Thin wrapper around a ManiSkill Gym/Gymnasium environment."""

    def __init__(
        self,
        env_name: str = "PickCube-v1",
        obs_mode: str = "rgbd",
        control_mode: str | None = None,
        render_mode: str | None = None,
        camera_name: str | None = None,
        **env_kwargs: Any,
    ) -> None:
        """Create a ManiSkill environment.

        Args:
            env_name: ManiSkill environment id, for example ``PickCube-v1``.
            obs_mode: Observation mode passed to ManiSkill. ``rgbd`` is the
                expected Phase 1 default.
            control_mode: Optional control mode. Left unset by default because
                valid values differ across ManiSkill tasks.
            render_mode: Optional render mode.
            camera_name: Preferred camera key for extraction.
            **env_kwargs: Extra keyword arguments forwarded to ``gym.make``.
        """

        self.env_name = env_name
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.env_kwargs = dict(env_kwargs)
        self.env = self._make_env()
        self.last_raw_observation: Mapping[str, Any] | None = None
        self.last_info: dict[str, Any] = {}

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> Mapping[str, Any]:
        """Reset the environment and return the raw observation."""

        LOGGER.info("Resetting %s with seed=%s", self.env_name, seed)
        try:
            result = self.env.reset(seed=seed, options=options)
        except TypeError:
            try:
                result = self.env.reset(seed=seed)
            except TypeError:
                result = self.env.reset()

        observation, info = _split_reset_result(result)
        self.last_raw_observation = observation
        self.last_info = info
        return observation

    def step(self, action: Any | None = None) -> EnvStep:
        """Step the environment once.

        If ``action`` is omitted, a sample from ``env.action_space`` is used.
        This is intended for smoke debugging only, not policy evaluation.
        """

        if action is None:
            action_space = getattr(self.env, "action_space", None)
            if action_space is None or not hasattr(action_space, "sample"):
                raise ValueError("No action was provided and env.action_space.sample() is unavailable.")
            action = action_space.sample()

        result = self.env.step(action)
        step = _split_step_result(result)
        self.last_raw_observation = step.observation
        self.last_info = step.info
        return step

    def get_observation(self, camera_name: str | None = None) -> ObservationFrame:
        """Return the normalized observation frame from the latest raw observation."""

        raw_observation = self._latest_observation()
        return extract_observation_frame(raw_observation, camera_name=camera_name or self.camera_name)

    def get_multiview_observations(self, view_ids: Sequence[str]) -> list[ObservationFrame]:
        """Extract multiple camera frames from the latest observation.

        This does not move the robot or camera. It simply asks the parser to
        select each named camera from the same observation dictionary.
        """

        raw_observation = self._latest_observation()
        return [extract_observation_frame(raw_observation, camera_name=view_id) for view_id in view_ids]

    def execute_pick(self, target_xyz: np.ndarray) -> dict[str, Any]:
        """Execute a minimal pick attempt for a 3D target.

        The current default is a safe placeholder that validates the target and
        returns a structured result without sending simulator actions.
        """

        from src.manipulation.pick_executor import execute_pick_placeholder

        return execute_pick_placeholder(self.env, target_xyz)

    def close(self) -> None:
        """Close the wrapped simulator environment."""

        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            close_fn()

    def __enter__(self) -> "ManiSkillScene":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _latest_observation(self) -> Mapping[str, Any]:
        if self.last_raw_observation is not None:
            return self.last_raw_observation

        get_obs = getattr(self.env, "get_obs", None)
        if callable(get_obs):
            LOGGER.info("No cached observation found; calling env.get_obs().")
            observation = get_obs()
            self.last_raw_observation = observation
            return observation

        raise RuntimeError("No observation is available. Call reset() before get_observation().")

    def _make_env(self) -> Any:
        gym = _import_gym()
        _try_register_maniskill_envs()

        make_kwargs = {"obs_mode": self.obs_mode, **self.env_kwargs}
        if self.control_mode is not None:
            make_kwargs["control_mode"] = self.control_mode
        if self.render_mode is not None:
            make_kwargs["render_mode"] = self.render_mode

        LOGGER.info("Creating ManiSkill env %s with kwargs=%s", self.env_name, make_kwargs)
        try:
            return gym.make(self.env_name, **make_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Failed to create ManiSkill environment. Check that ManiSkill is installed, "
                f"the env id '{self.env_name}' is valid, and the obs/control modes are supported. "
                f"Original error: {exc}"
            ) from exc


def _import_gym() -> Any:
    try:
        import gymnasium as gym  # type: ignore

        return gym
    except ImportError:
        try:
            import gym  # type: ignore

            return gym
        except ImportError as exc:
            raise RuntimeError(
                "Neither gymnasium nor gym is installed. Install ManiSkill with its Gym "
                "dependency before creating an environment."
            ) from exc


def _try_register_maniskill_envs() -> None:
    try:
        import mani_skill.envs  # noqa: F401

        LOGGER.debug("Imported mani_skill.envs for environment registration.")
    except ImportError:
        try:
            import mani_skill2.envs  # type: ignore # noqa: F401

            LOGGER.debug("Imported mani_skill2.envs for environment registration.")
        except ImportError:
            LOGGER.warning(
                "Could not import mani_skill.envs or mani_skill2.envs. "
                "Environment creation may still work if registration happened elsewhere."
            )


def _split_reset_result(result: Any) -> tuple[Mapping[str, Any], dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
        return observation, dict(info or {})
    return result, {}


def _split_step_result(result: Any) -> EnvStep:
    if not isinstance(result, tuple):
        raise ValueError(f"env.step() returned an unexpected non-tuple result: {type(result).__name__}")

    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return EnvStep(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info or {}),
        )

    if len(result) == 4:
        observation, reward, done, info = result
        return EnvStep(
            observation=observation,
            reward=float(reward),
            terminated=bool(done),
            truncated=False,
            info=dict(info or {}),
        )

    raise ValueError(f"env.step() returned {len(result)} values; expected 4 or 5.")
