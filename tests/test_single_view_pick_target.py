from __future__ import annotations

import numpy as np

from scripts.run_single_view_pick import choose_pick_target
from src.perception.mask_projector import Candidate3D


def _candidate(
    world_xyz: list[float] | None = None,
    grasp_world_xyz: list[float] | None = None,
    camera_xyz: list[float] | None = None,
) -> Candidate3D:
    return Candidate3D(
        box_xyxy=np.array([0, 0, 4, 4], dtype=np.float32),
        camera_xyz=None if camera_xyz is None else np.asarray(camera_xyz, dtype=np.float32),
        world_xyz=None if world_xyz is None else np.asarray(world_xyz, dtype=np.float32),
        num_points=10,
        depth_valid_ratio=1.0,
        grasp_world_xyz=None if grasp_world_xyz is None else np.asarray(grasp_world_xyz, dtype=np.float32),
    )


def test_choose_pick_target_semantic_mode_uses_world_xyz() -> None:
    candidate = _candidate(world_xyz=[1.0, 2.0, 3.0], grasp_world_xyz=[0.1, 0.2, 0.03])

    target_xyz, frame, source = choose_pick_target(candidate, grasp_target_mode="semantic")

    np.testing.assert_allclose(target_xyz, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert frame == "world"
    assert source == "world_xyz"


def test_choose_pick_target_refined_mode_uses_grasp_world_xyz() -> None:
    candidate = _candidate(world_xyz=[1.0, 2.0, 3.0], grasp_world_xyz=[0.1, 0.2, 0.03])

    target_xyz, frame, source = choose_pick_target(candidate, grasp_target_mode="refined")

    np.testing.assert_allclose(target_xyz, np.array([0.1, 0.2, 0.03], dtype=np.float32))
    assert frame == "world"
    assert source == "grasp_world_xyz"


def test_choose_pick_target_refined_mode_falls_back_to_semantic_world_xyz() -> None:
    candidate = _candidate(world_xyz=[1.0, 2.0, 3.0], camera_xyz=[4.0, 5.0, 6.0])

    target_xyz, frame, source = choose_pick_target(candidate, grasp_target_mode="refined")

    np.testing.assert_allclose(target_xyz, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert frame == "world"
    assert source == "world_xyz"


def test_choose_pick_target_default_still_falls_back_to_camera_xyz() -> None:
    candidate = _candidate(camera_xyz=[4.0, 5.0, 6.0])

    target_xyz, frame, source = choose_pick_target(candidate)

    np.testing.assert_allclose(target_xyz, np.array([4.0, 5.0, 6.0], dtype=np.float32))
    assert frame == "camera"
    assert source == "camera_xyz"
