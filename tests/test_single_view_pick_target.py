from __future__ import annotations

import numpy as np

from scripts.run_single_view_pick import choose_pick_target
from src.manipulation.place_targets import select_candidate_place_target, select_memory_place_target
from src.memory.object_memory_3d import ObjectMemory3D, ObjectObservation3D
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


def test_predicted_place_candidate_skips_pick_object_by_xy_distance() -> None:
    close_candidate = _candidate(world_xyz=[0.01, 0.01, 0.03])
    far_candidate = _candidate(world_xyz=[0.12, 0.0, 0.03])

    selected = select_candidate_place_target(
        candidates=[close_candidate, far_candidate],
        pick_xyz=np.array([0.0, 0.0, 0.03], dtype=np.float32),
        min_xy_distance=0.05,
        place_query="cube",
        place_target_z=0.02,
    )

    assert selected is not None
    np.testing.assert_allclose(selected.place_xyz, np.array([0.12, 0.0, 0.02], dtype=np.float32))
    assert selected.metadata["selected_candidate_index"] == 1
    assert selected.metadata["num_close_to_pick_rejected"] == 1
    assert selected.metadata["place_query"] == "cube"


def test_predicted_place_memory_selects_high_confidence_far_object() -> None:
    memory = ObjectMemory3D()
    memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.0, 0.0, 0.03], dtype=np.float32),
            label="cube",
            det_score=0.9,
            view_id="front",
            num_points=100,
        )
    )
    far = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.10, 0.0, 0.03], dtype=np.float32),
            label="cube",
            det_score=0.8,
            view_id="left",
            num_points=100,
        )
    )

    selected = select_memory_place_target(
        objects=memory.objects,
        pick_xyz=np.array([0.0, 0.0, 0.03], dtype=np.float32),
        min_xy_distance=0.05,
        place_query="cube",
        place_target_z=0.02,
    )

    assert selected is not None
    np.testing.assert_allclose(selected.place_xyz, np.array([far.world_xyz[0], far.world_xyz[1], 0.02]))
    assert selected.metadata["selected_object_id"] == far.object_id
    assert selected.metadata["num_close_to_pick_rejected"] == 1
