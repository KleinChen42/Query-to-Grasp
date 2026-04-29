from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.memory.fusion import FusionWeights
from src.memory.object_memory_3d import (
    ObjectMemory3D,
    ObjectMemoryConfig,
    ObjectObservation3D,
    observation_from_candidate,
)


def test_memory_merges_nearby_observations_across_views() -> None:
    memory = ObjectMemory3D(ObjectMemoryConfig(merge_distance=0.10))

    first = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            label="red cube",
            det_score=0.8,
            clip_score=0.7,
            fused_2d_score=0.75,
            view_id="front",
            num_points=500,
            depth_valid_ratio=0.8,
        )
    )
    second = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.04, 0.0, 0.0], dtype=np.float32),
            label="cube",
            det_score=0.7,
            clip_score=0.6,
            fused_2d_score=0.65,
            view_id="left",
            num_points=700,
            depth_valid_ratio=0.6,
        )
    )

    assert first.object_id == second.object_id
    assert len(memory.objects) == 1
    obj = memory.objects[0]
    np.testing.assert_allclose(obj.world_xyz, np.array([0.02, 0.0, 0.0], dtype=np.float32))
    assert obj.num_observations == 2
    assert obj.view_ids == ["front", "left"]
    assert set(obj.label_votes) == {"red cube", "cube"}
    assert obj.geometry_confidence > 0.0
    assert obj.overall_confidence > 0.0


def test_observation_json_includes_optional_grasp_fields() -> None:
    observation = ObjectObservation3D(
        world_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        label="red cube",
        det_score=0.8,
        fused_2d_score=0.8,
        view_id="front",
        num_points=500,
        depth_valid_ratio=0.9,
        grasp_world_xyz=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        grasp_camera_xyz=np.array([0.4, 0.5, 0.6], dtype=np.float32),
        grasp_num_points=42,
        grasp_metadata={"strategy": "shifted_crop"},
    )

    payload = observation.to_json_dict()

    assert payload["grasp_world_xyz"] == pytest.approx([0.1, 0.2, 0.3])
    assert payload["grasp_camera_xyz"] == pytest.approx([0.4, 0.5, 0.6])
    assert payload["grasp_num_points"] == 42
    assert payload["grasp_metadata"] == {"strategy": "shifted_crop"}


def test_memory_records_per_observation_grasp_history() -> None:
    memory = ObjectMemory3D(ObjectMemoryConfig(merge_distance=0.10))

    memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            label="red cube",
            det_score=0.8,
            fused_2d_score=0.8,
            view_id="front",
            num_points=500,
            depth_valid_ratio=0.9,
            grasp_world_xyz=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            grasp_num_points=42,
            grasp_metadata={"strategy": "component"},
            metadata={"rank": 0, "source": "detector_only"},
        )
    )

    obj = memory.objects[0]

    assert obj.metadata["observation_history"][0]["view_id"] == "front"
    assert obj.metadata["observation_history"][0]["label"] == "red cube"
    assert obj.metadata["observation_history"][0]["metadata"] == {"rank": 0, "source": "detector_only"}
    assert obj.metadata["grasp_observation_history"][0]["grasp_world_xyz"] == pytest.approx([0.1, 0.2, 0.3])
    assert obj.metadata["grasp_observation_history"][0]["grasp_num_points"] == 42


def test_memory_median_fuses_grasp_points_separately_from_semantic_center() -> None:
    memory = ObjectMemory3D(ObjectMemoryConfig(merge_distance=0.10))

    memory.add_observation(
        _obs([0.0, 0.0, 0.0], label="cube", det_score=0.8, grasp_world_xyz=[0.1, 0.0, 0.0])
    )
    memory.add_observation(
        _obs([0.02, 0.0, 0.0], label="cube", det_score=0.8, grasp_world_xyz=[0.3, 0.0, 0.0])
    )
    memory.add_observation(
        _obs([0.04, 0.0, 0.0], label="cube", det_score=0.8, grasp_world_xyz=[0.2, 0.0, 0.0])
    )

    obj = memory.objects[0]

    np.testing.assert_allclose(obj.world_xyz, np.array([0.02, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(obj.grasp_world_xyz, np.array([0.2, 0.0, 0.0], dtype=np.float32))
    assert len(obj.grasp_observation_xyzs) == 3
    payload = obj.to_json_dict()
    assert payload["grasp_world_xyz"] == pytest.approx([0.2, 0.0, 0.0])
    assert len(payload["grasp_observation_xyzs"]) == 3


def test_memory_keeps_distant_observations_separate() -> None:
    memory = ObjectMemory3D(ObjectMemoryConfig(merge_distance=0.05))

    first = memory.add_observation(_obs([0.0, 0.0, 0.0], label="cube", det_score=0.8))
    second = memory.add_observation(_obs([0.20, 0.0, 0.0], label="cube", det_score=0.8))

    assert first.object_id != second.object_id
    assert [obj.object_id for obj in memory.objects] == ["obj_0000", "obj_0001"]


def test_memory_can_prefer_selected_object_when_geometry_is_compatible() -> None:
    memory = ObjectMemory3D(ObjectMemoryConfig(merge_distance=0.08))

    selected = memory.add_observation(_obs([0.0, 0.0, 0.0], label="cube", det_score=0.8, view_id="front"))
    other = memory.add_observation(_obs([0.09, 0.0, 0.0], label="cube", det_score=0.8, view_id="left"))

    matched, assignment = memory.add_observation_with_preferred_object(
        _obs([0.07, 0.0, 0.0], label="cube", det_score=0.8, view_id="closer_left"),
        preferred_object_id=selected.object_id,
        preferred_merge_distance=0.08,
    )

    assert other.object_id != selected.object_id
    assert matched.object_id == selected.object_id
    assert assignment["used_preferred_object"] is True
    assert assignment["preferred_object_compatible"] is True
    assert assignment["created_new_object"] is False
    assert len(memory.objects) == 2


def test_preferred_new_view_merge_applies_support_floor_when_raw_confidence_drops() -> None:
    memory = ObjectMemory3D(ObjectMemoryConfig(merge_distance=0.08))

    selected = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            label="cube",
            det_score=0.9,
            clip_score=0.0,
            fused_2d_score=0.9,
            view_id="front",
            num_points=1000,
            depth_valid_ratio=1.0,
        )
    )
    confidence_before = float(selected.overall_confidence)

    matched, assignment = memory.add_observation_with_preferred_object(
        ObjectObservation3D(
            world_xyz=np.array([0.04, 0.0, 0.0], dtype=np.float32),
            label="cube",
            det_score=0.1,
            clip_score=0.0,
            fused_2d_score=0.1,
            view_id="closer_left",
            num_points=1,
            depth_valid_ratio=0.1,
        ),
        preferred_object_id=selected.object_id,
        preferred_merge_distance=0.08,
    )

    support_floor = matched.fusion_trace["new_view_support_floor"]
    assert matched.object_id == selected.object_id
    assert assignment["used_preferred_object"] is True
    assert support_floor["applied"] is True
    assert support_floor["raw_overall_confidence"] < confidence_before
    assert matched.overall_confidence > confidence_before
    assert support_floor["confidence_floor"] == pytest.approx(matched.overall_confidence)

    floor_after_new_view = float(matched.overall_confidence)
    matched_again, _ = memory.add_observation_with_preferred_object(
        ObjectObservation3D(
            world_xyz=np.array([0.03, 0.0, 0.0], dtype=np.float32),
            label="cube",
            det_score=0.1,
            clip_score=0.0,
            fused_2d_score=0.1,
            view_id="closer_left",
            num_points=1,
            depth_valid_ratio=0.1,
        ),
        preferred_object_id=selected.object_id,
        preferred_merge_distance=0.08,
    )

    same_view_support_floor = matched_again.fusion_trace["new_view_support_floor"]
    assert matched_again.object_id == selected.object_id
    assert same_view_support_floor["applied"] is True
    assert matched_again.overall_confidence == pytest.approx(floor_after_new_view)


def test_select_best_is_deterministic_with_view_tie_break() -> None:
    memory = ObjectMemory3D(
        ObjectMemoryConfig(
            merge_distance=0.05,
            fusion_weights=FusionWeights(
                det_score=1.0,
                clip_score=0.0,
                view_score=0.0,
                consistency_score=0.0,
                geometry_score=0.0,
            ),
        )
    )

    one_view = memory.add_observation(_obs([0.0, 0.0, 0.0], label="cube", det_score=0.8, view_id="front"))
    two_views = memory.add_observation(_obs([0.2, 0.0, 0.0], label="cube", det_score=0.8, view_id="front"))
    memory.add_observation(_obs([0.21, 0.0, 0.0], label="cube", det_score=0.8, view_id="left"))

    selected = memory.select_best(label="cube")

    assert selected is not None
    assert selected.object_id == two_views.object_id
    assert selected.object_id != one_view.object_id


def test_observation_from_candidate_requires_world_coordinates() -> None:
    candidate_3d = SimpleNamespace(
        world_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        num_points=42,
        depth_valid_ratio=0.5,
        point_cloud_path="candidate.ply",
        grasp_world_xyz=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        grasp_camera_xyz=np.array([0.4, 0.5, 0.6], dtype=np.float32),
        grasp_num_points=24,
        grasp_metadata={"strategy": "workspace_object_component"},
    )
    ranked = SimpleNamespace(phrase="red cube", det_score=0.7, clip_score=0.9, fused_2d_score=0.8)

    observation = observation_from_candidate(candidate_3d, ranked, view_id="front")

    assert observation.label == "red cube"
    assert observation.view_id == "front"
    assert observation.num_points == 42
    assert observation.depth_valid_ratio == 0.5
    np.testing.assert_allclose(observation.world_xyz, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(observation.grasp_world_xyz, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_allclose(observation.grasp_camera_xyz, np.array([0.4, 0.5, 0.6], dtype=np.float32))
    assert observation.grasp_num_points == 24
    assert observation.grasp_metadata == {"strategy": "workspace_object_component"}

    with pytest.raises(ValueError, match="world_xyz"):
        observation_from_candidate(SimpleNamespace(world_xyz=None), ranked)


def _obs(
    xyz: list[float],
    label: str,
    det_score: float,
    view_id: str = "front",
    grasp_world_xyz: list[float] | None = None,
) -> ObjectObservation3D:
    return ObjectObservation3D(
        world_xyz=np.array(xyz, dtype=np.float32),
        label=label,
        det_score=det_score,
        clip_score=0.0,
        fused_2d_score=det_score,
        view_id=view_id,
        num_points=1000,
        depth_valid_ratio=1.0,
        grasp_world_xyz=None if grasp_world_xyz is None else np.array(grasp_world_xyz, dtype=np.float32),
    )
