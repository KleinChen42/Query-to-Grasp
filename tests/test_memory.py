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


def test_memory_keeps_distant_observations_separate() -> None:
    memory = ObjectMemory3D(ObjectMemoryConfig(merge_distance=0.05))

    first = memory.add_observation(_obs([0.0, 0.0, 0.0], label="cube", det_score=0.8))
    second = memory.add_observation(_obs([0.20, 0.0, 0.0], label="cube", det_score=0.8))

    assert first.object_id != second.object_id
    assert [obj.object_id for obj in memory.objects] == ["obj_0000", "obj_0001"]


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
    )
    ranked = SimpleNamespace(phrase="red cube", det_score=0.7, clip_score=0.9, fused_2d_score=0.8)

    observation = observation_from_candidate(candidate_3d, ranked, view_id="front")

    assert observation.label == "red cube"
    assert observation.view_id == "front"
    assert observation.num_points == 42
    assert observation.depth_valid_ratio == 0.5
    np.testing.assert_allclose(observation.world_xyz, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    with pytest.raises(ValueError, match="world_xyz"):
        observation_from_candidate(SimpleNamespace(world_xyz=None), ranked)


def _obs(
    xyz: list[float],
    label: str,
    det_score: float,
    view_id: str = "front",
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
    )
