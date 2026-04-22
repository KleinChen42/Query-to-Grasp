from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np

import scripts.run_multiview_fusion_debug as multiview
from src.memory.object_memory_3d import ObjectMemory3D, ObjectObservation3D
from src.perception.clip_rerank import RankedCandidate
from src.perception.mask_projector import Candidate3D


def test_normalize_view_ids_uses_default_when_no_view_is_requested() -> None:
    assert multiview.normalize_view_ids(None, None) == [None]
    assert multiview.normalize_view_ids([], "base_camera") == ["base_camera"]
    assert multiview.normalize_view_ids(["", "front", " left "], None) == ["front", "left"]


def test_collect_frames_with_tabletop_preset_recaptures_base_camera() -> None:
    class FakeScene:
        def __init__(self) -> None:
            self.calls = []

        def capture_observation_from_camera_pose(self, camera_name, eye, target, up):
            self.calls.append((camera_name, eye, target, up))
            return SimpleNamespace(label=f"frame_{len(self.calls)}")

    scene = FakeScene()

    frames = multiview.collect_frames(
        scene,
        view_ids=[None],
        view_preset="tabletop_3",
        preset_camera_name="base_camera",
    )

    assert [view_id for view_id, _ in frames] == ["front", "left", "right"]
    assert len(scene.calls) == 3
    assert all(call[0] == "base_camera" for call in scene.calls)
    assert scene.calls[0][1] == (0.35, 0.0, 0.55)


def test_build_memory_config_uses_cli_weights() -> None:
    config = multiview.build_memory_config(
        argparse.Namespace(
            merge_distance=0.12,
            min_points_full_confidence=250,
            max_views_full_confidence=4,
            fusion_det_weight=0.1,
            fusion_clip_weight=0.2,
            fusion_view_weight=0.3,
            fusion_consistency_weight=0.4,
            fusion_geometry_weight=0.5,
        )
    )

    assert config.merge_distance == 0.12
    assert config.min_points_for_full_geometry_confidence == 250
    assert config.max_views_for_full_view_confidence == 4
    assert config.fusion_weights.det_score == 0.1
    assert config.fusion_weights.geometry_score == 0.5


def test_select_memory_target_prefers_exact_query_label() -> None:
    memory = ObjectMemory3D()
    memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            label="cube",
            det_score=0.9,
            fused_2d_score=0.9,
            view_id="front",
            num_points=1000,
            depth_valid_ratio=1.0,
        )
    )
    red_cube = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.2, 0.0, 0.0], dtype=np.float32),
            label="red cube",
            det_score=0.8,
            fused_2d_score=0.8,
            view_id="front",
            num_points=1000,
            depth_valid_ratio=1.0,
        )
    )

    selected, selection_label = multiview.select_memory_target(
        memory,
        {"normalized_prompt": "red cube", "target_name": "cube", "synonyms": ["cube", "block"]},
    )

    assert selected is not None
    assert selected.object_id == red_cube.object_id
    assert selection_label == "red cube"


def test_lift_and_add_candidates_updates_memory(monkeypatch, tmp_path) -> None:
    ranked = RankedCandidate(
        box_xyxy=np.array([1, 2, 3, 4], dtype=np.float32),
        det_score=0.7,
        clip_score=0.0,
        fused_2d_score=0.7,
        phrase="red cube",
        rank=0,
        source="detector_only",
    )
    lifted = Candidate3D(
        box_xyxy=ranked.box_xyxy,
        camera_xyz=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        world_xyz=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        num_points=25,
        depth_valid_ratio=0.5,
    )

    def fake_lift_box_to_3d(**kwargs):
        return lifted

    monkeypatch.setattr(multiview, "lift_box_to_3d", fake_lift_box_to_3d)

    memory = ObjectMemory3D()
    frame = SimpleNamespace(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth=np.ones((8, 8), dtype=np.float32),
        segmentation=None,
        camera_info=SimpleNamespace(intrinsic=None, extrinsic=None, extrinsic_key=None),
    )
    args = argparse.Namespace(
        save_candidate_pointclouds=False,
        segmentation_id=None,
        use_segmentation=False,
        depth_scale=1000.0,
        fallback_fov_degrees=60.0,
    )

    candidates_3d, observations_added = multiview.lift_and_add_candidates(
        args=args,
        frame=frame,
        view_id="front",
        reranked=[ranked],
        memory=memory,
        view_dir=tmp_path,
    )

    assert candidates_3d == [lifted]
    assert observations_added == 1
    assert len(memory.objects) == 1
    assert memory.objects[0].top_label == "red cube"
    np.testing.assert_allclose(memory.objects[0].world_xyz, np.array([1.0, 2.0, 3.0], dtype=np.float32))
