from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

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


def test_collect_reobserve_frames_uses_supported_virtual_views() -> None:
    class FakeScene:
        def __init__(self) -> None:
            self.calls = []

        def capture_observation_from_camera_pose(self, camera_name, eye, target, up):
            self.calls.append((camera_name, eye, target, up))
            return SimpleNamespace(label=f"frame_{len(self.calls)}")

    scene = FakeScene()

    frames = multiview.collect_reobserve_frames(
        scene=scene,
        suggested_view_ids=["unsupported", "top_down", "closer_left"],
        camera_name="base_camera",
        max_views=2,
    )

    assert [view_id for view_id, _ in frames] == ["top_down", "closer_left"]
    assert scene.calls[0][0] == "base_camera"
    assert scene.calls[0][1] == (0.0, 0.0, 0.75)
    assert scene.calls[1][1] == (0.0, 0.24, 0.42)


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


def test_build_reobserve_config_uses_cli_thresholds() -> None:
    config = multiview.build_reobserve_config(
        argparse.Namespace(
            reobserve_min_confidence=0.6,
            reobserve_min_confidence_gap=0.2,
            reobserve_min_views=3,
            reobserve_min_geometry_confidence=0.4,
            reobserve_min_mean_points=250.0,
            reobserve_suggested_view_ids=["top", "right"],
        )
    )

    assert config.min_overall_confidence == 0.6
    assert config.min_confidence_gap == 0.2
    assert config.min_views == 3
    assert config.min_geometry_confidence == 0.4
    assert config.min_mean_num_points == 250.0
    assert config.default_suggested_view_ids == ("top", "right")


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


def test_build_selection_trace_explains_exact_label_selection() -> None:
    memory = ObjectMemory3D()
    memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            label="cube",
            det_score=0.95,
            fused_2d_score=0.95,
            view_id="front",
            num_points=1000,
            depth_valid_ratio=1.0,
        )
    )
    red_cube = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.2, 0.0, 0.0], dtype=np.float32),
            label="red cube",
            det_score=0.7,
            fused_2d_score=0.7,
            view_id="left",
            num_points=500,
            depth_valid_ratio=0.8,
        )
    )
    parsed_query = {"raw_query": "red cube", "normalized_prompt": "red cube", "target_name": "cube", "synonyms": ["cube"]}
    selected, selection_label = multiview.select_memory_target(memory, parsed_query)

    trace = multiview.build_selection_trace(
        memory=memory,
        selected=selected,
        selection_label=selection_label,
        parsed_query=parsed_query,
    )
    markdown = multiview.render_selection_trace_markdown(trace)

    assert selected is not None
    assert selected.object_id == red_cube.object_id
    assert trace["selection"]["selected_object_id"] == red_cube.object_id
    assert trace["selection"]["selection_pool_label"] == "red cube"
    assert trace["selection"]["selection_pool_size"] == 1
    assert trace["selection"]["fallback_to_all_objects"] is False
    assert trace["ranked_selection_pool"][0]["is_selected"] is True
    assert trace["ranked_selection_pool"][0]["selection_label_vote"] > 0.0
    assert "# Selection Trace" in markdown
    assert "red cube" in markdown
    assert red_cube.object_id in markdown


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
        grasp_world_xyz=np.array([0.11, 0.22, 0.33], dtype=np.float32),
        grasp_camera_xyz=np.array([0.01, 0.02, 0.03], dtype=np.float32),
        grasp_num_points=21,
        grasp_metadata={"strategy": "unit_test"},
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

    candidates_3d, observations_added, observation_assignments = multiview.lift_and_add_candidates(
        args=args,
        frame=frame,
        view_id="front",
        reranked=[ranked],
        memory=memory,
        view_dir=tmp_path,
    )

    assert candidates_3d == [lifted]
    assert observations_added == 1
    assert observation_assignments[0]["object_id"] == memory.objects[0].object_id
    assert observation_assignments[0]["created_new_object"] is True
    assert observation_assignments[0]["has_grasp_world_xyz"] is True
    assert observation_assignments[0]["grasp_num_points"] == 21
    assert len(memory.objects) == 1
    assert memory.objects[0].top_label == "red cube"
    np.testing.assert_allclose(memory.objects[0].world_xyz, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(memory.objects[0].grasp_world_xyz, np.array([0.11, 0.22, 0.33], dtype=np.float32))


def test_build_selected_grasp_diagnostics_reports_offsets_and_spread() -> None:
    memory = ObjectMemory3D()
    selected = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            label="red cube",
            det_score=0.8,
            fused_2d_score=0.8,
            view_id="front",
            num_points=100,
            depth_valid_ratio=0.8,
            grasp_world_xyz=np.array([0.10, 0.0, 0.01], dtype=np.float32),
            grasp_num_points=30,
            grasp_metadata={"strategy": "component_front"},
        )
    )
    memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.01, 0.0, 0.0], dtype=np.float32),
            label="red cube",
            det_score=0.8,
            fused_2d_score=0.8,
            view_id="left",
            num_points=100,
            depth_valid_ratio=0.8,
            grasp_world_xyz=np.array([0.30, 0.0, 0.03], dtype=np.float32),
            grasp_num_points=28,
            grasp_metadata={"strategy": "component_left"},
        )
    )

    diagnostics = multiview.build_selected_grasp_diagnostics(selected)

    assert diagnostics["selected_grasp_observation_count"] == 2
    assert diagnostics["selected_semantic_to_grasp_xy_distance"] == pytest.approx(0.195)
    assert diagnostics["selected_semantic_to_grasp_z_delta"] == pytest.approx(0.02)
    assert diagnostics["selected_grasp_observation_xy_spread"] == pytest.approx(0.1)
    assert diagnostics["selected_grasp_observation_z_spread"] == pytest.approx(0.02)
    assert diagnostics["selected_grasp_observation_max_distance_to_fused"] == pytest.approx(np.sqrt(0.1**2 + 0.01**2))
    assert diagnostics["selected_grasp_observation_history"][0]["view_id"] == "front"
    assert diagnostics["selected_grasp_observation_history"][1]["grasp_metadata"] == {"strategy": "component_left"}


def test_build_closed_loop_reobserve_report_computes_deltas() -> None:
    before = {
        "num_views": 3,
        "num_memory_objects": 2,
        "num_observations_added": 5,
        "selected_object_id": "obj_0000",
        "selected_overall_confidence": 0.4,
        "selected_geometry_confidence": 0.3,
        "selected_semantic_confidence": 0.8,
        "selected_num_views": 1,
        "selected_num_observations": 2,
        "should_reobserve": True,
        "reobserve_reason": "low_overall_confidence",
    }
    after = {
        "num_views": 4,
        "num_memory_objects": 2,
        "num_observations_added": 7,
        "selected_object_id": "obj_0000",
        "selected_overall_confidence": 0.6,
        "selected_geometry_confidence": 0.5,
        "selected_semantic_confidence": 0.8,
        "selected_num_views": 2,
        "selected_num_observations": 3,
        "should_reobserve": False,
        "reobserve_reason": "confident_enough",
    }
    report = multiview.build_closed_loop_reobserve_report(
        before=before,
        after=after,
        extra_view_results=[
            {
                "view_id": "top_down",
                "num_detections": 1,
                "num_ranked_candidates": 1,
                "num_3d_candidates": 1,
                "num_observations_added": 2,
                "artifacts": "outputs/view",
            }
        ],
        selected_object_followup={
            "before_selected_object_id": "obj_0000",
            "present_after": True,
            "still_selected_after": True,
            "delta_num_views": 1,
            "delta_num_observations": 1,
            "received_observation": True,
            "gained_view_support": True,
            "merged_extra_view_ids": ["top_down"],
        },
        absorber_trace={
            "initial_selected_object_id": "obj_0000",
            "final_selected_object_id": "obj_0000",
            "absorber_object_ids": ["obj_0000"],
            "absorber_count": 1,
            "initial_selected_absorbed_extra_view": True,
            "final_selected_absorbed_extra_view": True,
            "third_object_ids": [],
            "third_object_involved": False,
            "observation_assignments": [{"object_id": "obj_0000"}],
        },
        preferred_merge_trace={
            "observation_assignment_count": 1,
            "preferred_merge_count": 1,
            "preferred_merge_rate": 1.0,
        },
        post_selection_continuity={
            "enabled": True,
            "eligible": True,
            "applied": True,
            "reason": "kept_preferred_object_within_margin",
        },
    )

    assert report["executed"] is True
    assert report["extra_views"][0]["view_id"] == "top_down"
    assert report["delta"]["num_views"] == 1
    assert report["delta"]["num_observations_added"] == 2
    assert report["delta"]["selected_overall_confidence"] == 0.19999999999999996
    assert report["delta"]["selected_geometry_confidence"] == 0.2
    assert report["delta"]["selected_num_views"] == 1
    assert report["delta"]["selected_num_observations"] == 1
    assert report["delta"]["should_reobserve_changed"] is True
    assert report["delta"]["reobserve_reason_changed"] is True
    assert report["delta"]["reobserve_resolved"] is True
    assert report["delta"]["reobserve_still_needed"] is False
    assert report["initial_selected_object_followup"]["received_observation"] is True
    assert report["initial_selected_object_followup"]["merged_extra_view_ids"] == ["top_down"]
    assert report["extra_view_absorber_trace"]["final_selected_absorbed_extra_view"] is True
    assert report["preferred_merge_trace"]["preferred_merge_rate"] == 1.0
    assert report["post_selection_continuity"]["applied"] is True


def test_build_initial_selected_object_followup_tracks_merge_into_before_selected() -> None:
    memory = ObjectMemory3D()
    selected = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            label="object",
            det_score=0.8,
            fused_2d_score=0.8,
            view_id="front",
            num_points=100,
            depth_valid_ratio=1.0,
        )
    )
    before = {
        "selected_object_id": selected.object_id,
        "selected_num_views": 1,
        "selected_view_ids": ["front"],
        "selected_num_observations": 1,
    }
    memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.01, 0.0, 0.0], dtype=np.float32),
            label="object",
            det_score=0.85,
            fused_2d_score=0.85,
            view_id="closer_left",
            num_points=120,
            depth_valid_ratio=1.0,
        )
    )
    after = {"selected_object_id": selected.object_id}

    followup = multiview.build_initial_selected_object_followup(
        before=before,
        after=after,
        memory=memory,
        extra_view_ids=["closer_left"],
    )

    assert followup["present_after"] is True
    assert followup["still_selected_after"] is True
    assert followup["delta_num_views"] == 1
    assert followup["delta_num_observations"] == 1
    assert followup["received_observation"] is True
    assert followup["gained_view_support"] is True
    assert followup["merged_extra_view_ids"] == ["closer_left"]


def test_build_closed_loop_absorber_trace_distinguishes_final_and_third_objects() -> None:
    trace = multiview.build_closed_loop_absorber_trace(
        before={"selected_object_id": "obj_0000"},
        after={"selected_object_id": "obj_0001"},
        extra_view_results=[
            {
                "observation_assignments": [
                    {"object_id": "obj_0001"},
                    {"object_id": "obj_0002"},
                    {"object_id": "obj_0001"},
                ]
            }
        ],
    )

    assert trace["absorber_object_ids"] == ["obj_0001", "obj_0002"]
    assert trace["initial_selected_absorbed_extra_view"] is False
    assert trace["final_selected_absorbed_extra_view"] is True
    assert trace["third_object_ids"] == ["obj_0002"]
    assert trace["third_object_involved"] is True


def test_collect_extra_view_absorber_object_ids_dedupes_stably() -> None:
    object_ids = multiview.collect_extra_view_absorber_object_ids(
        [
            {
                "observation_assignments": [
                    {"object_id": "obj_0002"},
                    {"object_id": "obj_0000"},
                    {"object_id": "obj_0002"},
                    {"object_id": ""},
                ]
            },
            {"observation_assignments": [{"object_id": "obj_0001"}]},
        ]
    )

    assert object_ids == ["obj_0002", "obj_0000", "obj_0001"]


def test_build_closed_loop_preferred_merge_trace_counts_used_merges() -> None:
    trace = multiview.build_closed_loop_preferred_merge_trace(
        [
            {
                "observation_assignments": [
                    {"used_preferred_object": True},
                    {"used_preferred_object": False},
                    {"used_preferred_object": True},
                ]
            }
        ]
    )

    assert trace["observation_assignment_count"] == 3
    assert trace["preferred_merge_count"] == 2
    assert trace["preferred_merge_rate"] == 2 / 3


def test_build_post_selection_continuity_trace_requires_received_observation() -> None:
    trace = multiview.build_post_selection_continuity_trace(
        args=argparse.Namespace(
            enable_post_reobserve_selection_continuity=True,
            post_reobserve_selection_margin=0.03,
        ),
        initial_snapshot={"selected_object_id": "obj_0000"},
        selected_object_followup={
            "received_observation": False,
            "gained_view_support": False,
        },
        base_selected=SimpleNamespace(object_id="obj_0001"),
        base_selection_label="cube",
    )

    assert trace["enabled"] is True
    assert trace["eligible"] is False
    assert trace["reason"] == "preferred_object_did_not_receive_extra_view"
    assert trace["selected_object_id_before"] == "obj_0001"


def test_build_post_selection_continuity_trace_allows_when_base_selected_did_not_absorb_extra_view() -> None:
    trace = multiview.build_post_selection_continuity_trace(
        args=argparse.Namespace(
            enable_post_reobserve_selection_continuity=True,
            post_reobserve_selection_margin=0.05,
        ),
        initial_snapshot={"selected_object_id": "obj_0000"},
        selected_object_followup={
            "received_observation": True,
            "gained_view_support": True,
        },
        base_selected=SimpleNamespace(object_id="obj_0001"),
        base_selection_label="cube",
        extra_view_absorber_object_ids=["obj_0000", "obj_0002"],
    )

    assert trace["eligible"] is True
    assert trace["reason"] == "eligible"
    assert trace["base_selected_absorbed_extra_view"] is False
    assert trace["extra_view_absorber_object_ids"] == ["obj_0000", "obj_0002"]


def test_build_post_selection_continuity_trace_blocks_when_base_selected_absorbed_extra_view() -> None:
    trace = multiview.build_post_selection_continuity_trace(
        args=argparse.Namespace(
            enable_post_reobserve_selection_continuity=True,
            post_reobserve_selection_margin=0.05,
        ),
        initial_snapshot={"selected_object_id": "obj_0002"},
        selected_object_followup={
            "received_observation": True,
            "gained_view_support": True,
        },
        base_selected=SimpleNamespace(object_id="obj_0000"),
        base_selection_label="cube",
        extra_view_absorber_object_ids=["obj_0002", "obj_0000"],
    )

    assert trace["eligible"] is False
    assert trace["reason"] == "base_selected_absorbed_extra_view"
    assert trace["base_selected_absorbed_extra_view"] is True
    assert trace["selected_object_id_before"] == "obj_0000"


def test_execute_selected_memory_pick_returns_not_attempted_without_selection() -> None:
    class FakeScene:
        def execute_pick(self, target_xyz, executor="placeholder"):
            raise AssertionError("execute_pick should not be called without a selected object")

    result = multiview.execute_selected_memory_pick(
        scene=FakeScene(),
        selected=None,
        args=argparse.Namespace(pick_executor="placeholder", grasp_target_mode="semantic"),
    )

    assert result["grasp_attempted"] is False
    assert result["stage"] == "not_attempted"
    assert result["target_xyz"] == []


def test_execute_selected_memory_pick_records_selected_world_target_source() -> None:
    selected = ObjectMemory3D().add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.01, -0.02, 0.03], dtype=np.float32),
            label="red cube",
            det_score=0.9,
            fused_2d_score=0.9,
            view_id="front",
            num_points=100,
            depth_valid_ratio=1.0,
        )
    )

    class FakeScene:
        def __init__(self) -> None:
            self.calls = []

        def execute_pick(self, target_xyz, executor="placeholder"):
            self.calls.append((np.asarray(target_xyz, dtype=float).tolist(), executor))
            return {
                "success": False,
                "pick_success": False,
                "grasp_attempted": False,
                "task_success": None,
                "is_grasped": None,
                "stage": "placeholder_not_executed",
                "target_xyz": np.asarray(target_xyz, dtype=float).tolist(),
                "message": "placeholder",
                "trajectory_summary": {"planned_stages": [], "executed_stages": [], "num_env_steps": 0},
                "metadata": {"executor": "SafePlaceholderPickExecutor"},
            }

    scene = FakeScene()
    result = multiview.execute_selected_memory_pick(
        scene=scene,
        selected=selected,
        args=argparse.Namespace(pick_executor="placeholder", grasp_target_mode="refined"),
    )

    assert scene.calls == [([0.009999999776482582, -0.019999999552965164, 0.029999999329447746], "placeholder")]
    assert result["metadata"]["target_used_for_pick"] == "selected_object_world_xyz"
    assert result["metadata"]["grasp_target_mode"] == "refined"
    assert result["metadata"]["grasp_target_mode_effective"] == "selected_object_world_xyz"
    assert result["metadata"]["refined_grasp_point_available"] is False
    assert result["metadata"]["semantic_world_xyz"] == [
        0.009999999776482582,
        -0.019999999552965164,
        0.029999999329447746,
    ]
    assert result["metadata"]["memory_grasp_world_xyz"] is None
    assert result["metadata"]["selected_object_id"] == selected.object_id


def test_execute_selected_memory_pick_uses_memory_grasp_target_in_refined_mode() -> None:
    selected = ObjectMemory3D().add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.01, -0.02, 0.03], dtype=np.float32),
            label="red cube",
            det_score=0.9,
            fused_2d_score=0.9,
            view_id="front",
            num_points=100,
            depth_valid_ratio=1.0,
            grasp_world_xyz=np.array([0.04, 0.05, 0.06], dtype=np.float32),
        )
    )

    class FakeScene:
        def __init__(self) -> None:
            self.calls = []

        def execute_pick(self, target_xyz, executor="placeholder"):
            self.calls.append((np.asarray(target_xyz, dtype=float).tolist(), executor))
            return {
                "success": False,
                "pick_success": False,
                "grasp_attempted": True,
                "task_success": False,
                "is_grasped": False,
                "stage": "grasp_not_confirmed",
                "target_xyz": np.asarray(target_xyz, dtype=float).tolist(),
                "message": "attempted",
                "trajectory_summary": {"planned_stages": [], "executed_stages": [], "num_env_steps": 0},
                "metadata": {"executor": "SimulatedTopDownPickExecutor"},
            }

    scene = FakeScene()
    result = multiview.execute_selected_memory_pick(
        scene=scene,
        selected=selected,
        args=argparse.Namespace(pick_executor="sim_topdown", grasp_target_mode="refined"),
    )

    assert scene.calls == [([0.03999999910593033, 0.05000000074505806, 0.05999999865889549], "sim_topdown")]
    assert result["metadata"]["target_used_for_pick"] == "memory_grasp_world_xyz"
    assert result["metadata"]["grasp_target_mode_effective"] == "memory_grasp_world_xyz"
    assert result["metadata"]["refined_grasp_point_available"] is True
    assert result["metadata"]["memory_grasp_world_xyz"] == [
        0.03999999910593033,
        0.05000000074505806,
        0.05999999865889549,
    ]


def test_choose_memory_pick_target_semantic_ignores_memory_grasp_point() -> None:
    selected = ObjectMemory3D().add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.01, -0.02, 0.03], dtype=np.float32),
            label="red cube",
            det_score=0.9,
            fused_2d_score=0.9,
            view_id="front",
            num_points=100,
            depth_valid_ratio=1.0,
            grasp_world_xyz=np.array([0.04, 0.05, 0.06], dtype=np.float32),
        )
    )

    target, source = multiview.choose_memory_pick_target(selected, grasp_target_mode="semantic")

    np.testing.assert_allclose(target, np.array([0.01, -0.02, 0.03], dtype=np.float32))
    assert source == "selected_object_world_xyz"


def test_build_summary_includes_pick_metrics() -> None:
    memory = ObjectMemory3D()
    selected = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.01, 0.02, 0.03], dtype=np.float32),
            label="cube",
            det_score=0.9,
            fused_2d_score=0.9,
            view_id="front",
            num_points=100,
            depth_valid_ratio=1.0,
            grasp_world_xyz=np.array([0.04, 0.05, 0.06], dtype=np.float32),
        )
    )
    pick_result = {
        "success": True,
        "pick_success": True,
        "grasp_attempted": True,
        "task_success": False,
        "is_grasped": True,
        "stage": "success",
        "target_xyz": [0.01, 0.02, 0.03],
        "message": "ok",
        "metadata": {"target_used_for_pick": "selected_object_world_xyz"},
    }

    summary = multiview.build_summary(
        args=argparse.Namespace(
            query="cube",
            skip_clip=True,
            detector_backend="mock",
            view_preset="tabletop_3",
            camera_name="base_camera",
            pick_executor="sim_topdown",
            grasp_target_mode="semantic",
        ),
        parsed_query={"normalized_prompt": "cube"},
        view_ids=["front"],
        memory=memory,
        selected=selected,
        selection_label="cube",
        total_observations_added=1,
        runtime_seconds=1.0,
        run_dir=Path("outputs/test"),
        pick_result=pick_result,
    )

    assert summary["pick_executor"] == "sim_topdown"
    assert summary["grasp_attempted"] is True
    assert summary["pick_success"] is True
    assert summary["task_success"] is False
    assert summary["is_grasped"] is True
    assert summary["pick_stage"] == "success"
    assert summary["pick_target_source"] == "selected_object_world_xyz"
    assert summary["selected_grasp_world_xyz"] == pytest.approx([0.04, 0.05, 0.06])
