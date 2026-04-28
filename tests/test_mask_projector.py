from __future__ import annotations

import numpy as np

from src.perception.clip_rerank import rerank_candidates_with_clip
from src.perception.grounding_dino import DetectionCandidate
from src.perception.mask_projector import estimate_workspace_low_z_grasp_candidate, lift_box_to_3d


def test_lift_box_to_3d_uses_synthetic_depth_center() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    intrinsic = np.array(
        [
            [1.0, 0.0, 1.5],
            [0.0, 1.0, 1.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    candidate = lift_box_to_3d(
        rgb=rgb,
        depth=depth,
        box_xyxy=np.array([1.0, 1.0, 3.0, 3.0], dtype=np.float32),
        intrinsic=intrinsic,
    )

    assert candidate.num_points == 4
    assert candidate.depth_valid_ratio == 1.0
    np.testing.assert_allclose(candidate.camera_xyz, np.array([0.0, 0.0, 2.0], dtype=np.float32))
    assert candidate.world_xyz is None
    assert candidate.grasp_world_xyz is None
    assert candidate.to_json_dict()["grasp_world_xyz"] is None
    assert candidate.to_json_dict()["grasp_num_points"] == 0


def test_lift_box_to_3d_handles_invalid_depth() -> None:
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    depth = np.zeros((3, 3), dtype=np.float32)

    candidate = lift_box_to_3d(rgb=rgb, depth=depth, box_xyxy=np.array([0, 0, 2, 2], dtype=np.float32))

    assert candidate.num_points == 0
    assert candidate.camera_xyz is None
    assert candidate.metadata["reason"] == "no_valid_depth"


def test_lift_box_to_3d_converts_cam2world_gl_from_opencv_points() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.full((2, 2), 2.0, dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)

    direct = lift_box_to_3d(
        rgb=rgb,
        depth=depth,
        box_xyxy=np.array([0, 0, 1, 1], dtype=np.float32),
        intrinsic=intrinsic,
        extrinsic=extrinsic,
    )
    converted = lift_box_to_3d(
        rgb=rgb,
        depth=depth,
        box_xyxy=np.array([0, 0, 1, 1], dtype=np.float32),
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        extrinsic_source="sensor_param.base_camera.cam2world_gl",
    )

    np.testing.assert_allclose(direct.camera_xyz, np.array([0.0, 0.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(direct.world_xyz, np.array([0.0, 0.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(converted.camera_xyz, np.array([0.0, 0.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(converted.world_xyz, np.array([0.0, 0.0, -2.0], dtype=np.float32))
    assert converted.metadata["extrinsics_source"] == "sensor_param.base_camera.cam2world_gl"
    assert converted.metadata["camera_frame_conversion"] == "opencv_to_opengl"


def test_lift_box_to_3d_adds_workspace_grasp_candidate_when_supported() -> None:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.full((8, 8), 0.04, dtype=np.float32)
    intrinsic = np.array(
        [
            [100.0, 0.0, 3.5],
            [0.0, 100.0, 3.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    candidate = lift_box_to_3d(
        rgb=rgb,
        depth=depth,
        box_xyxy=np.array([0, 0, 8, 8], dtype=np.float32),
        intrinsic=intrinsic,
        extrinsic=np.eye(4, dtype=np.float32),
    )

    assert candidate.grasp_num_points == 64
    assert candidate.grasp_world_xyz is not None
    assert candidate.grasp_camera_xyz is not None
    assert candidate.grasp_metadata["applied"] is True
    assert candidate.grasp_metadata["reason"] == "workspace_filter_passed"
    assert candidate.grasp_metadata["fallback_reason"] == "too_few_elevated_points"
    np.testing.assert_allclose(candidate.grasp_world_xyz, candidate.world_xyz)
    json_dict = candidate.to_json_dict()
    assert json_dict["grasp_world_xyz"] is not None
    assert json_dict["grasp_camera_xyz"] is not None
    assert json_dict["grasp_metadata"]["strategy"] == "workspace_elevated_xy_low_z"


def test_lift_box_to_3d_uses_shifted_grasp_crop_when_original_has_no_elevated_support() -> None:
    rgb = np.zeros((12, 12, 3), dtype=np.uint8)
    depth = np.full((12, 12), 0.04, dtype=np.float32)
    depth[6:8, 3:9] = 0.07
    intrinsic = np.array(
        [
            [100.0, 0.0, 6.0],
            [0.0, 100.0, 6.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    candidate = lift_box_to_3d(
        rgb=rgb,
        depth=depth,
        box_xyxy=np.array([0, 0, 12, 6], dtype=np.float32),
        intrinsic=intrinsic,
        extrinsic=np.eye(4, dtype=np.float32),
        min_grasp_workspace_points=8,
    )

    assert candidate.grasp_world_xyz is not None
    assert candidate.grasp_metadata["source_box"] == "y_shifted_grasp_fallback"
    assert candidate.grasp_metadata["original_elevated_point_count"] == 0
    assert candidate.grasp_metadata["elevated_point_count"] >= 12
    assert candidate.grasp_metadata["reason"] == "component_elevated_xy_with_local_support_z"
    assert candidate.grasp_world_xyz[2] < 0.055
    np.testing.assert_allclose(candidate.world_xyz, np.array([-0.0002, -0.0014, 0.04]), atol=0.01)


def test_workspace_grasp_candidate_uses_elevated_object_xy_with_low_support_z() -> None:
    table_x, table_y = np.meshgrid(np.linspace(-0.15, 0.15, 12), np.linspace(-0.15, 0.15, 12))
    table = np.stack([table_x.reshape(-1), table_y.reshape(-1), np.zeros(table_x.size)], axis=1)
    local_support_x, local_support_y = np.meshgrid(np.linspace(0.04, 0.08, 4), np.linspace(-0.06, -0.02, 4))
    local_support = np.stack(
        [local_support_x.reshape(-1), local_support_y.reshape(-1), np.zeros(local_support_x.size)],
        axis=1,
    )
    elevated_x, elevated_y = np.meshgrid(np.linspace(0.055, 0.075, 4), np.linspace(-0.052, -0.032, 4))
    elevated = np.stack(
        [elevated_x.reshape(-1), elevated_y.reshape(-1), np.full(elevated_x.size, 0.035)],
        axis=1,
    )
    world_points = np.concatenate([table, local_support, elevated], axis=0).astype(np.float32)

    candidate = estimate_workspace_low_z_grasp_candidate(
        camera_points=world_points,
        world_points=world_points,
        min_points=20,
        min_elevated_points=8,
    )

    assert candidate["grasp_world_xyz"] is not None
    np.testing.assert_allclose(candidate["grasp_world_xyz"], np.array([0.065, -0.042, 0.0]), atol=0.01)
    assert candidate["grasp_num_points"] == 16
    assert candidate["grasp_metadata"]["applied"] is True
    assert candidate["grasp_metadata"]["reason"] == "component_elevated_xy_with_local_support_z"
    assert candidate["grasp_metadata"]["elevated_point_count"] == 16
    assert candidate["grasp_metadata"]["selected_component_size"] == 16
    assert candidate["grasp_metadata"]["component_selection_strategy"] == "largest_component_then_low_spread"
    assert candidate["grasp_metadata"]["local_support_point_count"] > 0


def test_workspace_grasp_candidate_selects_largest_elevated_component() -> None:
    table_x, table_y = np.meshgrid(np.linspace(-0.18, 0.18, 14), np.linspace(-0.18, 0.18, 14))
    table = np.stack([table_x.reshape(-1), table_y.reshape(-1), np.zeros(table_x.size)], axis=1)
    small_x, small_y = np.meshgrid(np.linspace(-0.12, -0.10, 3), np.linspace(0.10, 0.12, 3))
    small_component = np.stack(
        [small_x.reshape(-1), small_y.reshape(-1), np.full(small_x.size, 0.035)],
        axis=1,
    )
    large_x, large_y = np.meshgrid(np.linspace(0.055, 0.075, 4), np.linspace(-0.052, -0.032, 4))
    large_component = np.stack(
        [large_x.reshape(-1), large_y.reshape(-1), np.full(large_x.size, 0.035)],
        axis=1,
    )
    world_points = np.concatenate([table, small_component, large_component], axis=0).astype(np.float32)

    candidate = estimate_workspace_low_z_grasp_candidate(
        camera_points=world_points,
        world_points=world_points,
        min_points=20,
        min_elevated_points=8,
        component_radius=0.025,
    )

    assert candidate["grasp_world_xyz"] is not None
    np.testing.assert_allclose(candidate["grasp_world_xyz"], np.array([0.065, -0.042, 0.0]), atol=0.01)
    assert candidate["grasp_num_points"] == 16
    assert candidate["grasp_metadata"]["component_count"] == 2
    assert candidate["grasp_metadata"]["selected_component_size"] == 16
    assert candidate["grasp_metadata"]["component_fallback_reason"] is None


def test_workspace_grasp_candidate_falls_back_when_components_are_too_small() -> None:
    table_x, table_y = np.meshgrid(np.linspace(-0.18, 0.18, 14), np.linspace(-0.18, 0.18, 14))
    table = np.stack([table_x.reshape(-1), table_y.reshape(-1), np.zeros(table_x.size)], axis=1)
    elevated = np.array(
        [
            [-0.16, -0.16, 0.035],
            [-0.12, 0.12, 0.035],
            [-0.08, -0.02, 0.035],
            [-0.04, 0.15, 0.035],
            [0.00, -0.12, 0.035],
            [0.04, 0.04, 0.035],
            [0.08, -0.16, 0.035],
            [0.12, 0.12, 0.035],
            [0.16, -0.04, 0.035],
        ],
        dtype=np.float32,
    )
    world_points = np.concatenate([table, elevated], axis=0).astype(np.float32)

    candidate = estimate_workspace_low_z_grasp_candidate(
        camera_points=world_points,
        world_points=world_points,
        min_points=20,
        min_elevated_points=8,
        component_radius=0.005,
    )

    assert candidate["grasp_world_xyz"] is not None
    np.testing.assert_allclose(candidate["grasp_world_xyz"][:2], np.median(elevated[:, :2], axis=0), atol=1e-6)
    assert candidate["grasp_num_points"] == 9
    assert candidate["grasp_metadata"]["component_fallback_reason"] == "no_component_met_min_size"
    assert candidate["grasp_metadata"]["component_selection_strategy"] == "global_elevated_median"


def test_workspace_grasp_candidate_can_use_low_workspace_component_when_no_elevated_points() -> None:
    small_x, small_y = np.meshgrid(np.linspace(-0.12, -0.10, 3), np.linspace(0.10, 0.12, 3))
    small_component = np.stack(
        [small_x.reshape(-1), small_y.reshape(-1), np.zeros(small_x.size)],
        axis=1,
    )
    large_x, large_y = np.meshgrid(np.linspace(0.055, 0.075, 4), np.linspace(-0.052, -0.032, 4))
    large_component = np.stack(
        [large_x.reshape(-1), large_y.reshape(-1), np.zeros(large_x.size)],
        axis=1,
    )
    world_points = np.concatenate([small_component, large_component], axis=0).astype(np.float32)

    candidate = estimate_workspace_low_z_grasp_candidate(
        camera_points=world_points,
        world_points=world_points,
        min_points=8,
        min_elevated_points=8,
        component_radius=0.025,
    )

    assert candidate["grasp_world_xyz"] is not None
    np.testing.assert_allclose(candidate["grasp_world_xyz"], np.array([0.065, -0.042, 0.0]), atol=0.01)
    assert candidate["grasp_num_points"] == 16
    assert candidate["grasp_metadata"]["fallback_reason"] == "too_few_elevated_points"
    assert candidate["grasp_metadata"]["component_selection_strategy"] == (
        "largest_workspace_component_then_low_spread"
    )
    assert candidate["grasp_metadata"]["component_fallback_reason"] == "too_few_elevated_points"
    assert candidate["grasp_metadata"]["selected_component_size"] == 16


def test_workspace_grasp_candidate_falls_back_when_elevated_support_is_sparse() -> None:
    table_x, table_y = np.meshgrid(np.linspace(-0.15, 0.15, 8), np.linspace(-0.15, 0.15, 8))
    table = np.stack([table_x.reshape(-1), table_y.reshape(-1), np.zeros(table_x.size)], axis=1)
    elevated = np.array(
        [
            [0.06, -0.04, 0.035],
            [0.07, -0.04, 0.035],
            [0.06, -0.03, 0.035],
        ],
        dtype=np.float32,
    )
    world_points = np.concatenate([table, elevated], axis=0).astype(np.float32)

    candidate = estimate_workspace_low_z_grasp_candidate(
        camera_points=world_points,
        world_points=world_points,
        min_points=20,
        min_elevated_points=8,
    )

    assert candidate["grasp_world_xyz"] is not None
    np.testing.assert_allclose(candidate["grasp_world_xyz"], np.median(world_points, axis=0), atol=1e-6)
    assert candidate["grasp_num_points"] == world_points.shape[0]
    assert candidate["grasp_metadata"]["reason"] == "workspace_filter_passed"
    assert candidate["grasp_metadata"]["fallback_reason"] == "too_few_elevated_points"
    assert candidate["grasp_metadata"]["elevated_point_count"] == 3


def test_lift_box_to_3d_leaves_grasp_candidate_empty_when_unsupported() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.full((4, 4), 0.04, dtype=np.float32)
    intrinsic = np.array(
        [
            [100.0, 0.0, 1.5],
            [0.0, 100.0, 1.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    candidate = lift_box_to_3d(
        rgb=rgb,
        depth=depth,
        box_xyxy=np.array([0, 0, 4, 4], dtype=np.float32),
        intrinsic=intrinsic,
        extrinsic=np.eye(4, dtype=np.float32),
    )

    assert candidate.grasp_world_xyz is None
    assert candidate.grasp_camera_xyz is None
    assert candidate.grasp_num_points == 16
    assert candidate.grasp_metadata["applied"] is False
    assert candidate.grasp_metadata["reason"] == "too_few_workspace_points"


def test_lift_box_to_3d_can_use_segmentation_id() -> None:
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    depth = np.ones((3, 3), dtype=np.float32)
    segmentation = np.array(
        [
            [0, 1, 1],
            [0, 1, 2],
            [0, 2, 2],
        ],
        dtype=np.int32,
    )

    candidate = lift_box_to_3d(
        rgb=rgb,
        depth=depth,
        box_xyxy=np.array([0, 0, 3, 3], dtype=np.float32),
        segmentation=segmentation,
        use_segmentation=True,
        segmentation_id=2,
    )

    assert candidate.segmentation_id == 2
    assert candidate.num_points == 3
    assert candidate.depth_valid_ratio == 3 / 9


def test_reranking_order_is_deterministic_with_mock_scores() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    candidates = [
        DetectionCandidate(box_xyxy=np.array([0, 0, 4, 4], dtype=np.float32), det_score=0.9, phrase="first"),
        DetectionCandidate(box_xyxy=np.array([1, 1, 5, 5], dtype=np.float32), det_score=0.5, phrase="second"),
        DetectionCandidate(box_xyxy=np.array([2, 2, 6, 6], dtype=np.float32), det_score=0.8, phrase="third"),
    ]

    def score_fn(crops, prompts):
        assert len(crops) == 3
        assert prompts == ["red cube"]
        return np.array([0.2, 0.95, 0.2], dtype=np.float32)

    ranked = rerank_candidates_with_clip(
        image=image,
        candidates=candidates,
        text_prompt="red cube",
        detector_weight=0.5,
        clip_weight=0.5,
        score_fn=score_fn,
    )

    assert [candidate.phrase for candidate in ranked] == ["second", "first", "third"]
    assert [candidate.rank for candidate in ranked] == [0, 1, 2]
