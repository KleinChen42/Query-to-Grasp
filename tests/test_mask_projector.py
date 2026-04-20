from __future__ import annotations

import numpy as np

from src.perception.clip_rerank import rerank_candidates_with_clip
from src.perception.grounding_dino import DetectionCandidate
from src.perception.mask_projector import lift_box_to_3d


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


def test_lift_box_to_3d_handles_invalid_depth() -> None:
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    depth = np.zeros((3, 3), dtype=np.float32)

    candidate = lift_box_to_3d(rgb=rgb, depth=depth, box_xyxy=np.array([0, 0, 2, 2], dtype=np.float32))

    assert candidate.num_points == 0
    assert candidate.camera_xyz is None
    assert candidate.metadata["reason"] == "no_valid_depth"


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
