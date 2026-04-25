from __future__ import annotations

import numpy as np

from src.memory.object_memory_3d import ObjectMemory3D, ObjectObservation3D
from src.policy.target_selector import (
    apply_selection_continuity,
    build_selection_trace,
    candidate_selection_labels,
    render_selection_trace_markdown,
    select_memory_target,
)


def test_candidate_selection_labels_are_deduped_in_priority_order() -> None:
    labels = candidate_selection_labels(
        {
            "normalized_prompt": "red cube",
            "target_name": "cube",
            "synonyms": ["cube", "block", "red cube"],
        }
    )

    assert labels == ["red cube", "cube", "block"]


def test_select_memory_target_prefers_first_matching_query_label() -> None:
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

    selected, selection_label = select_memory_target(
        memory,
        {"normalized_prompt": "red cube", "target_name": "cube", "synonyms": ["cube", "block"]},
    )

    assert selected is not None
    assert selected.object_id == red_cube.object_id
    assert selection_label == "red cube"


def test_selection_trace_renders_ranked_pool_and_reason() -> None:
    memory = ObjectMemory3D()
    selected = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.0, 0.1, 0.0], dtype=np.float32),
            label="red cube",
            det_score=0.9,
            clip_score=0.1,
            fused_2d_score=0.9,
            view_id="front",
            num_points=1000,
            depth_valid_ratio=1.0,
        )
    )
    memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.3, 0.1, 0.0], dtype=np.float32),
            label="red cube",
            det_score=0.4,
            clip_score=0.1,
            fused_2d_score=0.4,
            view_id="left",
            num_points=500,
            depth_valid_ratio=0.5,
        )
    )
    parsed_query = {"raw_query": "red cube", "normalized_prompt": "red cube", "target_name": "cube", "synonyms": []}

    trace = build_selection_trace(
        memory=memory,
        selected=selected,
        selection_label="red cube",
        parsed_query=parsed_query,
    )
    markdown = render_selection_trace_markdown(trace)

    assert trace["selection"]["selected_object_id"] == selected.object_id
    assert trace["selection"]["selection_pool_size"] == 2
    assert trace["ranked_selection_pool"][0]["is_selected"] is True
    assert trace["ranked_selection_pool"][0]["selection_label_vote"] > trace["ranked_selection_pool"][1]["selection_label_vote"]
    assert "deterministic tie-breaks" in trace["selection"]["reason"]
    assert "# Selection Trace" in markdown
    assert selected.object_id in markdown


def test_select_memory_target_falls_back_to_best_object_without_label_match() -> None:
    memory = ObjectMemory3D()
    selected_object = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            label="mystery object",
            det_score=0.8,
            fused_2d_score=0.8,
            view_id="front",
            num_points=1000,
            depth_valid_ratio=1.0,
        )
    )

    selected, selection_label = select_memory_target(
        memory,
        {"normalized_prompt": "red cube", "target_name": "cube", "synonyms": ["block"]},
    )
    trace = build_selection_trace(
        memory=memory,
        selected=selected,
        selection_label=selection_label,
        parsed_query={"raw_query": "red cube", "normalized_prompt": "red cube", "target_name": "cube", "synonyms": ["block"]},
    )

    assert selected is not None
    assert selected.object_id == selected_object.object_id
    assert selection_label is None
    assert trace["selection"]["fallback_to_all_objects"] is True


def test_select_memory_target_replaces_single_view_geometry_outlier_within_margin() -> None:
    memory = ObjectMemory3D()
    outlier = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([-0.05, 0.02, 0.30], dtype=np.float32),
            label="cube",
            det_score=0.65,
            fused_2d_score=0.65,
            view_id="front",
            num_points=2200,
            depth_valid_ratio=1.0,
        )
    )
    supported = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.0, -0.03, 0.01], dtype=np.float32),
            label="cube",
            det_score=0.50,
            fused_2d_score=0.50,
            view_id="right",
            num_points=14000,
            depth_valid_ratio=1.0,
        )
    )

    selected, selection_label = select_memory_target(
        memory,
        {"normalized_prompt": "cube", "target_name": "cube", "synonyms": ["block"]},
    )
    trace = build_selection_trace(
        memory=memory,
        selected=selected,
        selection_label=selection_label,
        parsed_query={"raw_query": "cube", "normalized_prompt": "cube", "target_name": "cube", "synonyms": ["block"]},
    )

    assert selected is not None
    assert outlier.overall_confidence > supported.overall_confidence
    assert selected.object_id == supported.object_id
    assert trace["selection"]["selected_rank"] == 2
    assert "support sanity" in trace["selection"]["reason"]


def test_select_memory_target_keeps_geometry_outlier_when_confidence_gap_is_large() -> None:
    memory = ObjectMemory3D()
    outlier = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([-0.05, 0.02, 0.30], dtype=np.float32),
            label="cube",
            det_score=0.95,
            fused_2d_score=0.95,
            view_id="front",
            num_points=2200,
            depth_valid_ratio=1.0,
        )
    )
    memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.0, -0.03, 0.01], dtype=np.float32),
            label="cube",
            det_score=0.50,
            fused_2d_score=0.50,
            view_id="right",
            num_points=14000,
            depth_valid_ratio=1.0,
        )
    )

    selected, _ = select_memory_target(
        memory,
        {"normalized_prompt": "cube", "target_name": "cube", "synonyms": ["block"]},
    )

    assert selected is not None
    assert selected.object_id == outlier.object_id


def test_apply_selection_continuity_prefers_previous_object_within_margin() -> None:
    memory = ObjectMemory3D()
    preferred = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            label="cube",
            det_score=0.7,
            fused_2d_score=0.7,
            view_id="front",
            num_points=1000,
            depth_valid_ratio=1.0,
        )
    )
    winner = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.3, 0.0, 0.0], dtype=np.float32),
            label="cube",
            det_score=0.72,
            fused_2d_score=0.72,
            view_id="left",
            num_points=1000,
            depth_valid_ratio=1.0,
        )
    )

    selected, selection_label, diagnostics = apply_selection_continuity(
        memory=memory,
        parsed_query={"normalized_prompt": "cube", "target_name": "cube", "synonyms": ["block"]},
        selected=winner,
        selection_label="cube",
        preferred_object_id=preferred.object_id,
        max_confidence_gap=0.05,
    )

    assert selected is not None
    assert selected.object_id == preferred.object_id
    assert selection_label == "cube"
    assert diagnostics["applied"] is True
    assert diagnostics["reason"] == "kept_preferred_object_within_margin"


def test_apply_selection_continuity_respects_confidence_margin() -> None:
    memory = ObjectMemory3D()
    preferred = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            label="cube",
            det_score=0.5,
            fused_2d_score=0.5,
            view_id="front",
            num_points=1000,
            depth_valid_ratio=1.0,
        )
    )
    winner = memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.array([0.3, 0.0, 0.0], dtype=np.float32),
            label="cube",
            det_score=0.9,
            fused_2d_score=0.9,
            view_id="left",
            num_points=1000,
            depth_valid_ratio=1.0,
        )
    )

    selected, selection_label, diagnostics = apply_selection_continuity(
        memory=memory,
        parsed_query={"normalized_prompt": "cube", "target_name": "cube", "synonyms": ["block"]},
        selected=winner,
        selection_label="cube",
        preferred_object_id=preferred.object_id,
        max_confidence_gap=0.05,
    )

    assert selected is not None
    assert selected.object_id == winner.object_id
    assert selection_label == "cube"
    assert diagnostics["applied"] is False
    assert diagnostics["reason"] == "confidence_gap_exceeds_margin"
