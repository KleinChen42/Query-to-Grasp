from __future__ import annotations

import numpy as np

from src.memory.object_memory_3d import ObjectMemory3D, ObjectObservation3D
from src.policy.target_selector import (
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
