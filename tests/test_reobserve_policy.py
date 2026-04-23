from __future__ import annotations

import numpy as np

from src.memory.object_memory_3d import ObjectMemory3D, ObjectObservation3D
from src.policy.reobserve_policy import ReobservePolicyConfig, decide_reobserve


def test_decide_reobserve_requests_view_when_no_object_is_selected() -> None:
    memory = ObjectMemory3D()

    decision = decide_reobserve(memory=memory, selected=None, candidate_view_ids=["front", "left"])

    assert decision.should_reobserve is True
    assert decision.reason == "no_selected_object"
    assert decision.suggested_view_ids == ["closer_front", "closer_left"]
    assert decision.diagnostics["suggested_view_plan"][0]["priority_reason"] == "increase_selected_view_support"
    assert decision.diagnostics["suggested_view_plan"][0]["requested_view_id"] == "front"
    assert decision.diagnostics["selected_object_id"] is None


def test_decide_reobserve_accepts_confident_multiview_selection() -> None:
    memory = ObjectMemory3D()
    selected = _add_object(memory, [0.0, 0.0, 0.0], "red cube", det=0.9, view_id="front", points=1000)
    _add_object(memory, [0.01, 0.0, 0.0], "red cube", det=0.9, view_id="left", points=1000)
    _add_object(memory, [0.02, 0.0, 0.0], "red cube", det=0.9, view_id="right", points=1000)
    assert selected.object_id == memory.objects[0].object_id

    decision = decide_reobserve(
        memory=memory,
        selected=memory.objects[0],
        selection_label="red cube",
        candidate_view_ids=["front", "left", "right"],
    )

    assert decision.should_reobserve is False
    assert decision.reason == "confident_enough"
    assert decision.suggested_view_ids == []
    assert decision.diagnostics["selected_num_views"] == 3


def test_decide_reobserve_flags_close_top_candidates() -> None:
    memory = ObjectMemory3D()
    obj_a = _add_object(memory, [0.0, 0.0, 0.0], "red cube", det=0.8, view_id="front", points=1000)
    _add_object(memory, [0.0, 0.01, 0.0], "red cube", det=0.8, view_id="left", points=1000)
    obj_b = _add_object(memory, [0.3, 0.0, 0.0], "red cube", det=0.79, view_id="front", points=1000)
    _add_object(memory, [0.3, 0.01, 0.0], "red cube", det=0.79, view_id="left", points=1000)

    decision = decide_reobserve(
        memory=memory,
        selected=obj_a,
        selection_label="red cube",
        config=ReobservePolicyConfig(min_confidence_gap=0.2),
        candidate_view_ids=["front", "left", "right"],
    )

    assert obj_b.object_id != obj_a.object_id
    assert decision.should_reobserve is True
    assert decision.reason == "ambiguous_top_candidates"
    assert decision.diagnostics["confidence_gap"] < 0.2
    assert decision.suggested_view_ids == ["closer_right"]
    assert decision.diagnostics["suggested_view_plan"][0]["source"] == "candidate_missing_support"
    assert decision.diagnostics["suggested_view_plan"][0]["requested_view_id"] == "right"


def test_decide_reobserve_flags_insufficient_view_support_before_confidence() -> None:
    memory = ObjectMemory3D()
    selected = _add_object(memory, [0.0, 0.0, 0.0], "red cube", det=0.95, view_id="front", points=1000)

    decision = decide_reobserve(
        memory=memory,
        selected=selected,
        selection_label="red cube",
        config=ReobservePolicyConfig(min_views=2, min_overall_confidence=0.99),
        candidate_view_ids=["front", "left", "right"],
    )

    assert decision.should_reobserve is True
    assert decision.reason == "insufficient_view_support"
    assert decision.suggested_view_ids == ["closer_left", "closer_right"]
    assert decision.diagnostics["suggested_view_plan"][0]["priority_reason"] == "increase_selected_view_support"


def test_decide_reobserve_deduplicates_suggested_views() -> None:
    memory = ObjectMemory3D()
    selected = _add_object(memory, [0.0, 0.0, 0.0], "object", det=0.95, view_id="front", points=1000)

    decision = decide_reobserve(
        memory=memory,
        selected=selected,
        selection_label="object",
        config=ReobservePolicyConfig(min_views=2, max_suggested_views=3),
        candidate_view_ids=["front", "left", "left", "right"],
    )

    assert decision.should_reobserve is True
    assert decision.suggested_view_ids == ["closer_left", "closer_right"]


def test_decide_reobserve_flags_low_point_support() -> None:
    memory = ObjectMemory3D()
    selected = _add_object(memory, [0.0, 0.0, 0.0], "red cube", det=0.9, view_id="front", points=10)
    _add_object(memory, [0.01, 0.0, 0.0], "red cube", det=0.9, view_id="left", points=10)

    decision = decide_reobserve(
        memory=memory,
        selected=selected,
        selection_label="red cube",
        config=ReobservePolicyConfig(min_mean_num_points=100.0),
        candidate_view_ids=["front", "left", "right"],
    )

    assert decision.should_reobserve is True
    assert decision.reason == "too_few_3d_points"
    assert decision.diagnostics["selected_mean_num_points"] == 10.0
    assert decision.suggested_view_ids == ["top_down", "closer_oblique"]
    assert decision.diagnostics["suggested_view_plan"][0]["priority_reason"] == "improve_geometry_evidence"


def _add_object(
    memory: ObjectMemory3D,
    xyz: list[float],
    label: str,
    det: float,
    view_id: str,
    points: int,
):
    return memory.add_observation(
        ObjectObservation3D(
            world_xyz=np.asarray(xyz, dtype=np.float32),
            label=label,
            det_score=det,
            fused_2d_score=det,
            view_id=view_id,
            num_points=points,
            depth_valid_ratio=1.0,
        )
    )
