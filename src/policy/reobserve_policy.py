"""Rule-based re-observation decisions for confidence-aware fusion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from src.memory.object_memory_3d import MemoryObject3D, ObjectMemory3D
from src.policy.target_selector import object_selection_sort_key


@dataclass(frozen=True)
class ReobservePolicyConfig:
    """Thresholds for the first deterministic re-observation policy."""

    min_overall_confidence: float = 0.50
    min_confidence_gap: float = 0.05
    min_views: int = 2
    min_geometry_confidence: float = 0.50
    min_mean_num_points: float = 100.0
    default_suggested_view_ids: tuple[str, ...] = ("top_down", "closer_oblique")
    max_suggested_views: int = 2

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable config."""

        return {
            "min_overall_confidence": float(self.min_overall_confidence),
            "min_confidence_gap": float(self.min_confidence_gap),
            "min_views": int(self.min_views),
            "min_geometry_confidence": float(self.min_geometry_confidence),
            "min_mean_num_points": float(self.min_mean_num_points),
            "default_suggested_view_ids": list(self.default_suggested_view_ids),
            "max_suggested_views": int(self.max_suggested_views),
        }


@dataclass(frozen=True)
class ReobserveDecision:
    """Structured rule-based re-observation decision."""

    should_reobserve: bool
    reason: str
    suggested_view_ids: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable decision."""

        return {
            "should_reobserve": bool(self.should_reobserve),
            "reason": self.reason,
            "suggested_view_ids": list(self.suggested_view_ids),
            "diagnostics": self.diagnostics,
        }


def decide_reobserve(
    memory: ObjectMemory3D,
    selected: MemoryObject3D | None,
    selection_label: str | None = None,
    config: ReobservePolicyConfig | None = None,
    candidate_view_ids: Sequence[str] | None = None,
) -> ReobserveDecision:
    """Decide whether another view would be useful.

    This policy intentionally returns a decision artifact only. It does not move
    cameras or rerun perception; that keeps the first milestone testable and
    ablation-friendly.
    """

    config = config or ReobservePolicyConfig()
    selection_pool = selection_pool_for_label(memory.objects, selection_label)
    ranked_pool = sorted(selection_pool, key=object_selection_sort_key)
    diagnostics = build_reobserve_diagnostics(
        memory=memory,
        selected=selected,
        selection_label=selection_label,
        ranked_pool=ranked_pool,
        config=config,
    )
    suggested_view_ids = suggest_reobserve_views(
        selected=selected,
        candidate_view_ids=candidate_view_ids,
        config=config,
    )

    if selected is None:
        return ReobserveDecision(
            should_reobserve=True,
            reason="no_selected_object",
            suggested_view_ids=suggested_view_ids,
            diagnostics=diagnostics,
        )

    if diagnostics["selected_num_views"] < config.min_views:
        return ReobserveDecision(
            should_reobserve=True,
            reason="insufficient_view_support",
            suggested_view_ids=suggested_view_ids,
            diagnostics=diagnostics,
        )

    confidence_gap = diagnostics.get("confidence_gap")
    if confidence_gap is not None and confidence_gap < config.min_confidence_gap:
        return ReobserveDecision(
            should_reobserve=True,
            reason="ambiguous_top_candidates",
            suggested_view_ids=suggested_view_ids,
            diagnostics=diagnostics,
        )

    if diagnostics["selected_overall_confidence"] < config.min_overall_confidence:
        return ReobserveDecision(
            should_reobserve=True,
            reason="low_overall_confidence",
            suggested_view_ids=suggested_view_ids,
            diagnostics=diagnostics,
        )

    if diagnostics["selected_geometry_confidence"] < config.min_geometry_confidence:
        return ReobserveDecision(
            should_reobserve=True,
            reason="low_geometry_confidence",
            suggested_view_ids=suggested_view_ids,
            diagnostics=diagnostics,
        )

    if diagnostics["selected_mean_num_points"] < config.min_mean_num_points:
        return ReobserveDecision(
            should_reobserve=True,
            reason="too_few_3d_points",
            suggested_view_ids=suggested_view_ids,
            diagnostics=diagnostics,
        )

    return ReobserveDecision(
        should_reobserve=False,
        reason="confident_enough",
        suggested_view_ids=[],
        diagnostics=diagnostics,
    )


def selection_pool_for_label(
    objects: Sequence[MemoryObject3D],
    selection_label: str | None,
) -> list[MemoryObject3D]:
    """Return objects eligible under the selected query label."""

    if selection_label is None:
        return list(objects)
    return [obj for obj in objects if selection_label in obj.label_votes]


def build_reobserve_diagnostics(
    memory: ObjectMemory3D,
    selected: MemoryObject3D | None,
    selection_label: str | None,
    ranked_pool: Sequence[MemoryObject3D],
    config: ReobservePolicyConfig,
) -> dict[str, Any]:
    """Build transparent numeric diagnostics for the rule decision."""

    top1 = ranked_pool[0] if ranked_pool else None
    top2 = ranked_pool[1] if len(ranked_pool) >= 2 else None
    selected_mean_num_points = mean_float(selected.metadata.get("num_points_history", [])) if selected else 0.0
    confidence_gap = (
        float(top1.overall_confidence) - float(top2.overall_confidence)
        if top1 is not None and top2 is not None
        else None
    )
    return {
        "num_memory_objects": len(memory.objects),
        "selection_pool_label": selection_label,
        "selection_pool_size": len(ranked_pool),
        "top1_object_id": None if top1 is None else top1.object_id,
        "top2_object_id": None if top2 is None else top2.object_id,
        "confidence_gap": confidence_gap,
        "selected_object_id": None if selected is None else selected.object_id,
        "selected_overall_confidence": 0.0 if selected is None else float(selected.overall_confidence),
        "selected_geometry_confidence": 0.0 if selected is None else float(selected.geometry_confidence),
        "selected_semantic_confidence": 0.0 if selected is None else float(selected.semantic_confidence),
        "selected_num_views": 0 if selected is None else len(selected.view_ids),
        "selected_view_ids": [] if selected is None else list(selected.view_ids),
        "selected_num_observations": 0 if selected is None else int(selected.num_observations),
        "selected_mean_num_points": selected_mean_num_points,
        "thresholds": config.to_json_dict(),
    }


def suggest_reobserve_views(
    selected: MemoryObject3D | None,
    candidate_view_ids: Sequence[str] | None,
    config: ReobservePolicyConfig,
) -> list[str]:
    """Suggest missing known views, falling back to abstract extra view labels."""

    used_views = set(selected.view_ids if selected is not None else [])
    cleaned_candidates = dedupe_view_ids(view_id for view_id in candidate_view_ids or [] if view_id)
    suggestions = [view_id for view_id in cleaned_candidates if view_id not in used_views]
    if not suggestions:
        suggestions = [
            view_id
            for view_id in dedupe_view_ids(config.default_suggested_view_ids)
            if view_id not in used_views
        ]
    return suggestions[: max(0, int(config.max_suggested_views))]


def dedupe_view_ids(view_ids: Sequence[Any]) -> list[str]:
    """Return non-empty view ids with stable de-duplication."""

    seen: set[str] = set()
    deduped: list[str] = []
    for view_id in view_ids:
        text = str(view_id or "").strip()
        if text and text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped


def mean_float(values: Sequence[Any]) -> float:
    """Return a numeric mean with zero fallback."""

    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)
