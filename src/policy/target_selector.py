"""Deterministic target selection and trace rendering for 3D object memory."""

from __future__ import annotations

from typing import Any, Sequence

from src.memory.object_memory_3d import MemoryObject3D, ObjectMemory3D

GEOMETRY_SANITY_CONFIDENCE_MARGIN = 0.05
GEOMETRY_SANITY_MIN_Z_GAP = 0.15
GEOMETRY_SANITY_MIN_POINT_RATIO = 2.0
TRACE_REOBSERVE_POINT_FLOOR = 100.0
TRACE_REOBSERVE_VIEW_FLOOR = 2


def select_memory_target(
    memory: ObjectMemory3D,
    parsed_query: dict[str, Any],
) -> tuple[MemoryObject3D | None, str | None]:
    """Select the best object, trying exact query labels before falling back."""

    for label in candidate_selection_labels(parsed_query):
        selected = select_best_supported(memory.objects, label=label)
        if selected is not None:
            return selected, label
    return select_best_supported(memory.objects), None


def select_best_supported(
    objects: Sequence[MemoryObject3D],
    label: str | None = None,
) -> MemoryObject3D | None:
    """Select the best object, with a conservative geometry sanity check."""

    candidates = [
        obj
        for obj in objects
        if label is None or label in obj.label_votes
    ]
    if not candidates:
        return None

    ranked = sorted(candidates, key=object_selection_sort_key)
    return support_sanity_alternative(ranked) or ranked[0]


def support_sanity_alternative(ranked_candidates: Sequence[MemoryObject3D]) -> MemoryObject3D | None:
    """Prefer stronger tabletop support when the nominal winner is a likely 3D outlier."""

    if len(ranked_candidates) <= 1:
        return None

    winner = ranked_candidates[0]
    for candidate in ranked_candidates[1:]:
        if should_replace_geometry_outlier(winner, candidate):
            return candidate
    return None


def should_replace_geometry_outlier(winner: MemoryObject3D, candidate: MemoryObject3D) -> bool:
    """Return whether a near-tied candidate has much saner tabletop support."""

    confidence_gap = float(winner.overall_confidence) - float(candidate.overall_confidence)
    if confidence_gap < 0.0 or confidence_gap > GEOMETRY_SANITY_CONFIDENCE_MARGIN:
        return False
    return is_geometry_outlier_against(outlier=winner, supported=candidate)


def is_geometry_outlier_against(outlier: MemoryObject3D, supported: MemoryObject3D) -> bool:
    """Return whether one object is a weak high-z outlier beside a supported object."""

    if len(outlier.view_ids) > 1:
        return False

    outlier_z = float(outlier.world_xyz[2])
    supported_z = float(supported.world_xyz[2])
    if outlier_z - supported_z < GEOMETRY_SANITY_MIN_Z_GAP:
        return False

    outlier_points = mean_object_points(outlier)
    supported_points = mean_object_points(supported)
    if supported_points < max(1.0, outlier_points * GEOMETRY_SANITY_MIN_POINT_RATIO):
        return False

    return supported.geometry_confidence >= outlier.geometry_confidence - 0.05


def apply_selection_continuity(
    memory: ObjectMemory3D,
    parsed_query: dict[str, Any],
    selected: MemoryObject3D | None,
    selection_label: str | None,
    preferred_object_id: str | None,
    max_confidence_gap: float,
) -> tuple[MemoryObject3D | None, str | None, dict[str, Any]]:
    """Prefer the previous selected object when it stays competitively ranked."""

    diagnostics = {
        "preferred_object_id": preferred_object_id,
        "selected_object_id_before": None if selected is None else selected.object_id,
        "selected_selection_label_before": selection_label,
        "selected_overall_confidence_before": 0.0 if selected is None else float(selected.overall_confidence),
        "max_confidence_gap": float(max_confidence_gap),
        "applied": False,
        "reason": "disabled_or_missing_preferred_object",
    }
    preferred = memory.get_object_by_id(preferred_object_id)
    if preferred is None:
        diagnostics["reason"] = "preferred_object_missing"
        return selected, selection_label, diagnostics

    preferred_selection_label = preferred_selection_label_for_object(preferred, parsed_query)
    preferred_priority = selection_label_priority(parsed_query, preferred_selection_label)
    selected_priority = selection_label_priority(parsed_query, selection_label)
    diagnostics.update(
        {
            "preferred_top_label": preferred.top_label,
            "preferred_selection_label": preferred_selection_label,
            "preferred_overall_confidence": float(preferred.overall_confidence),
            "preferred_priority": preferred_priority,
            "selected_priority_before": selected_priority,
        }
    )

    if selected is None:
        if preferred_selection_label is None and candidate_selection_labels(parsed_query):
            diagnostics["reason"] = "preferred_object_not_query_eligible"
            return selected, selection_label, diagnostics
        diagnostics["applied"] = True
        diagnostics["reason"] = "preferred_object_restored"
        diagnostics["selected_object_id_after"] = preferred.object_id
        diagnostics["selected_selection_label_after"] = preferred_selection_label
        return preferred, preferred_selection_label, diagnostics

    if preferred.object_id == selected.object_id:
        diagnostics["reason"] = "preferred_already_selected"
        diagnostics["selected_object_id_after"] = selected.object_id
        diagnostics["selected_selection_label_after"] = selection_label
        return selected, selection_label, diagnostics

    if preferred_selection_label is None and selection_label is not None:
        diagnostics["reason"] = "preferred_object_not_eligible_for_selected_label"
        diagnostics["selected_object_id_after"] = selected.object_id
        diagnostics["selected_selection_label_after"] = selection_label
        return selected, selection_label, diagnostics

    if preferred_priority > selected_priority:
        diagnostics["reason"] = "preferred_object_has_lower_label_priority"
        diagnostics["selected_object_id_after"] = selected.object_id
        diagnostics["selected_selection_label_after"] = selection_label
        return selected, selection_label, diagnostics

    confidence_gap = float(selected.overall_confidence) - float(preferred.overall_confidence)
    diagnostics["confidence_gap_to_selected"] = confidence_gap
    if confidence_gap > float(max_confidence_gap):
        diagnostics["reason"] = "confidence_gap_exceeds_margin"
        diagnostics["selected_object_id_after"] = selected.object_id
        diagnostics["selected_selection_label_after"] = selection_label
        return selected, selection_label, diagnostics

    result_label = selection_label
    if preferred_priority < selected_priority:
        result_label = preferred_selection_label
    elif result_label is None:
        result_label = preferred_selection_label
    diagnostics["applied"] = True
    diagnostics["reason"] = "kept_preferred_object_within_margin"
    diagnostics["selected_object_id_after"] = preferred.object_id
    diagnostics["selected_selection_label_after"] = result_label
    return preferred, result_label, diagnostics


def build_selection_trace(
    memory: ObjectMemory3D,
    selected: MemoryObject3D | None,
    selection_label: str | None,
    parsed_query: dict[str, Any],
) -> dict[str, Any]:
    """Build a concise, paper-friendly explanation of target selection."""

    objects = memory.objects
    selected_id = None if selected is None else selected.object_id
    selection_pool = [
        obj
        for obj in objects
        if selection_label is None or selection_label in obj.label_votes
    ]
    ranked_pool = sorted(selection_pool, key=object_selection_sort_key)
    ranked_all = sorted(objects, key=object_selection_sort_key)
    candidate_labels = candidate_selection_labels(parsed_query)
    query_attributes = parsed_query_attributes(parsed_query)

    selected_rank = _rank_for_object(ranked_pool, selected_id)
    selected_attribute_coverage = (
        None if selected is None else object_attribute_diagnostics(selected, parsed_query)["attribute_coverage"]
    )
    if selected is None:
        reason = "No memory object was available for selection."
    elif selection_label is None:
        reason = (
            "No candidate query label matched the memory. Selected the highest "
            "confidence object across all memory objects."
        )
    elif selected_rank is not None and selected_rank > 1:
        reason = (
            f"Selected {selected.object_id} from {len(selection_pool)} object(s) "
            f"containing label {selection_label!r} after support sanity favored a "
            "near-tied object with stronger tabletop geometry."
        )
    else:
        reason = (
            f"Selected {selected.object_id} from {len(selection_pool)} object(s) "
            f"containing label {selection_label!r}, ranked by confidence and "
            "deterministic tie-breaks."
        )

    return {
        "query": {
            "raw_query": parsed_query.get("raw_query"),
            "normalized_prompt": parsed_query.get("normalized_prompt"),
            "target_name": parsed_query.get("target_name"),
            "attributes": query_attributes,
            "candidate_selection_labels": candidate_labels,
        },
        "selection": {
            "selected_object_id": selected_id,
            "selected_rank": selected_rank,
            "selection_pool_label": selection_label,
            "selection_pool_size": len(selection_pool),
            "same_phrase_competitor_count": count_same_phrase_competitors(
                objects=objects,
                parsed_query=parsed_query,
                selected_object_id=selected_id,
            ),
            "selected_attribute_coverage": selected_attribute_coverage,
            "fallback_to_all_objects": selected is not None and selection_label is None,
            "tie_break_order": [
                "overall_confidence desc",
                "num_unique_views desc",
                "geometry_confidence desc",
                "object_id asc",
            ],
            "reason": reason,
        },
        "ranked_selection_pool": [
            memory_object_trace_row(
                obj,
                rank=index + 1,
                selected_object_id=selected_id,
                selection_label=selection_label,
                parsed_query=parsed_query,
            )
            for index, obj in enumerate(ranked_pool)
        ],
        "all_memory_objects": [
            memory_object_trace_row(
                obj,
                rank=index + 1,
                selected_object_id=selected_id,
                selection_label=selection_label,
                parsed_query=parsed_query,
            )
            for index, obj in enumerate(ranked_all)
        ],
    }


def memory_object_trace_row(
    obj: MemoryObject3D,
    rank: int,
    selected_object_id: str | None,
    selection_label: str | None,
    parsed_query: dict[str, Any],
) -> dict[str, Any]:
    """Return selection-relevant fields for one memory object."""

    num_points_history = obj.metadata.get("num_points_history", [])
    depth_valid_ratio_history = obj.metadata.get("depth_valid_ratio_history", [])
    mean_num_points = _mean_floats(num_points_history)
    attribute_diagnostics = object_attribute_diagnostics(obj, parsed_query)
    return {
        "rank": int(rank),
        "object_id": obj.object_id,
        "is_selected": obj.object_id == selected_object_id,
        "eligible_for_selection": selection_label is None or selection_label in obj.label_votes,
        "selection_label_vote": None if selection_label is None else float(obj.label_votes.get(selection_label, 0.0)),
        "top_label": obj.top_label,
        "label_votes": {label: float(score) for label, score in sorted(obj.label_votes.items())},
        "world_xyz": obj.world_xyz.astype(float).tolist(),
        "view_ids": list(obj.view_ids),
        "num_unique_views": len(obj.view_ids),
        "num_observations": int(obj.num_observations),
        "overall_confidence": float(obj.overall_confidence),
        "semantic_confidence": float(obj.semantic_confidence),
        "geometry_confidence": float(obj.geometry_confidence),
        "score_terms": obj.score_terms.to_json_dict(),
        "fusion_contributions": {
            key: float(value)
            for key, value in obj.fusion_trace.get("contributions", {}).items()
        },
        "mean_det_score": _mean_floats(obj.det_scores),
        "mean_clip_score": _mean_floats(obj.clip_scores),
        "mean_fused_2d_score": _mean_floats(obj.fused_2d_scores),
        "mean_num_points": mean_num_points,
        "mean_depth_valid_ratio": _mean_floats(depth_valid_ratio_history),
        "has_full_prompt_label": attribute_diagnostics["has_full_prompt_label"],
        "has_target_label": attribute_diagnostics["has_target_label"],
        "query_attribute_hits": attribute_diagnostics["query_attribute_hits"],
        "query_attribute_misses": attribute_diagnostics["query_attribute_misses"],
        "labels_matching_all_query_attributes": attribute_diagnostics["labels_matching_all_query_attributes"],
        "attribute_coverage": attribute_diagnostics["attribute_coverage"],
        "below_default_reobserve_point_floor": mean_num_points < TRACE_REOBSERVE_POINT_FLOOR,
        "below_default_reobserve_view_floor": len(obj.view_ids) < TRACE_REOBSERVE_VIEW_FLOOR,
        "selection_sort_key": {
            "overall_confidence_desc": float(obj.overall_confidence),
            "num_unique_views_desc": len(obj.view_ids),
            "geometry_confidence_desc": float(obj.geometry_confidence),
            "object_id_asc": obj.object_id,
        },
    }


def render_selection_trace_markdown(trace: dict[str, Any]) -> str:
    """Render a compact Markdown explanation of target selection."""

    query = trace["query"]
    selection = trace["selection"]
    lines = [
        "# Selection Trace",
        "",
        f"- Query: `{query.get('raw_query')}`",
        f"- Normalized prompt: `{query.get('normalized_prompt')}`",
        f"- Attributes: `{_format_string_list(query.get('attributes', []))}`",
        f"- Candidate labels: `{', '.join(query.get('candidate_selection_labels', []))}`",
        f"- Selected object: `{selection.get('selected_object_id')}`",
        f"- Selection pool label: `{selection.get('selection_pool_label')}`",
        f"- Selection pool size: {selection.get('selection_pool_size')}",
        f"- Same-phrase competitors: {selection.get('same_phrase_competitor_count')}",
        f"- Selected attribute coverage: {_format_optional_float(selection.get('selected_attribute_coverage'))}",
        f"- Fallback to all objects: {selection.get('fallback_to_all_objects')}",
        f"- Reason: {selection.get('reason')}",
        "",
        "## Ranked Selection Pool",
        "",
        "| rank | selected | object_id | top_label | overall | semantic | geometry | views | observations | attr_cov | attr_hits | low_points | low_views | label_votes | world_xyz |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in trace.get("ranked_selection_pool", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["rank"]),
                    "yes" if row["is_selected"] else "no",
                    str(row["object_id"]),
                    str(row["top_label"]),
                    _format_float(row["overall_confidence"]),
                    _format_float(row["semantic_confidence"]),
                    _format_float(row["geometry_confidence"]),
                    str(row["num_unique_views"]),
                    str(row["num_observations"]),
                    _format_float(row["attribute_coverage"]),
                    _format_string_list(row["query_attribute_hits"]),
                    str(row["below_default_reobserve_point_floor"]),
                    str(row["below_default_reobserve_view_floor"]),
                    _format_label_votes(row["label_votes"]),
                    _format_vector(row["world_xyz"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def candidate_selection_labels(parsed_query: dict[str, Any]) -> list[str]:
    """Return query labels tried by the target selector in priority order."""

    return _dedupe_strings(
        [
            parsed_query.get("normalized_prompt"),
            parsed_query.get("target_name"),
            *list(parsed_query.get("synonyms", [])),
        ]
    )


def parsed_query_attributes(parsed_query: dict[str, Any]) -> list[str]:
    """Return parsed query attributes in stable trace order."""

    return _dedupe_strings(parsed_query.get("attributes", []))


def object_attribute_diagnostics(
    obj: MemoryObject3D,
    parsed_query: dict[str, Any],
) -> dict[str, Any]:
    """Return diagnostic-only attribute evidence fields for one object."""

    attributes = parsed_query_attributes(parsed_query)
    label_votes = obj.label_votes
    labels = sorted(label_votes)
    matching_attribute_labels = [
        label
        for label in labels
        if attributes and all(label_contains_attribute(label, attribute) for attribute in attributes)
    ]
    hits = [
        attribute
        for attribute in attributes
        if any(label_contains_attribute(label, attribute) for label in labels)
    ]
    misses = [attribute for attribute in attributes if attribute not in hits]
    return {
        "has_full_prompt_label": object_has_label(obj, parsed_query.get("normalized_prompt")),
        "has_target_label": object_has_label(obj, parsed_query.get("target_name")),
        "query_attribute_hits": hits,
        "query_attribute_misses": misses,
        "labels_matching_all_query_attributes": matching_attribute_labels,
        "attribute_coverage": 1.0 if not attributes else len(hits) / len(attributes),
    }


def count_same_phrase_competitors(
    objects: Sequence[MemoryObject3D],
    parsed_query: dict[str, Any],
    selected_object_id: str | None,
) -> int:
    """Count non-selected objects carrying the full normalized query label."""

    normalized_prompt = parsed_query.get("normalized_prompt")
    if not str(normalized_prompt or "").strip():
        return 0
    return sum(
        1
        for obj in objects
        if obj.object_id != selected_object_id and object_has_label(obj, normalized_prompt)
    )


def preferred_selection_label_for_object(
    obj: MemoryObject3D,
    parsed_query: dict[str, Any],
) -> str | None:
    """Return the highest-priority query label present in one object."""

    for label in candidate_selection_labels(parsed_query):
        if label in obj.label_votes:
            return label
    return None


def selection_label_priority(parsed_query: dict[str, Any], selection_label: str | None) -> int:
    """Return the query-label priority index used for continuity checks."""

    labels = candidate_selection_labels(parsed_query)
    if selection_label is None:
        return len(labels)
    try:
        return labels.index(selection_label)
    except ValueError:
        return len(labels) + 1


def object_selection_sort_key(obj: MemoryObject3D) -> tuple[float, int, float, str]:
    """Return the deterministic object selection sort key."""

    return (
        -float(obj.overall_confidence),
        -len(obj.view_ids),
        -float(obj.geometry_confidence),
        obj.object_id,
    )


def mean_object_points(obj: MemoryObject3D) -> float:
    """Return the mean 3D point support recorded for one object."""

    return _mean_floats(obj.metadata.get("num_points_history", []))


def _rank_for_object(objects: Sequence[MemoryObject3D], object_id: str | None) -> int | None:
    if object_id is None:
        return None
    for index, obj in enumerate(objects, start=1):
        if obj.object_id == object_id:
            return index
    return None


def _dedupe_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped


def object_has_label(obj: MemoryObject3D, label: Any) -> bool:
    """Return whether an object has a label after conservative normalization."""

    normalized_label = normalize_label(label)
    if not normalized_label:
        return False
    return any(normalize_label(candidate) == normalized_label for candidate in obj.label_votes)


def label_contains_attribute(label: Any, attribute: Any) -> bool:
    """Return whether a label contains one parsed attribute as a token."""

    normalized_attribute = normalize_label(attribute)
    if not normalized_attribute:
        return False
    return normalized_attribute in label_tokens(label)


def label_tokens(label: Any) -> set[str]:
    """Return simple lowercase alphanumeric-ish tokens for trace diagnostics."""

    return set(normalize_label(label).replace("-", " ").replace("_", " ").split())


def normalize_label(label: Any) -> str:
    """Normalize labels for diagnostic comparisons without changing label votes."""

    return " ".join(str(label or "").strip().lower().split())


def _mean_floats(values: Sequence[Any]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def _format_float(value: Any) -> str:
    return f"{float(value):.4f}"


def _format_optional_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return _format_float(value)


def _format_string_list(values: Sequence[Any]) -> str:
    values_list = [str(value) for value in values if str(value)]
    if not values_list:
        return "n/a"
    return ", ".join(values_list)


def _format_label_votes(label_votes: dict[str, float]) -> str:
    if not label_votes:
        return "n/a"
    return ", ".join(f"{label}:{float(score):.3f}" for label, score in sorted(label_votes.items()))


def _format_vector(value: Sequence[Any]) -> str:
    return "[" + ", ".join(f"{float(item):.3f}" for item in value) + "]"
