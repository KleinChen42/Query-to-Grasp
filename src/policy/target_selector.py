"""Deterministic target selection and trace rendering for 3D object memory."""

from __future__ import annotations

from typing import Any, Sequence

from src.memory.object_memory_3d import MemoryObject3D, ObjectMemory3D


def select_memory_target(
    memory: ObjectMemory3D,
    parsed_query: dict[str, Any],
) -> tuple[MemoryObject3D | None, str | None]:
    """Select the best object, trying exact query labels before falling back."""

    for label in candidate_selection_labels(parsed_query):
        selected = memory.select_best(label=label)
        if selected is not None:
            return selected, label
    return memory.select_best(), None


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

    if selected is None:
        reason = "No memory object was available for selection."
    elif selection_label is None:
        reason = (
            "No candidate query label matched the memory. Selected the highest "
            "confidence object across all memory objects."
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
            "candidate_selection_labels": candidate_labels,
        },
        "selection": {
            "selected_object_id": selected_id,
            "selected_rank": _rank_for_object(ranked_pool, selected_id),
            "selection_pool_label": selection_label,
            "selection_pool_size": len(selection_pool),
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
            )
            for index, obj in enumerate(ranked_pool)
        ],
        "all_memory_objects": [
            memory_object_trace_row(
                obj,
                rank=index + 1,
                selected_object_id=selected_id,
                selection_label=selection_label,
            )
            for index, obj in enumerate(ranked_all)
        ],
    }


def memory_object_trace_row(
    obj: MemoryObject3D,
    rank: int,
    selected_object_id: str | None,
    selection_label: str | None,
) -> dict[str, Any]:
    """Return selection-relevant fields for one memory object."""

    num_points_history = obj.metadata.get("num_points_history", [])
    depth_valid_ratio_history = obj.metadata.get("depth_valid_ratio_history", [])
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
        "mean_num_points": _mean_floats(num_points_history),
        "mean_depth_valid_ratio": _mean_floats(depth_valid_ratio_history),
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
        f"- Candidate labels: `{', '.join(query.get('candidate_selection_labels', []))}`",
        f"- Selected object: `{selection.get('selected_object_id')}`",
        f"- Selection pool label: `{selection.get('selection_pool_label')}`",
        f"- Selection pool size: {selection.get('selection_pool_size')}",
        f"- Fallback to all objects: {selection.get('fallback_to_all_objects')}",
        f"- Reason: {selection.get('reason')}",
        "",
        "## Ranked Selection Pool",
        "",
        "| rank | selected | object_id | top_label | overall | semantic | geometry | views | observations | label_votes | world_xyz |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
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


def object_selection_sort_key(obj: MemoryObject3D) -> tuple[float, int, float, str]:
    """Return the deterministic object selection sort key."""

    return (
        -float(obj.overall_confidence),
        -len(obj.view_ids),
        -float(obj.geometry_confidence),
        obj.object_id,
    )


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


def _mean_floats(values: Sequence[Any]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def _format_float(value: Any) -> str:
    return f"{float(value):.4f}"


def _format_label_votes(label_votes: dict[str, float]) -> str:
    if not label_votes:
        return "n/a"
    return ", ".join(f"{label}:{float(score):.3f}" for label, score in sorted(label_votes.items()))


def _format_vector(value: Sequence[Any]) -> str:
    return "[" + ", ".join(f"{float(item):.3f}" for item in value) + "]"
