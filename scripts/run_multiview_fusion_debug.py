"""Run a minimal multi-view perception pass and fuse 3D object memory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_single_view_query import build_clip_prompts, rank_detections_without_clip  # noqa: E402
from src.env.camera_utils import ObservationFrame  # noqa: E402
from src.env.maniskill_env import ManiSkillScene  # noqa: E402
from src.io.export_utils import export_observation_frame, write_json  # noqa: E402
from src.memory.fusion import FusionWeights  # noqa: E402
from src.memory.object_memory_3d import (  # noqa: E402
    MemoryObject3D,
    ObjectMemory3D,
    ObjectMemoryConfig,
    ObjectObservation3D,
)
from src.policy.reobserve_policy import ReobserveDecision, ReobservePolicyConfig, decide_reobserve  # noqa: E402
from src.policy.target_selector import (  # noqa: E402
    apply_selection_continuity,
    build_selection_trace,
    render_selection_trace_markdown,
    select_memory_target,
)
from src.perception.clip_rerank import RankedCandidate, rerank_candidates_with_clip  # noqa: E402
from src.perception.grounding_dino import DetectionCandidate, detect_candidates  # noqa: E402
from src.perception.mask_projector import Candidate3D, lift_box_to_3d  # noqa: E402
from src.perception.query_parser import parse_query  # noqa: E402

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VirtualCameraView:
    """One synthetic camera pose used to recapture an existing ManiSkill sensor."""

    label: str
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)


VIEW_PRESETS: dict[str, tuple[VirtualCameraView, ...]] = {
    "tabletop_3": (
        VirtualCameraView(label="front", eye=(0.35, 0.0, 0.55), target=(0.0, 0.0, 0.05)),
        VirtualCameraView(label="left", eye=(0.0, 0.35, 0.55), target=(0.0, 0.0, 0.05)),
        VirtualCameraView(label="right", eye=(0.0, -0.35, 0.55), target=(0.0, 0.0, 0.05)),
    )
}

REOBSERVE_VIEW_POSES: dict[str, VirtualCameraView] = {
    "closer_front": VirtualCameraView(
        label="closer_front",
        eye=(0.24, 0.0, 0.42),
        target=(0.0, 0.0, 0.05),
    ),
    "closer_left": VirtualCameraView(
        label="closer_left",
        eye=(0.0, 0.24, 0.42),
        target=(0.0, 0.0, 0.05),
    ),
    "closer_right": VirtualCameraView(
        label="closer_right",
        eye=(0.0, -0.24, 0.42),
        target=(0.0, 0.0, 0.05),
    ),
    "top_down": VirtualCameraView(
        label="top_down",
        eye=(0.0, 0.0, 0.75),
        target=(0.0, 0.0, 0.05),
        up=(1.0, 0.0, 0.0),
    ),
    "closer_oblique": VirtualCameraView(
        label="closer_oblique",
        eye=(0.22, -0.22, 0.38),
        target=(0.0, 0.0, 0.05),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug multi-view 3D semantic fusion on one ManiSkill reset.")
    parser.add_argument("--query", default="red cube", help="Natural language target query.")
    parser.add_argument("--view-ids", nargs="*", default=None, help="Camera keys to extract from the reset observation.")
    parser.add_argument(
        "--view-preset",
        default="none",
        choices=["none", *sorted(VIEW_PRESETS)],
        help="Optional virtual camera pose preset. Reuses --camera-name, defaulting to base_camera.",
    )
    parser.add_argument("--env-id", default="PickCube-v1", help="ManiSkill environment id.")
    parser.add_argument("--obs-mode", default="rgbd", help="ManiSkill observation mode.")
    parser.add_argument("--control-mode", default=None, help="Optional ManiSkill control mode.")
    parser.add_argument(
        "--pick-executor",
        default="placeholder",
        choices=["placeholder", "sim_topdown"],
        help="Optional pick executor for the final selected fused object.",
    )
    parser.add_argument(
        "--grasp-target-mode",
        default="semantic",
        choices=["semantic", "refined"],
        help=(
            "Target mode for pick execution. Multi-view currently uses the selected fused "
            "object world_xyz for both modes and records that source explicitly."
        ),
    )
    parser.add_argument("--camera-name", default=None, help="Camera key to extract or recapture for preset views.")
    parser.add_argument("--seed", type=int, default=0, help="Environment reset seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "multiview_fusion_debug")
    parser.add_argument("--prefer-llm-parser", action="store_true", help="Try LLM parsing before deterministic rules.")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="GroundingDINO box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold.")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum number of detector candidates per view.")
    parser.add_argument(
        "--detector-backend",
        default="auto",
        choices=["auto", "hf", "transformers", "groundingdino", "original", "mock"],
    )
    parser.add_argument("--mock-box-position", default="center", choices=["center", "left", "right", "all"])
    parser.add_argument("--detector-model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--model-config-path", type=Path, default=None)
    parser.add_argument("--model-checkpoint-path", type=Path, default=None)
    parser.add_argument("--clip-model-name", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--skip-clip", dest="skip_clip", action="store_true", default=False, help="Skip CLIP and rank by detector score.")
    parser.add_argument("--use-clip", dest="skip_clip", action="store_false", help="Run CLIP reranking. This is the default unless --skip-clip is set.")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda or cpu.")
    parser.add_argument("--detector-weight", type=float, default=0.5)
    parser.add_argument("--clip-weight", type=float, default=0.5)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--fallback-fov-degrees", type=float, default=60.0)
    parser.add_argument("--use-segmentation", action="store_true")
    parser.add_argument("--segmentation-id", type=int, default=None)
    parser.add_argument("--save-candidate-pointclouds", action="store_true")
    parser.add_argument("--merge-distance", type=float, default=0.08)
    parser.add_argument("--min-points-full-confidence", type=int, default=1000)
    parser.add_argument("--max-views-full-confidence", type=int, default=3)
    parser.add_argument("--fusion-det-weight", type=float, default=0.30)
    parser.add_argument("--fusion-clip-weight", type=float, default=0.30)
    parser.add_argument("--fusion-view-weight", type=float, default=0.15)
    parser.add_argument("--fusion-consistency-weight", type=float, default=0.15)
    parser.add_argument("--fusion-geometry-weight", type=float, default=0.10)
    parser.add_argument("--reobserve-min-confidence", type=float, default=0.50)
    parser.add_argument("--reobserve-min-confidence-gap", type=float, default=0.05)
    parser.add_argument("--reobserve-min-views", type=int, default=2)
    parser.add_argument("--reobserve-min-geometry-confidence", type=float, default=0.50)
    parser.add_argument("--reobserve-min-mean-points", type=float, default=100.0)
    parser.add_argument("--reobserve-suggested-view-ids", nargs="*", default=["top_down", "closer_oblique"])
    parser.add_argument("--reobserve-max-suggested-views", type=int, default=2)
    parser.add_argument(
        "--enable-closed-loop-reobserve",
        action="store_true",
        help="If the policy requests another view, capture one suggested virtual view and update memory.",
    )
    parser.add_argument(
        "--closed-loop-max-extra-views",
        type=int,
        default=1,
        help="Maximum suggested virtual views to execute when closed-loop re-observation is enabled.",
    )
    parser.add_argument(
        "--enable-selected-object-continuity",
        action="store_true",
        help="Prefer merging extra-view observations back into the initially selected object when geometry is compatible.",
    )
    parser.add_argument(
        "--selected-object-continuity-distance-scale",
        type=float,
        default=1.0,
        help="Compatibility distance for selected-object continuity as a multiple of --merge-distance.",
    )
    parser.add_argument(
        "--enable-post-reobserve-selection-continuity",
        action="store_true",
        help="Prefer keeping the initial selected object after re-observation when it remains competitive.",
    )
    parser.add_argument(
        "--post-reobserve-selection-margin",
        type=float,
        default=0.03,
        help="Maximum confidence gap allowed when keeping the initial selected object after re-observation.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pick_executor == "sim_topdown" and args.control_mode is None:
        args.control_mode = "pd_ee_delta_pos"
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    start_time = time.perf_counter()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{timestamp}_{_slug(args.query)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    parsed_query = parse_query(args.query, prefer_llm=args.prefer_llm_parser)
    clip_prompts = build_clip_prompts(parsed_query)
    memory = ObjectMemory3D(build_memory_config(args))
    view_ids = normalize_view_ids(args.view_ids, args.camera_name)

    write_json(parsed_query, run_dir / "parsed_query.json")

    scene = ManiSkillScene(
        env_name=args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        camera_name=args.camera_name,
    )
    try:
        scene.reset(seed=args.seed)
        frames = collect_frames(
            scene,
            view_ids=view_ids,
            view_preset=args.view_preset,
            preset_camera_name=args.camera_name or "base_camera",
        )
        view_results: list[dict[str, Any]] = []
        total_observations_added = 0
        for view_id, frame in frames:
            view_result = process_view(
                args=args,
                frame=frame,
                view_id=view_id,
                parsed_query=parsed_query,
                clip_prompts=clip_prompts,
                memory=memory,
                run_dir=run_dir,
            )
            view_results.append(view_result)
            total_observations_added += int(view_result["num_observations_added"])

        initial_view_ids = [view_id for view_id, _ in frames]
        selected, selection_label = select_memory_target(memory, parsed_query)
        initial_decision = decide_reobserve(
            memory=memory,
            selected=selected,
            selection_label=selection_label,
            config=build_reobserve_config(args),
            candidate_view_ids=initial_view_ids,
        )
        initial_snapshot = build_reobserve_stage_snapshot(
            stage="before",
            memory=memory,
            selected=selected,
            selection_label=selection_label,
            decision=initial_decision,
            view_ids=initial_view_ids,
            total_observations_added=total_observations_added,
        )
        if args.enable_closed_loop_reobserve:
            write_json(
                build_memory_state(
                    memory=memory,
                    selected=selected,
                    selection_label=selection_label,
                    parsed_query=parsed_query,
                    view_results=view_results,
                ),
                run_dir / "memory_state_before_reobserve.json",
            )
            initial_selection_trace = build_selection_trace(
                memory=memory,
                selected=selected,
                selection_label=selection_label,
                parsed_query=parsed_query,
            )
            write_json(initial_selection_trace, run_dir / "selection_trace_before_reobserve.json")
            (run_dir / "selection_trace_before_reobserve.md").write_text(
                render_selection_trace_markdown(initial_selection_trace),
                encoding="utf-8",
            )
            write_json(initial_decision.to_json_dict(), run_dir / "reobserve_decision_before.json")
        extra_view_results: list[dict[str, Any]] = []
        extra_view_ids: list[str] = []
        closed_loop_executed = False
        continuity_target_object_id = (
            initial_snapshot["selected_object_id"] if args.enable_selected_object_continuity else None
        )
        continuity_merge_distance = (
            float(args.merge_distance) * float(args.selected_object_continuity_distance_scale)
            if args.enable_selected_object_continuity
            else None
        )
        if args.enable_closed_loop_reobserve and initial_decision.should_reobserve:
            extra_frames = collect_reobserve_frames(
                scene=scene,
                suggested_view_ids=initial_decision.suggested_view_ids,
                camera_name=args.camera_name or "base_camera",
                max_views=args.closed_loop_max_extra_views,
            )
            for view_id, frame in extra_frames:
                closed_loop_executed = True
                view_result = process_view(
                    args=args,
                    frame=frame,
                    view_id=view_id,
                    parsed_query=parsed_query,
                    clip_prompts=clip_prompts,
                    memory=memory,
                    run_dir=run_dir,
                    preferred_object_id=continuity_target_object_id,
                    preferred_merge_distance=continuity_merge_distance,
                )
                view_results.append(view_result)
                extra_view_results.append(view_result)
                extra_view_ids.append(view_id)
                total_observations_added += int(view_result["num_observations_added"])

        final_view_ids = [*initial_view_ids, *extra_view_ids]
        selected_object_followup_preselection = build_initial_selected_object_followup(
            before=initial_snapshot,
            after=None,
            memory=memory,
            extra_view_ids=extra_view_ids,
        )
        base_selected, base_selection_label = select_memory_target(memory, parsed_query)
        selected = base_selected
        selection_label = base_selection_label
        extra_view_absorber_object_ids = collect_extra_view_absorber_object_ids(extra_view_results)
        post_selection_continuity = build_post_selection_continuity_trace(
            args=args,
            initial_snapshot=initial_snapshot,
            selected_object_followup=selected_object_followup_preselection,
            base_selected=base_selected,
            base_selection_label=base_selection_label,
            extra_view_absorber_object_ids=extra_view_absorber_object_ids,
        )
        if post_selection_continuity["eligible"]:
            selected, selection_label, continuity_diagnostics = apply_selection_continuity(
                memory=memory,
                parsed_query=parsed_query,
                selected=base_selected,
                selection_label=base_selection_label,
                preferred_object_id=initial_snapshot["selected_object_id"],
                max_confidence_gap=float(args.post_reobserve_selection_margin),
            )
            post_selection_continuity = {
                **post_selection_continuity,
                **continuity_diagnostics,
                "applied": bool(continuity_diagnostics.get("applied")),
                "reason": str(continuity_diagnostics.get("reason") or "not_applied"),
                "selected_object_id_after": None if selected is None else selected.object_id,
                "selected_selection_label_after": selection_label,
            }
        final_decision = decide_reobserve(
            memory=memory,
            selected=selected,
            selection_label=selection_label,
            config=build_reobserve_config(args),
            candidate_view_ids=final_view_ids,
        )
        final_snapshot = build_reobserve_stage_snapshot(
            stage="after",
            memory=memory,
            selected=selected,
            selection_label=selection_label,
            decision=final_decision,
            view_ids=final_view_ids,
            total_observations_added=total_observations_added,
        )
        closed_loop_delta = build_closed_loop_delta(initial_snapshot, final_snapshot)
        selected_object_followup = build_initial_selected_object_followup(
            before=initial_snapshot,
            after=final_snapshot,
            memory=memory,
            extra_view_ids=extra_view_ids,
        )
        absorber_trace = build_closed_loop_absorber_trace(
            before=initial_snapshot,
            after=final_snapshot,
            extra_view_results=extra_view_results,
        )
        preferred_merge_trace = build_closed_loop_preferred_merge_trace(extra_view_results)
        if args.enable_closed_loop_reobserve:
            write_closed_loop_reobserve_artifacts(
                run_dir=run_dir,
                memory=memory,
                parsed_query=parsed_query,
                view_results=view_results,
                initial_snapshot=initial_snapshot,
                final_snapshot=final_snapshot,
                extra_view_results=extra_view_results,
                final_selected=selected,
                final_selection_label=selection_label,
                final_decision=final_decision,
                post_selection_continuity=post_selection_continuity,
            )

        memory_state = build_memory_state(
            memory=memory,
            selected=selected,
            selection_label=selection_label,
            parsed_query=parsed_query,
            view_results=view_results,
        )
        write_json(memory_state, run_dir / "memory_state.json")
        selection_trace = build_selection_trace(
            memory=memory,
            selected=selected,
            selection_label=selection_label,
            parsed_query=parsed_query,
        )
        selection_trace_json = run_dir / "selection_trace.json"
        selection_trace_md = run_dir / "selection_trace.md"
        write_json(selection_trace, selection_trace_json)
        selection_trace_md.write_text(render_selection_trace_markdown(selection_trace), encoding="utf-8")
        reobserve_decision_path = run_dir / "reobserve_decision.json"
        write_json(final_decision.to_json_dict(), reobserve_decision_path)

        pick_result = execute_selected_memory_pick(scene=scene, selected=selected, args=args)
        pick_result_path = run_dir / "pick_result.json"
        write_json(pick_result, pick_result_path)

        summary = build_summary(
            args=args,
            parsed_query=parsed_query,
            view_ids=final_view_ids,
            memory=memory,
            selected=selected,
            selection_label=selection_label,
            total_observations_added=total_observations_added,
            runtime_seconds=time.perf_counter() - start_time,
            run_dir=run_dir,
            pick_result=pick_result,
        )
        summary["selection_trace_json"] = str(selection_trace_json)
        summary["selection_trace_md"] = str(selection_trace_md)
        summary["reobserve_decision_json"] = str(reobserve_decision_path)
        summary["pick_result_json"] = str(pick_result_path)
        summary["should_reobserve"] = bool(final_decision.should_reobserve)
        summary["reobserve_reason"] = final_decision.reason
        summary["initial_should_reobserve"] = bool(initial_decision.should_reobserve)
        summary["initial_reobserve_reason"] = initial_decision.reason
        summary["initial_selected_object_id"] = initial_snapshot["selected_object_id"]
        summary["initial_selected_overall_confidence"] = initial_snapshot["selected_overall_confidence"]
        summary["initial_selected_num_views"] = initial_snapshot["selected_num_views"]
        summary["initial_selected_num_observations"] = initial_snapshot["selected_num_observations"]
        summary["initial_num_memory_objects"] = initial_snapshot["num_memory_objects"]
        summary["final_should_reobserve"] = bool(final_decision.should_reobserve)
        summary["final_reobserve_reason"] = final_decision.reason
        summary["final_selected_num_views"] = final_snapshot["selected_num_views"]
        summary["final_selected_num_observations"] = final_snapshot["selected_num_observations"]
        summary["closed_loop_reobserve_enabled"] = bool(args.enable_closed_loop_reobserve)
        summary["closed_loop_reobserve_executed"] = bool(closed_loop_executed)
        summary["closed_loop_reobserve_view_ids"] = extra_view_ids
        summary["closed_loop_delta"] = closed_loop_delta
        summary["closed_loop_delta_num_views"] = closed_loop_delta["num_views"]
        summary["closed_loop_delta_num_memory_objects"] = closed_loop_delta["num_memory_objects"]
        summary["closed_loop_delta_num_observations_added"] = closed_loop_delta["num_observations_added"]
        summary["closed_loop_delta_selected_overall_confidence"] = closed_loop_delta[
            "selected_overall_confidence"
        ]
        summary["closed_loop_delta_selected_num_views"] = closed_loop_delta["selected_num_views"]
        summary["closed_loop_delta_selected_num_observations"] = closed_loop_delta[
            "selected_num_observations"
        ]
        summary["closed_loop_selected_object_changed"] = closed_loop_delta["selected_object_changed"]
        summary["closed_loop_reobserve_reason_changed"] = closed_loop_delta["reobserve_reason_changed"]
        summary["closed_loop_reobserve_resolved"] = closed_loop_delta["reobserve_resolved"]
        summary["closed_loop_reobserve_still_needed"] = closed_loop_delta["reobserve_still_needed"]
        summary["closed_loop_before_selected_present_after"] = selected_object_followup["present_after"]
        summary["closed_loop_before_selected_still_selected"] = selected_object_followup["still_selected_after"]
        summary["closed_loop_before_selected_received_observation"] = selected_object_followup[
            "received_observation"
        ]
        summary["closed_loop_before_selected_gained_view_support"] = selected_object_followup[
            "gained_view_support"
        ]
        summary["closed_loop_before_selected_merged_extra_view_ids"] = selected_object_followup[
            "merged_extra_view_ids"
        ]
        summary["closed_loop_before_selected_delta_num_observations"] = selected_object_followup[
            "delta_num_observations"
        ]
        summary["closed_loop_before_selected_delta_num_views"] = selected_object_followup[
            "delta_num_views"
        ]
        summary["closed_loop_extra_view_absorber_object_ids"] = absorber_trace["absorber_object_ids"]
        summary["closed_loop_extra_view_absorber_count"] = absorber_trace["absorber_count"]
        summary["closed_loop_final_selected_absorbed_extra_view"] = absorber_trace[
            "final_selected_absorbed_extra_view"
        ]
        summary["closed_loop_extra_view_third_object_ids"] = absorber_trace["third_object_ids"]
        summary["closed_loop_extra_view_third_object_involved"] = absorber_trace[
            "third_object_involved"
        ]
        summary["closed_loop_selected_object_continuity_enabled"] = bool(
            args.enable_selected_object_continuity
        )
        summary["closed_loop_preferred_merge_count"] = preferred_merge_trace["preferred_merge_count"]
        summary["closed_loop_preferred_merge_rate"] = preferred_merge_trace["preferred_merge_rate"]
        summary["closed_loop_post_selection_continuity_enabled"] = bool(
            args.enable_post_reobserve_selection_continuity
        )
        summary["closed_loop_post_selection_continuity_eligible"] = bool(
            post_selection_continuity["eligible"]
        )
        summary["closed_loop_post_selection_continuity_applied"] = bool(
            post_selection_continuity["applied"]
        )
        summary["closed_loop_post_selection_continuity_reason"] = post_selection_continuity["reason"]
        write_json(summary, run_dir / "summary.json")
        print_summary(summary)
    finally:
        scene.close()


def build_memory_config(args: argparse.Namespace) -> ObjectMemoryConfig:
    """Build object memory config from CLI args."""

    return ObjectMemoryConfig(
        merge_distance=float(args.merge_distance),
        min_points_for_full_geometry_confidence=int(args.min_points_full_confidence),
        max_views_for_full_view_confidence=int(args.max_views_full_confidence),
        fusion_weights=FusionWeights(
            det_score=float(args.fusion_det_weight),
            clip_score=float(args.fusion_clip_weight),
            view_score=float(args.fusion_view_weight),
            consistency_score=float(args.fusion_consistency_weight),
            geometry_score=float(args.fusion_geometry_weight),
        ),
    )


def build_reobserve_config(args: argparse.Namespace) -> ReobservePolicyConfig:
    """Build re-observation policy config from CLI args."""

    return ReobservePolicyConfig(
        min_overall_confidence=float(args.reobserve_min_confidence),
        min_confidence_gap=float(args.reobserve_min_confidence_gap),
        min_views=int(args.reobserve_min_views),
        min_geometry_confidence=float(args.reobserve_min_geometry_confidence),
        min_mean_num_points=float(args.reobserve_min_mean_points),
        default_suggested_view_ids=tuple(str(item) for item in args.reobserve_suggested_view_ids),
        max_suggested_views=int(getattr(args, "reobserve_max_suggested_views", 2)),
    )


def normalize_view_ids(view_ids: Sequence[str] | None, camera_name: str | None) -> list[str | None]:
    """Normalize requested camera views.

    A single ``None`` view means "use the default parser-selected frame".
    """

    cleaned = [view_id.strip() for view_id in view_ids or [] if view_id and view_id.strip()]
    if cleaned:
        return cleaned
    return [camera_name.strip()] if camera_name and camera_name.strip() else [None]


def collect_frames(
    scene: ManiSkillScene,
    view_ids: Sequence[str | None],
    view_preset: str = "none",
    preset_camera_name: str = "base_camera",
) -> list[tuple[str, ObservationFrame]]:
    """Collect frames for requested view ids."""

    if view_preset != "none":
        return collect_preset_frames(scene, view_preset=view_preset, camera_name=preset_camera_name)
    if len(view_ids) == 1 and view_ids[0] is None:
        return [("default", scene.get_observation())]
    frames: list[tuple[str, ObservationFrame]] = []
    for view_id in view_ids:
        if view_id is None:
            frames.append(("default", scene.get_observation()))
        else:
            frames.append((view_id, scene.get_observation(camera_name=view_id)))
    return frames


def collect_preset_frames(
    scene: ManiSkillScene,
    view_preset: str,
    camera_name: str,
) -> list[tuple[str, ObservationFrame]]:
    """Collect virtual multi-view frames by moving one existing ManiSkill camera."""

    preset = VIEW_PRESETS.get(view_preset)
    if preset is None:
        raise ValueError(f"Unknown view preset {view_preset!r}. Available presets: {sorted(VIEW_PRESETS)}")

    frames: list[tuple[str, ObservationFrame]] = []
    for view in preset:
        frame = scene.capture_observation_from_camera_pose(
            camera_name=camera_name,
            eye=view.eye,
            target=view.target,
            up=view.up,
        )
        frames.append((view.label, frame))
    return frames


def lookup_virtual_camera_view(view_id: str) -> VirtualCameraView | None:
    """Return a configured virtual camera pose by label."""

    cleaned = str(view_id or "").strip()
    if cleaned in REOBSERVE_VIEW_POSES:
        return REOBSERVE_VIEW_POSES[cleaned]
    for preset in VIEW_PRESETS.values():
        for view in preset:
            if view.label == cleaned:
                return view
    return None


def collect_reobserve_frames(
    scene: ManiSkillScene,
    suggested_view_ids: Sequence[str],
    camera_name: str,
    max_views: int = 1,
) -> list[tuple[str, ObservationFrame]]:
    """Collect extra virtual frames requested by the policy."""

    frames: list[tuple[str, ObservationFrame]] = []
    for suggested_view_id in suggested_view_ids:
        if len(frames) >= max(0, int(max_views)):
            break
        view = lookup_virtual_camera_view(suggested_view_id)
        if view is None:
            LOGGER.warning("Skipping unsupported re-observation view id %r", suggested_view_id)
            continue
        frame = scene.capture_observation_from_camera_pose(
            camera_name=camera_name,
            eye=view.eye,
            target=view.target,
            up=view.up,
        )
        frames.append((view.label, frame))
    return frames


def process_view(
    args: argparse.Namespace,
    frame: ObservationFrame,
    view_id: str,
    parsed_query: dict[str, Any],
    clip_prompts: list[str],
    memory: ObjectMemory3D,
    run_dir: Path,
    preferred_object_id: str | None = None,
    preferred_merge_distance: float | None = None,
) -> dict[str, Any]:
    """Run detection/ranking/lifting for one view and update object memory."""

    if frame.rgb is None:
        raise RuntimeError(f"View {view_id!r} has no RGB image; cannot run detection.")
    if frame.depth is None:
        raise RuntimeError(f"View {view_id!r} has no depth map; cannot lift candidates.")

    view_dir = run_dir / "views" / _slug(view_id)
    export_observation_frame(frame=frame, output_dir=view_dir / "observation", env_name=args.env_id, step_name=view_id)

    detections = detect_candidates(
        image=frame.rgb,
        text_prompt=parsed_query["normalized_prompt"],
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        top_k=args.top_k,
        save_overlay_path=view_dir / "detection_overlay.png",
        backend=args.detector_backend,
        model_config_path=args.model_config_path,
        model_checkpoint_path=args.model_checkpoint_path,
        model_id=args.detector_model_id,
        device=args.device,
        mock_box_position=args.mock_box_position,
    )
    reranked = rank_view_candidates(
        args=args,
        frame=frame,
        detections=detections,
        clip_prompts=clip_prompts,
        crop_output_dir=view_dir / "candidate_crops",
    )
    candidates_3d, observations_added, observation_assignments = lift_and_add_candidates(
        args=args,
        frame=frame,
        view_id=view_id,
        reranked=reranked,
        memory=memory,
        view_dir=view_dir,
        preferred_object_id=preferred_object_id,
        preferred_merge_distance=preferred_merge_distance,
    )

    result = {
        "view_id": view_id,
        "num_detections": len(detections),
        "num_ranked_candidates": len(reranked),
        "num_3d_candidates": len(candidates_3d),
        "num_observations_added": observations_added,
        "added_object_ids": [assignment["object_id"] for assignment in observation_assignments],
        "observation_assignments": observation_assignments,
        "detections": [candidate.to_json_dict() for candidate in detections],
        "reranked_candidates": [candidate.to_json_dict() for candidate in reranked],
        "candidates_3d": [candidate.to_json_dict() for candidate in candidates_3d],
        "artifacts": str(view_dir),
    }
    write_json(result, view_dir / "view_result.json")
    return result


def rank_view_candidates(
    args: argparse.Namespace,
    frame: ObservationFrame,
    detections: list[DetectionCandidate],
    clip_prompts: list[str],
    crop_output_dir: Path,
) -> list[RankedCandidate]:
    """Rank candidates with either detector-only scores or CLIP."""

    if args.skip_clip:
        return rank_detections_without_clip(detections)
    if frame.rgb is None:
        raise RuntimeError("RGB image is required for CLIP reranking.")
    return rerank_candidates_with_clip(
        image=frame.rgb,
        candidates=detections,
        text_prompt=clip_prompts,
        detector_weight=args.detector_weight,
        clip_weight=args.clip_weight,
        crop_output_dir=crop_output_dir,
        model_name=args.clip_model_name,
        pretrained=args.clip_pretrained,
        device=args.device,
    )


def lift_and_add_candidates(
    args: argparse.Namespace,
    frame: ObservationFrame,
    view_id: str,
    reranked: list[RankedCandidate],
    memory: ObjectMemory3D,
    view_dir: Path,
    preferred_object_id: str | None = None,
    preferred_merge_distance: float | None = None,
) -> tuple[list[Candidate3D], int, list[dict[str, Any]]]:
    """Lift all ranked candidates into 3D and add valid world targets to memory."""

    if frame.rgb is None or frame.depth is None:
        raise RuntimeError("RGB and depth are required for 3D lifting.")

    candidates_3d: list[Candidate3D] = []
    observations_added = 0
    observation_assignments: list[dict[str, Any]] = []
    for index, ranked in enumerate(reranked):
        pointcloud_path = (
            view_dir / "candidate_pointclouds" / f"candidate_{index:03d}.ply"
            if args.save_candidate_pointclouds
            else None
        )
        candidate_3d = lift_box_to_3d(
            rgb=frame.rgb,
            depth=frame.depth,
            box_xyxy=ranked.box_xyxy,
            intrinsic=frame.camera_info.intrinsic,
            extrinsic=frame.camera_info.extrinsic,
            extrinsic_source=frame.camera_info.extrinsic_key,
            segmentation=frame.segmentation,
            segmentation_id=args.segmentation_id,
            use_segmentation=args.use_segmentation,
            output_point_cloud_path=pointcloud_path,
            depth_scale=args.depth_scale,
            fallback_fov_degrees=args.fallback_fov_degrees,
        )
        candidates_3d.append(candidate_3d)
        if candidate_3d.world_xyz is None:
            continue
        matched_object, assignment = memory.add_observation_with_preferred_object(
            ObjectObservation3D(
                world_xyz=candidate_3d.world_xyz,
                label=ranked.phrase,
                det_score=ranked.det_score,
                clip_score=ranked.clip_score,
                fused_2d_score=ranked.fused_2d_score,
                view_id=view_id,
                num_points=candidate_3d.num_points,
                depth_valid_ratio=candidate_3d.depth_valid_ratio,
                point_cloud_path=candidate_3d.point_cloud_path,
                grasp_world_xyz=candidate_3d.grasp_world_xyz,
                grasp_camera_xyz=candidate_3d.grasp_camera_xyz,
                grasp_num_points=candidate_3d.grasp_num_points,
                grasp_metadata=candidate_3d.grasp_metadata,
                metadata={
                    "rank": ranked.rank,
                    "box_xyxy": ranked.box_xyxy,
                    "source": ranked.source,
                },
            ),
            preferred_object_id=preferred_object_id,
            preferred_merge_distance=preferred_merge_distance,
        )
        observations_added += 1
        observation_assignments.append(
            {
                "rank": int(ranked.rank),
                "phrase": ranked.phrase,
                "view_id": view_id,
                "object_id": matched_object.object_id,
                "num_points": int(candidate_3d.num_points),
                "has_grasp_world_xyz": candidate_3d.grasp_world_xyz is not None,
                "grasp_num_points": int(candidate_3d.grasp_num_points),
                **assignment,
            }
        )
    return candidates_3d, observations_added, observation_assignments


def build_memory_state(
    memory: ObjectMemory3D,
    selected: MemoryObject3D | None,
    selection_label: str | None,
    parsed_query: dict[str, Any],
    view_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the detailed memory artifact."""

    return {
        "query": parsed_query,
        "selection_label": selection_label,
        "selected_object_id": None if selected is None else selected.object_id,
        "selected_object": None if selected is None else selected.to_json_dict(),
        "memory": memory.to_json_dict(),
        "views": view_results,
    }


def build_reobserve_stage_snapshot(
    stage: str,
    memory: ObjectMemory3D,
    selected: MemoryObject3D | None,
    selection_label: str | None,
    decision: ReobserveDecision,
    view_ids: Sequence[str],
    total_observations_added: int,
) -> dict[str, Any]:
    """Build a compact before/after closed-loop policy snapshot."""

    return {
        "stage": stage,
        "view_ids": list(view_ids),
        "num_views": len(view_ids),
        "num_memory_objects": len(memory.objects),
        "num_observations_added": int(total_observations_added),
        "selected_object_id": None if selected is None else selected.object_id,
        "selection_label": selection_label,
        "selected_top_label": None if selected is None else selected.top_label,
        "selected_world_xyz": None if selected is None else selected.world_xyz.tolist(),
        "selected_grasp_world_xyz": (
            None if selected is None or selected.grasp_world_xyz is None else selected.grasp_world_xyz.tolist()
        ),
        "selected_overall_confidence": 0.0 if selected is None else float(selected.overall_confidence),
        "selected_semantic_confidence": 0.0 if selected is None else float(selected.semantic_confidence),
        "selected_geometry_confidence": 0.0 if selected is None else float(selected.geometry_confidence),
        "selected_num_views": 0 if selected is None else len(selected.view_ids),
        "selected_view_ids": [] if selected is None else list(selected.view_ids),
        "selected_num_observations": 0 if selected is None else int(selected.num_observations),
        "should_reobserve": bool(decision.should_reobserve),
        "reobserve_reason": decision.reason,
        "suggested_view_ids": list(decision.suggested_view_ids),
        "decision": decision.to_json_dict(),
    }


def build_selected_grasp_diagnostics(selected: MemoryObject3D | None) -> dict[str, Any]:
    """Summarize grasp-point dispersion for the selected memory object."""

    if selected is None:
        return _empty_selected_grasp_diagnostics()

    semantic_xyz = np.asarray(selected.world_xyz, dtype=float).reshape(3)
    grasp_xyz = None if selected.grasp_world_xyz is None else np.asarray(selected.grasp_world_xyz, dtype=float).reshape(3)
    if grasp_xyz is None:
        xy_distance = 0.0
        z_delta = 0.0
    else:
        delta = grasp_xyz - semantic_xyz
        xy_distance = float(np.linalg.norm(delta[:2]))
        z_delta = float(delta[2])

    grasp_observations = _valid_xyz_array(selected.grasp_observation_xyzs)
    if len(grasp_observations) == 0:
        xy_spread = 0.0
        z_spread = 0.0
        max_distance_to_fused = 0.0
    else:
        xy_center = np.median(grasp_observations[:, :2], axis=0)
        xy_spread = float(np.max(np.linalg.norm(grasp_observations[:, :2] - xy_center, axis=1)))
        z_spread = float(np.max(grasp_observations[:, 2]) - np.min(grasp_observations[:, 2]))
        if grasp_xyz is None:
            max_distance_to_fused = 0.0
        else:
            max_distance_to_fused = float(np.max(np.linalg.norm(grasp_observations - grasp_xyz, axis=1)))

    history = selected.metadata.get("grasp_observation_history", [])
    if not isinstance(history, list):
        history = []

    return {
        "selected_semantic_to_grasp_xy_distance": xy_distance,
        "selected_semantic_to_grasp_z_delta": z_delta,
        "selected_grasp_observation_count": int(len(grasp_observations)),
        "selected_grasp_observation_xy_spread": xy_spread,
        "selected_grasp_observation_z_spread": z_spread,
        "selected_grasp_observation_max_distance_to_fused": max_distance_to_fused,
        "selected_grasp_observation_history": history,
    }


def _empty_selected_grasp_diagnostics() -> dict[str, Any]:
    return {
        "selected_semantic_to_grasp_xy_distance": 0.0,
        "selected_semantic_to_grasp_z_delta": 0.0,
        "selected_grasp_observation_count": 0,
        "selected_grasp_observation_xy_spread": 0.0,
        "selected_grasp_observation_z_spread": 0.0,
        "selected_grasp_observation_max_distance_to_fused": 0.0,
        "selected_grasp_observation_history": [],
    }


def _valid_xyz_array(values: Sequence[Any]) -> np.ndarray:
    xyzs: list[np.ndarray] = []
    for value in values:
        xyz = np.asarray(value, dtype=float)
        if xyz.shape == (3,) and np.all(np.isfinite(xyz)):
            xyzs.append(xyz)
    if not xyzs:
        return np.empty((0, 3), dtype=float)
    return np.asarray(xyzs, dtype=float)


def build_closed_loop_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Build compact before/after deltas for closed-loop diagnostics."""

    before_should_reobserve = bool(before.get("should_reobserve"))
    after_should_reobserve = bool(after.get("should_reobserve"))
    return {
        "num_views": _snapshot_int(after, "num_views") - _snapshot_int(before, "num_views"),
        "num_memory_objects": _snapshot_int(after, "num_memory_objects")
        - _snapshot_int(before, "num_memory_objects"),
        "num_observations_added": _snapshot_int(after, "num_observations_added")
        - _snapshot_int(before, "num_observations_added"),
        "selected_overall_confidence": _snapshot_float(after, "selected_overall_confidence")
        - _snapshot_float(before, "selected_overall_confidence"),
        "selected_semantic_confidence": _snapshot_float(after, "selected_semantic_confidence")
        - _snapshot_float(before, "selected_semantic_confidence"),
        "selected_geometry_confidence": _snapshot_float(after, "selected_geometry_confidence")
        - _snapshot_float(before, "selected_geometry_confidence"),
        "selected_num_views": _snapshot_int(after, "selected_num_views")
        - _snapshot_int(before, "selected_num_views"),
        "selected_num_observations": _snapshot_int(after, "selected_num_observations")
        - _snapshot_int(before, "selected_num_observations"),
        "should_reobserve_changed": before_should_reobserve != after_should_reobserve,
        "reobserve_reason_changed": before.get("reobserve_reason") != after.get("reobserve_reason"),
        "selected_object_changed": before.get("selected_object_id") != after.get("selected_object_id"),
        "reobserve_resolved": before_should_reobserve and not after_should_reobserve,
        "reobserve_still_needed": before_should_reobserve and after_should_reobserve,
    }


def build_initial_selected_object_followup(
    before: dict[str, Any],
    after: dict[str, Any] | None,
    memory: ObjectMemory3D,
    extra_view_ids: Sequence[str],
) -> dict[str, Any]:
    """Track what happened to the initially selected object after extra views."""

    before_selected_object_id = str(before.get("selected_object_id") or "").strip() or None
    before_selected_view_ids = [
        str(view_id).strip()
        for view_id in before.get("selected_view_ids", [])
        if str(view_id).strip()
    ]
    before_selected_num_views = _snapshot_int(before, "selected_num_views")
    before_selected_num_observations = _snapshot_int(before, "selected_num_observations")
    after_object = find_memory_object(memory, before_selected_object_id)
    if after_object is None:
        return {
            "before_selected_object_id": before_selected_object_id,
            "present_after": False,
            "still_selected_after": False,
            "delta_num_views": 0,
            "delta_num_observations": 0,
            "received_observation": False,
            "gained_view_support": False,
            "merged_extra_view_ids": [],
        }

    after_selected_object_id = None if after is None else str(after.get("selected_object_id") or "").strip() or None
    after_view_ids = [str(view_id).strip() for view_id in after_object.view_ids if str(view_id).strip()]
    merged_extra_view_ids = [
        str(view_id).strip()
        for view_id in extra_view_ids
        if str(view_id).strip() in after_view_ids and str(view_id).strip() not in before_selected_view_ids
    ]
    delta_num_views = len(after_view_ids) - before_selected_num_views
    delta_num_observations = int(after_object.num_observations) - before_selected_num_observations
    return {
        "before_selected_object_id": before_selected_object_id,
        "present_after": True,
        "still_selected_after": before_selected_object_id == after_selected_object_id,
        "delta_num_views": delta_num_views,
        "delta_num_observations": delta_num_observations,
        "received_observation": delta_num_observations > 0,
        "gained_view_support": delta_num_views > 0,
        "merged_extra_view_ids": merged_extra_view_ids,
    }


def build_post_selection_continuity_trace(
    args: argparse.Namespace,
    initial_snapshot: dict[str, Any],
    selected_object_followup: dict[str, Any],
    base_selected: MemoryObject3D | None,
    base_selection_label: str | None,
    extra_view_absorber_object_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Explain whether post-reobserve selection continuity can be applied."""

    preferred_object_id = str(initial_snapshot.get("selected_object_id") or "").strip() or None
    base_selected_object_id = None if base_selected is None else base_selected.object_id
    absorber_object_ids = dedupe_object_ids(extra_view_absorber_object_ids or [])
    base_selected_absorbed_extra_view = (
        base_selected_object_id is not None
        and base_selected_object_id != preferred_object_id
        and base_selected_object_id in absorber_object_ids
    )
    if not args.enable_post_reobserve_selection_continuity:
        reason = "disabled"
        eligible = False
    elif preferred_object_id is None:
        reason = "no_initial_selected_object"
        eligible = False
    elif not selected_object_followup.get("received_observation"):
        reason = "preferred_object_did_not_receive_extra_view"
        eligible = False
    elif base_selected_absorbed_extra_view:
        reason = "base_selected_absorbed_extra_view"
        eligible = False
    else:
        reason = "eligible"
        eligible = True
    return {
        "enabled": bool(args.enable_post_reobserve_selection_continuity),
        "eligible": bool(eligible),
        "applied": False,
        "reason": reason,
        "preferred_object_id": preferred_object_id,
        "preferred_object_received_observation": bool(selected_object_followup.get("received_observation")),
        "preferred_object_gained_view_support": bool(selected_object_followup.get("gained_view_support")),
        "base_selected_absorbed_extra_view": bool(base_selected_absorbed_extra_view),
        "extra_view_absorber_object_ids": absorber_object_ids,
        "max_confidence_gap": float(args.post_reobserve_selection_margin),
        "selected_object_id_before": base_selected_object_id,
        "selected_selection_label_before": base_selection_label,
        "selected_object_id_after": base_selected_object_id,
        "selected_selection_label_after": base_selection_label,
    }


def collect_extra_view_observation_assignments(extra_view_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return observation assignments from extra re-observation views."""

    return [
        assignment
        for result in extra_view_results
        for assignment in result.get("observation_assignments", [])
        if isinstance(assignment, dict)
    ]


def collect_extra_view_absorber_object_ids(extra_view_results: list[dict[str, Any]]) -> list[str]:
    """Return stable object ids that absorbed extra-view observations."""

    assignments = collect_extra_view_observation_assignments(extra_view_results)
    return dedupe_object_ids(assignment.get("object_id") for assignment in assignments)


def build_closed_loop_absorber_trace(
    before: dict[str, Any],
    after: dict[str, Any],
    extra_view_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize which objects absorbed the extra-view observations."""

    before_selected_object_id = str(before.get("selected_object_id") or "").strip() or None
    after_selected_object_id = str(after.get("selected_object_id") or "").strip() or None
    assignments = collect_extra_view_observation_assignments(extra_view_results)
    absorber_object_ids = dedupe_object_ids(assignment.get("object_id") for assignment in assignments)
    selected_object_ids = {
        object_id for object_id in (before_selected_object_id, after_selected_object_id) if object_id
    }
    third_object_ids = [object_id for object_id in absorber_object_ids if object_id not in selected_object_ids]
    return {
        "initial_selected_object_id": before_selected_object_id,
        "final_selected_object_id": after_selected_object_id,
        "absorber_object_ids": absorber_object_ids,
        "absorber_count": len(absorber_object_ids),
        "initial_selected_absorbed_extra_view": before_selected_object_id in absorber_object_ids
        if before_selected_object_id is not None
        else False,
        "final_selected_absorbed_extra_view": after_selected_object_id in absorber_object_ids
        if after_selected_object_id is not None
        else False,
        "third_object_ids": third_object_ids,
        "third_object_involved": bool(third_object_ids),
        "observation_assignments": assignments,
    }


def build_closed_loop_preferred_merge_trace(extra_view_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether selected-object continuity was actually used."""

    assignments = collect_extra_view_observation_assignments(extra_view_results)
    preferred_merge_count = sum(
        1 for assignment in assignments if bool(assignment.get("used_preferred_object"))
    )
    return {
        "observation_assignment_count": len(assignments),
        "preferred_merge_count": preferred_merge_count,
        "preferred_merge_rate": (
            0.0
            if not assignments
            else float(preferred_merge_count) / float(len(assignments))
        ),
    }


def build_closed_loop_reobserve_report(
    before: dict[str, Any],
    after: dict[str, Any],
    extra_view_results: list[dict[str, Any]],
    selected_object_followup: dict[str, Any],
    absorber_trace: dict[str, Any],
    preferred_merge_trace: dict[str, Any],
    post_selection_continuity: dict[str, Any],
) -> dict[str, Any]:
    """Build the before/after artifact for closed-loop re-observation."""

    return {
        "enabled": True,
        "executed": bool(extra_view_results),
        "extra_views": [
            {
                "view_id": result.get("view_id"),
                "num_detections": int(result.get("num_detections", 0)),
                "num_ranked_candidates": int(result.get("num_ranked_candidates", 0)),
                "num_3d_candidates": int(result.get("num_3d_candidates", 0)),
                "num_observations_added": int(result.get("num_observations_added", 0)),
                "artifacts": result.get("artifacts"),
            }
            for result in extra_view_results
        ],
        "before": before,
        "after": after,
        "delta": build_closed_loop_delta(before, after),
        "initial_selected_object_followup": selected_object_followup,
        "extra_view_absorber_trace": absorber_trace,
        "preferred_merge_trace": preferred_merge_trace,
        "post_selection_continuity": post_selection_continuity,
    }


def write_closed_loop_reobserve_artifacts(
    run_dir: Path,
    memory: ObjectMemory3D,
    parsed_query: dict[str, Any],
    view_results: list[dict[str, Any]],
    initial_snapshot: dict[str, Any],
    final_snapshot: dict[str, Any],
    extra_view_results: list[dict[str, Any]],
    final_selected: MemoryObject3D | None,
    final_selection_label: str | None,
    final_decision: ReobserveDecision,
    post_selection_continuity: dict[str, Any],
) -> None:
    """Write opt-in closed-loop before/after artifacts."""

    write_json(final_decision.to_json_dict(), run_dir / "reobserve_decision_after.json")
    selected_object_followup = build_initial_selected_object_followup(
        before=initial_snapshot,
        after=final_snapshot,
        memory=memory,
        extra_view_ids=[result.get("view_id") for result in extra_view_results if result.get("view_id")],
    )
    absorber_trace = build_closed_loop_absorber_trace(
        before=initial_snapshot,
        after=final_snapshot,
        extra_view_results=extra_view_results,
    )
    preferred_merge_trace = build_closed_loop_preferred_merge_trace(extra_view_results)
    report = build_closed_loop_reobserve_report(
        before=initial_snapshot,
        after=final_snapshot,
        extra_view_results=extra_view_results,
        selected_object_followup=selected_object_followup,
        absorber_trace=absorber_trace,
        preferred_merge_trace=preferred_merge_trace,
        post_selection_continuity=post_selection_continuity,
    )
    report["final_selection_trace"] = build_selection_trace(
        memory=memory,
        selected=final_selected,
        selection_label=final_selection_label,
        parsed_query=parsed_query,
    )
    report["num_total_view_results"] = len(view_results)
    write_json(report, run_dir / "closed_loop_reobserve.json")


def execute_selected_memory_pick(
    scene: ManiSkillScene,
    selected: MemoryObject3D | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Execute the requested pick mode against the final selected memory object."""

    if selected is None:
        return _pick_not_attempted("No selected memory object was available for pick execution.")

    target, target_source = choose_memory_pick_target(selected=selected, grasp_target_mode=args.grasp_target_mode)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        return _pick_not_attempted("Selected memory object had an invalid world-frame target.")

    result = scene.execute_pick(target, executor=args.pick_executor)
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        result["metadata"] = metadata
    metadata.update(
        {
            "target_coordinate_frame": "world",
            "target_used_for_pick": target_source,
            "pick_executor_cli": args.pick_executor,
            "grasp_target_mode": args.grasp_target_mode,
            "grasp_target_mode_effective": target_source,
            "refined_grasp_point_available": selected.grasp_world_xyz is not None,
            "semantic_world_xyz": np.asarray(selected.world_xyz, dtype=float).tolist(),
            "memory_grasp_world_xyz": (
                None
                if selected.grasp_world_xyz is None
                else np.asarray(selected.grasp_world_xyz, dtype=float).tolist()
            ),
            "selected_object_id": selected.object_id,
            "selected_top_label": selected.top_label,
            "selected_overall_confidence": float(selected.overall_confidence),
        }
    )
    return result


def choose_memory_pick_target(
    selected: MemoryObject3D,
    grasp_target_mode: str = "semantic",
) -> tuple[np.ndarray, str]:
    """Choose a multi-view pick target while preserving semantic target reporting."""

    if grasp_target_mode not in {"semantic", "refined"}:
        raise ValueError(f"Unknown grasp target mode: {grasp_target_mode}")
    if grasp_target_mode == "refined" and selected.grasp_world_xyz is not None:
        return np.asarray(selected.grasp_world_xyz, dtype=np.float32).reshape(-1), "memory_grasp_world_xyz"
    return np.asarray(selected.world_xyz, dtype=np.float32).reshape(-1), "selected_object_world_xyz"


def _pick_not_attempted(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "pick_success": False,
        "grasp_attempted": False,
        "task_success": None,
        "is_grasped": None,
        "stage": "not_attempted",
        "target_xyz": [],
        "message": message,
        "trajectory_summary": {"planned_stages": [], "executed_stages": [], "num_env_steps": 0},
        "metadata": {"executor": None},
    }


def build_summary(
    args: argparse.Namespace,
    parsed_query: dict[str, Any],
    view_ids: list[str],
    memory: ObjectMemory3D,
    selected: MemoryObject3D | None,
    selection_label: str | None,
    total_observations_added: int,
    runtime_seconds: float,
    run_dir: Path,
    pick_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact run summary."""

    pick_result = pick_result or _pick_not_attempted("Pick execution was not requested.")
    pick_metadata = pick_result.get("metadata") if isinstance(pick_result.get("metadata"), dict) else {}
    selected_grasp_diagnostics = build_selected_grasp_diagnostics(selected)
    return {
        "query": args.query,
        "normalized_prompt": parsed_query["normalized_prompt"],
        "pick_executor": getattr(args, "pick_executor", "placeholder"),
        "grasp_target_mode": getattr(args, "grasp_target_mode", "semantic"),
        "view_ids": view_ids,
        "num_views": len(view_ids),
        "num_memory_objects": len(memory.objects),
        "num_observations_added": int(total_observations_added),
        "selected_object_id": None if selected is None else selected.object_id,
        "selection_label": selection_label,
        "selected_top_label": None if selected is None else selected.top_label,
        "selected_world_xyz": None if selected is None else selected.world_xyz.tolist(),
        "selected_grasp_world_xyz": (
            None if selected is None or selected.grasp_world_xyz is None else selected.grasp_world_xyz.tolist()
        ),
        **selected_grasp_diagnostics,
        "selected_overall_confidence": 0.0 if selected is None else float(selected.overall_confidence),
        "pick_target_xyz": pick_result.get("target_xyz", []),
        "pick_target_source": pick_metadata.get("target_used_for_pick"),
        "target_used_for_pick": pick_metadata.get("target_used_for_pick"),
        "grasp_attempted": bool(pick_result.get("grasp_attempted", False)),
        "pick_success": bool(pick_result.get("pick_success", pick_result.get("success", False))),
        "task_success": pick_result.get("task_success"),
        "is_grasped": pick_result.get("is_grasped"),
        "pick_stage": pick_result.get("stage"),
        "pick_message": pick_result.get("message"),
        "runtime_seconds": float(runtime_seconds),
        "skip_clip": bool(args.skip_clip),
        "detector_backend": args.detector_backend,
        "view_preset": args.view_preset,
        "camera_name": args.camera_name,
        "artifacts": str(run_dir),
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print a concise terminal summary."""

    print("Multi-view fusion debug complete")
    print(f"  Query:          {summary['query']}")
    print(f"  Prompt:         {summary['normalized_prompt']}")
    print(f"  Views:          {summary['view_ids']}")
    print(f"  Observations:   {summary['num_observations_added']}")
    print(f"  Memory objects: {summary['num_memory_objects']}")
    print(f"  Selected:       {summary['selected_object_id']}")
    print(f"  Confidence:     {summary['selected_overall_confidence']:.3f}")
    print(f"  Reobserve:      {summary.get('should_reobserve')} ({summary.get('reobserve_reason')})")
    print(f"  Pick executor:  {summary.get('pick_executor', 'placeholder')}")
    print(f"  Pick success:   {summary.get('pick_success', False)} ({summary.get('pick_stage')})")
    print(f"  Runtime:        {summary['runtime_seconds']:.3f}s")
    print(f"  Artifacts:      {summary['artifacts']}")


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    return slug[:40] or "view"


def _snapshot_int(snapshot: dict[str, Any], key: str) -> int:
    try:
        return int(snapshot.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _snapshot_float(snapshot: dict[str, Any], key: str) -> float:
    try:
        return float(snapshot.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def dedupe_object_ids(values: Sequence[Any]) -> list[str]:
    """Return stable unique non-empty object ids."""

    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        object_id = str(value or "").strip()
        if object_id and object_id not in seen:
            seen.add(object_id)
            deduped.append(object_id)
    return deduped


def find_memory_object(memory: ObjectMemory3D, object_id: str | None) -> MemoryObject3D | None:
    """Return one memory object by id."""

    if object_id is None:
        return None
    for obj in memory.objects:
        if obj.object_id == object_id:
            return obj
    return None


if __name__ == "__main__":
    main()
