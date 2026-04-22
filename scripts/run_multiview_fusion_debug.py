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
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP and rank by detector score.")
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
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

        selected, selection_label = select_memory_target(memory, parsed_query)
        memory_state = build_memory_state(
            memory=memory,
            selected=selected,
            selection_label=selection_label,
            parsed_query=parsed_query,
            view_results=view_results,
        )
        write_json(memory_state, run_dir / "memory_state.json")

        summary = build_summary(
            args=args,
            parsed_query=parsed_query,
            view_ids=[view_id for view_id, _ in frames],
            memory=memory,
            selected=selected,
            selection_label=selection_label,
            total_observations_added=total_observations_added,
            runtime_seconds=time.perf_counter() - start_time,
            run_dir=run_dir,
        )
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


def process_view(
    args: argparse.Namespace,
    frame: ObservationFrame,
    view_id: str,
    parsed_query: dict[str, Any],
    clip_prompts: list[str],
    memory: ObjectMemory3D,
    run_dir: Path,
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
    candidates_3d, observations_added = lift_and_add_candidates(
        args=args,
        frame=frame,
        view_id=view_id,
        reranked=reranked,
        memory=memory,
        view_dir=view_dir,
    )

    result = {
        "view_id": view_id,
        "num_detections": len(detections),
        "num_ranked_candidates": len(reranked),
        "num_3d_candidates": len(candidates_3d),
        "num_observations_added": observations_added,
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
) -> tuple[list[Candidate3D], int]:
    """Lift all ranked candidates into 3D and add valid world targets to memory."""

    if frame.rgb is None or frame.depth is None:
        raise RuntimeError("RGB and depth are required for 3D lifting.")

    candidates_3d: list[Candidate3D] = []
    observations_added = 0
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
        memory.add_observation(
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
                metadata={
                    "rank": ranked.rank,
                    "box_xyxy": ranked.box_xyxy,
                    "source": ranked.source,
                },
            )
        )
        observations_added += 1
    return candidates_3d, observations_added


def select_memory_target(
    memory: ObjectMemory3D,
    parsed_query: dict[str, Any],
) -> tuple[MemoryObject3D | None, str | None]:
    """Select the best object, trying exact query labels before falling back."""

    candidate_labels = [
        parsed_query.get("normalized_prompt"),
        parsed_query.get("target_name"),
        *list(parsed_query.get("synonyms", [])),
    ]
    for label in _dedupe_strings(candidate_labels):
        selected = memory.select_best(label=label)
        if selected is not None:
            return selected, label
    return memory.select_best(), None


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
) -> dict[str, Any]:
    """Build a compact run summary."""

    return {
        "query": args.query,
        "normalized_prompt": parsed_query["normalized_prompt"],
        "view_ids": view_ids,
        "num_views": len(view_ids),
        "num_memory_objects": len(memory.objects),
        "num_observations_added": int(total_observations_added),
        "selected_object_id": None if selected is None else selected.object_id,
        "selection_label": selection_label,
        "selected_top_label": None if selected is None else selected.top_label,
        "selected_world_xyz": None if selected is None else selected.world_xyz.tolist(),
        "selected_overall_confidence": 0.0 if selected is None else float(selected.overall_confidence),
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
    print(f"  Runtime:        {summary['runtime_seconds']:.3f}s")
    print(f"  Artifacts:      {summary['artifacts']}")


def _dedupe_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            deduped.append(text)
    return deduped


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    return slug[:40] or "view"


if __name__ == "__main__":
    main()
