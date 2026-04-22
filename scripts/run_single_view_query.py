"""Run the Phase 2A single-view semantic retrieval baseline."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.maniskill_env import ManiSkillScene  # noqa: E402
from src.io.export_utils import export_observation_frame, write_json  # noqa: E402
from src.perception.clip_rerank import RankedCandidate, rerank_candidates_with_clip  # noqa: E402
from src.perception.grounding_dino import DetectionCandidate, detect_candidates  # noqa: E402
from src.perception.mask_projector import lift_box_to_3d  # noqa: E402
from src.perception.query_parser import parse_query  # noqa: E402

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-view query-to-3D-target baseline.")
    parser.add_argument("--query", default="red cube", help="Natural language target query.")
    parser.add_argument("--env-id", default="PickCube-v1", help="ManiSkill environment id.")
    parser.add_argument("--obs-mode", default="rgbd", help="ManiSkill observation mode.")
    parser.add_argument("--control-mode", default=None, help="Optional ManiSkill control mode.")
    parser.add_argument("--camera-name", default=None, help="Optional camera key to prefer.")
    parser.add_argument("--seed", type=int, default=0, help="Environment reset seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "single_view_query")
    parser.add_argument("--prefer-llm-parser", action="store_true", help="Try LLM parsing before deterministic rules.")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="GroundingDINO box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold.")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum number of detector candidates.")
    parser.add_argument("--detector-backend", default="auto", choices=["auto", "hf", "transformers", "groundingdino", "original", "mock"])
    parser.add_argument("--mock-box-position", default="center", choices=["center", "left", "right", "all"])
    parser.add_argument("--detector-model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--model-config-path", type=Path, default=None, help="Original GroundingDINO config path.")
    parser.add_argument("--model-checkpoint-path", type=Path, default=None, help="Original GroundingDINO checkpoint path.")
    parser.add_argument("--clip-model-name", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP reranking and use detector scores directly.")
    parser.add_argument("--device", default=None, help="Torch device, for example cuda or cpu.")
    parser.add_argument("--detector-weight", type=float, default=0.5)
    parser.add_argument("--clip-weight", type=float, default=0.5)
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--fallback-fov-degrees", type=float, default=60.0)
    parser.add_argument("--use-segmentation", action="store_true", help="Restrict 3D lifting to a dominant segmentation id.")
    parser.add_argument("--segmentation-id", type=int, default=None, help="Specific segmentation id to use for 3D lifting.")
    parser.add_argument("--save-candidate-pointcloud", action="store_true", help="Save local point cloud for the top candidate.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{timestamp}_{_slug(args.query)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    parsed_query = parse_query(args.query, prefer_llm=args.prefer_llm_parser)
    write_json(parsed_query, run_dir / "parsed_query.json")
    clip_prompts = build_clip_prompts(parsed_query)

    scene = ManiSkillScene(
        env_name=args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        camera_name=args.camera_name,
    )
    try:
        scene.reset(seed=args.seed)
        frame = scene.get_observation(camera_name=args.camera_name)
        if frame.rgb is None:
            raise RuntimeError("The observation parser did not find an RGB image; cannot run detection.")
        if frame.depth is None:
            raise RuntimeError("The observation parser did not find a depth map; cannot lift detections to 3D.")

        export_observation_frame(frame=frame, output_dir=run_dir / "observation", env_name=args.env_id, step_name="reset")

        detections = detect_candidates(
            image=frame.rgb,
            text_prompt=parsed_query["normalized_prompt"],
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            top_k=args.top_k,
            save_overlay_path=run_dir / "detection_overlay.png",
            backend=args.detector_backend,
            model_config_path=args.model_config_path,
            model_checkpoint_path=args.model_checkpoint_path,
            model_id=args.detector_model_id,
            device=args.device,
            mock_box_position=args.mock_box_position,
        )
        write_json(
            {
                "query": parsed_query,
                "parameters": {
                    "box_threshold": args.box_threshold,
                    "text_threshold": args.text_threshold,
                    "top_k": args.top_k,
                    "detector_backend": args.detector_backend,
                    "mock_box_position": args.mock_box_position if args.detector_backend == "mock" else None,
                },
                "candidates": [candidate.to_json_dict() for candidate in detections],
            },
            run_dir / "detections.json",
        )

        if not detections:
            write_json({"top_candidate": None, "reason": "no_detections"}, run_dir / "top_candidate_3d.json")
            write_json({"candidates": []}, run_dir / "reranked_candidates.json")
            print(f"No detections found for query {args.query!r}. Artifacts: {run_dir}")
            return

        if args.skip_clip:
            reranked = rank_detections_without_clip(detections)
        else:
            reranked = rerank_candidates_with_clip(
                image=frame.rgb,
                candidates=detections,
                text_prompt=clip_prompts,
                detector_weight=args.detector_weight,
                clip_weight=args.clip_weight,
                crop_output_dir=run_dir / "candidate_crops",
                model_name=args.clip_model_name,
                pretrained=args.clip_pretrained,
                device=args.device,
            )
        write_json(
            {
                "clip_prompts": clip_prompts,
                "parameters": {
                    "skip_clip": args.skip_clip,
                    "detector_weight": args.detector_weight,
                    "clip_weight": args.clip_weight,
                    "clip_model_name": args.clip_model_name,
                    "clip_pretrained": args.clip_pretrained,
                },
                "candidates": [candidate.to_json_dict() for candidate in reranked],
            },
            run_dir / "reranked_candidates.json",
        )

        if not reranked:
            write_json({"top_candidate": None, "reason": "no_valid_crops"}, run_dir / "top_candidate_3d.json")
            print(f"No valid crops remained for query {args.query!r}. Artifacts: {run_dir}")
            return

        top_candidate = reranked[0]
        pointcloud_path = run_dir / "top_candidate_pointcloud.ply" if args.save_candidate_pointcloud else None
        candidate_3d = lift_box_to_3d(
            rgb=frame.rgb,
            depth=frame.depth,
            box_xyxy=top_candidate.box_xyxy,
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
        write_json(
            {
                "query": parsed_query,
                "top_2d_candidate": top_candidate.to_json_dict(),
                "top_candidate_3d": candidate_3d.to_json_dict(),
            },
            run_dir / "top_candidate_3d.json",
        )

        summary = {
            "query": args.query,
            "normalized_prompt": parsed_query["normalized_prompt"],
            "num_detections": len(detections),
            "top_phrase": top_candidate.phrase,
            "top_fused_2d_score": top_candidate.fused_2d_score,
            "camera_xyz": None if candidate_3d.camera_xyz is None else candidate_3d.camera_xyz.tolist(),
            "world_xyz": None if candidate_3d.world_xyz is None else candidate_3d.world_xyz.tolist(),
            "num_3d_points": candidate_3d.num_points,
            "artifacts": str(run_dir),
        }
        write_json(summary, run_dir / "summary.json")
        print_single_view_summary(summary)
    finally:
        scene.close()


def build_clip_prompts(parsed_query: dict[str, Any]) -> list[str]:
    """Build a small prompt set from parser output for robust CLIP scoring."""

    attributes = [str(item) for item in parsed_query.get("attributes", [])]
    synonyms = [str(item) for item in parsed_query.get("synonyms", [])]
    normalized_prompt = str(parsed_query.get("normalized_prompt", "")).strip()
    prompts = [normalized_prompt] if normalized_prompt else []
    for synonym in synonyms:
        prompt = " ".join([*attributes, synonym]).strip()
        if prompt:
            prompts.append(prompt)
    return _dedupe(prompts)


def rank_detections_without_clip(candidates: list[DetectionCandidate]) -> list[RankedCandidate]:
    """Convert detector candidates into ranked candidates without CLIP scoring."""

    ranked = [
        RankedCandidate(
            box_xyxy=candidate.box_xyxy,
            det_score=float(candidate.det_score),
            clip_score=0.0,
            fused_2d_score=float(candidate.det_score),
            phrase=candidate.phrase,
            image_crop_path=candidate.image_crop_path,
            source="detector_only",
            metadata={
                "skip_clip": True,
                "detector_source": candidate.source,
                "detector_metadata": candidate.metadata,
            },
        )
        for candidate in candidates
    ]
    ranked.sort(key=lambda item: (-item.fused_2d_score, item.phrase))
    for rank, candidate in enumerate(ranked):
        candidate.rank = rank
    return ranked


def print_single_view_summary(summary: dict[str, Any]) -> None:
    """Print a concise terminal summary."""

    print("Single-view query complete")
    print(f"  Query:        {summary['query']}")
    print(f"  Prompt:       {summary['normalized_prompt']}")
    print(f"  Detections:   {summary['num_detections']}")
    print(f"  Top phrase:   {summary['top_phrase']}")
    print(f"  2D score:     {summary['top_fused_2d_score']:.3f}")
    print(f"  Camera XYZ:   {summary['camera_xyz']}")
    print(f"  World XYZ:    {summary['world_xyz']}")
    print(f"  3D points:    {summary['num_3d_points']}")
    print(f"  Artifacts:    {summary['artifacts']}")


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    return slug[:40] or "query"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


if __name__ == "__main__":
    main()
