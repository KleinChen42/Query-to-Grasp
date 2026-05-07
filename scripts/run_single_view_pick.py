"""Run query -> single-view target localization -> placeholder pick."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_single_view_query import build_clip_prompts, rank_detections_without_clip  # noqa: E402
from src.env.maniskill_env import ManiSkillScene  # noqa: E402
from src.io.execution_video import ExecutionVideoRecorder  # noqa: E402
from src.io.export_utils import export_observation_frame, write_json  # noqa: E402
from src.manipulation.oracle_targets import find_stackcube_oracle_place_xyz  # noqa: E402
from src.manipulation.place_targets import PredictedPlaceTarget, select_candidate_place_target  # noqa: E402
from src.perception.clip_rerank import rerank_candidates_with_clip  # noqa: E402
from src.perception.grounding_dino import detect_candidates  # noqa: E402
from src.perception.mask_projector import Candidate3D, lift_box_to_3d  # noqa: E402
from src.perception.query_parser import parse_query  # noqa: E402

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-view query-to-placeholder-pick baseline.")
    parser.add_argument("--query", default="red cube", help="Natural language target query.")
    parser.add_argument("--env-id", default="PickCube-v1", help="ManiSkill environment id.")
    parser.add_argument("--obs-mode", default="rgbd", help="ManiSkill observation mode.")
    parser.add_argument("--control-mode", default=None, help="Optional ManiSkill control mode.")
    parser.add_argument(
        "--pick-executor",
        default="placeholder",
        choices=["placeholder", "sim_topdown", "sim_pick_place"],
        help="Execution backend. sim_topdown picks; sim_pick_place uses query pick target plus a place target.",
    )
    parser.add_argument(
        "--grasp-target-mode",
        default="semantic",
        choices=["semantic", "refined"],
        help="Target point used for pick execution. refined uses an opt-in workspace-filtered point when available.",
    )
    parser.add_argument(
        "--place-target-source",
        default="none",
        choices=["none", "oracle_cubeB_pose", "predicted_place_object"],
        help="Optional place target source for sim_pick_place.",
    )
    parser.add_argument(
        "--place-query",
        default="cube",
        help="Reference-object query used when --place-target-source predicted_place_object.",
    )
    parser.add_argument(
        "--place-min-distance-from-pick",
        type=float,
        default=0.05,
        help="Minimum XY distance between pick target and predicted place object.",
    )
    parser.add_argument(
        "--place-target-z",
        type=float,
        default=0.02,
        help="World-frame Z used for predicted StackCube place object centers.",
    )
    parser.add_argument("--camera-name", default=None, help="Optional camera key to prefer.")
    parser.add_argument("--sensor-width", type=int, default=None, help="Optional ManiSkill RGB-D sensor width.")
    parser.add_argument("--sensor-height", type=int, default=None, help="Optional ManiSkill RGB-D sensor height.")
    parser.add_argument("--seed", type=int, default=0, help="Environment reset seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "single_view_pick")
    parser.add_argument("--prefer-llm-parser", action="store_true", help="Try LLM parsing before deterministic rules.")
    parser.add_argument("--box-threshold", type=float, default=0.35, help="GroundingDINO box threshold.")
    parser.add_argument("--text-threshold", type=float, default=0.25, help="GroundingDINO text threshold.")
    parser.add_argument("--top-k", type=int, default=10, help="Maximum number of detector candidates.")
    parser.add_argument(
        "--detector-backend",
        default="auto",
        choices=["auto", "hf", "transformers", "groundingdino", "original", "mock"],
    )
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
    parser.add_argument("--capture-execution-video", action="store_true", help="Opt-in demo capture of continuous execution RGB frames.")
    parser.add_argument("--execution-video-fps", type=float, default=24.0, help="FPS for opt-in execution demo videos.")
    parser.add_argument("--execution-video-camera-name", default="base_camera", help="Camera used for execution video extraction.")
    parser.add_argument("--execution-video-every-n-steps", type=int, default=1, help="Frame sampling interval for execution video capture.")
    parser.add_argument("--execution-video-width", type=int, default=None, help="Optional output width for captured execution video frames.")
    parser.add_argument("--execution-video-height", type=int, default=None, help="Optional output height for captured execution video frames.")
    parser.add_argument("--oracle-pick-noise-std", type=float, default=0.0, help="Gaussian noise std (meters) added to the pick target for sensitivity analysis.")
    parser.add_argument("--oracle-place-noise-std", type=float, default=0.0, help="Gaussian noise std (meters) added to the oracle place target for sensitivity analysis.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    pipeline_start_time = time.perf_counter()
    if args.pick_executor in {"sim_topdown", "sim_pick_place"} and args.control_mode is None:
        args.control_mode = "pd_ee_delta_pos"
        LOGGER.info("Using control_mode=pd_ee_delta_pos for simulated executor.")

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
        **build_sensor_kwargs(args),
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
            candidate_3d = None
            pick_result = _pick_not_attempted("No valid 2D candidates remained after ranking.")
            write_json({"top_candidate": None, "reason": "no_valid_candidates"}, run_dir / "top_candidate_3d.json")
        else:
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
            target_xyz, coordinate_frame, target_source = choose_pick_target(
                candidate_3d,
                grasp_target_mode=args.grasp_target_mode,
            )
            noise_std = float(getattr(args, "oracle_pick_noise_std", 0.0))
            if target_xyz is not None and noise_std > 0.0:
                rng = np.random.default_rng(seed=args.seed)
                noise = rng.normal(0.0, noise_std, size=3).astype(np.float32)
                target_xyz_before_noise = target_xyz.copy()
                target_xyz = target_xyz + noise
                LOGGER.info(
                    "Applied oracle noise std=%.4f: before=%s after=%s noise=%s",
                    noise_std, target_xyz_before_noise.tolist(), target_xyz.tolist(), noise.tolist(),
                )
            if target_xyz is None:
                pick_result = _pick_not_attempted("Top candidate had no valid 3D target point.")
            elif args.pick_executor in {"sim_topdown", "sim_pick_place"} and coordinate_frame != "world":
                pick_result = _pick_not_attempted("Simulated execution requires a world-frame target.")
            else:
                predicted_place_target = None
                if args.pick_executor == "sim_pick_place" and args.place_target_source == "predicted_place_object":
                    predicted_place_target = build_single_view_predicted_place_target(
                        args=args,
                        frame=frame,
                        pick_xyz=target_xyz,
                        run_dir=run_dir,
                    )
                pick_result = execute_single_view_target(
                    scene=scene,
                    target_xyz=target_xyz,
                    args=args,
                    run_dir=run_dir,
                    target_source=target_source,
                    predicted_place_target=predicted_place_target,
                )
                pick_result.setdefault("metadata", {})
                pick_result["metadata"]["target_coordinate_frame"] = coordinate_frame
                pick_result["metadata"]["target_used_for_pick"] = target_source
                pick_result["metadata"]["pick_executor_cli"] = args.pick_executor
                pick_result["metadata"]["grasp_target_mode"] = args.grasp_target_mode
                pick_result["metadata"]["oracle_pick_noise_std"] = float(getattr(args, "oracle_pick_noise_std", 0.0))
                if noise_std > 0.0:
                    pick_result["metadata"]["target_xyz_before_noise"] = _array_to_list(target_xyz_before_noise)
                    pick_result["metadata"]["applied_noise"] = noise.tolist()
                pick_result["metadata"]["semantic_world_xyz"] = _array_to_list(candidate_3d.world_xyz)
                pick_result["metadata"]["grasp_world_xyz"] = _array_to_list(candidate_3d.grasp_world_xyz)
                pick_result["metadata"]["grasp_metadata"] = candidate_3d.grasp_metadata

        write_json(pick_result, run_dir / "pick_result.json")

        summary = build_summary(
            args=args,
            parsed_query=parsed_query,
            num_detections=len(detections),
            num_ranked=len(reranked),
            candidate_3d=candidate_3d,
            pick_result=pick_result,
            run_dir=run_dir,
            runtime_seconds=time.perf_counter() - pipeline_start_time,
            detector_top_phrase=_top_phrase(detections),
            final_top_phrase=_top_phrase(reranked),
            top1_changed_by_rerank=_top1_changed_by_rerank(detections, reranked, skip_clip=args.skip_clip),
        )
        write_json(summary, run_dir / "summary.json")
        print_pick_summary(summary)
    finally:
        scene.close()


def choose_pick_target(
    candidate_3d: Candidate3D,
    grasp_target_mode: str = "semantic",
) -> tuple[np.ndarray | None, str | None, str | None]:
    """Choose the point used for pick execution without changing semantic 3D reporting."""

    if grasp_target_mode not in {"semantic", "refined"}:
        raise ValueError(f"Unknown grasp target mode: {grasp_target_mode}")
    if grasp_target_mode == "refined" and candidate_3d.grasp_world_xyz is not None:
        return np.asarray(candidate_3d.grasp_world_xyz, dtype=np.float32), "world", "grasp_world_xyz"
    if candidate_3d.world_xyz is not None:
        return np.asarray(candidate_3d.world_xyz, dtype=np.float32), "world", "world_xyz"
    if candidate_3d.camera_xyz is not None:
        return np.asarray(candidate_3d.camera_xyz, dtype=np.float32), "camera", "camera_xyz"
    return None, None, None


def build_sensor_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build optional ManiSkill sensor config kwargs for presentation captures."""

    width = getattr(args, "sensor_width", None)
    height = getattr(args, "sensor_height", None)
    if width is None and height is None:
        return {}
    if width is None or height is None:
        raise ValueError("--sensor-width and --sensor-height must be provided together.")
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("--sensor-width and --sensor-height must be positive.")
    return {"sensor_configs": {"width": int(width), "height": int(height)}}


def execute_single_view_target(
    scene: ManiSkillScene,
    target_xyz: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path | None = None,
    target_source: str | None = None,
    predicted_place_target: PredictedPlaceTarget | None = None,
) -> dict[str, Any]:
    """Execute the selected single-view target with the requested executor."""

    recorder = make_execution_recorder(
        scene=scene,
        args=args,
        run_dir=run_dir,
        metadata={
            "runner": "single_view",
            "query": args.query,
            "seed": args.seed,
            "env_id": args.env_id,
            "pick_executor": args.pick_executor,
            "target_source": target_source,
            "place_target_source": args.place_target_source,
            "place_query": getattr(args, "place_query", None),
        },
    )
    step_callback = None if recorder is None else recorder.record_step
    if args.pick_executor == "sim_pick_place":
        if args.place_target_source == "oracle_cubeB_pose":
            place_xyz, place_metadata = find_stackcube_oracle_place_xyz(scene.env)
            place_noise_std = float(getattr(args, "oracle_place_noise_std", 0.0))
            if place_noise_std > 0.0:
                place_rng = np.random.default_rng(seed=args.seed + 10000)
                place_noise = place_rng.normal(0.0, place_noise_std, size=3).astype(np.float32)
                place_xyz_before_noise = place_xyz.copy()
                place_xyz = place_xyz + place_noise
                place_metadata["oracle_place_noise_std"] = place_noise_std
                place_metadata["place_xyz_before_noise"] = place_xyz_before_noise.tolist()
                place_metadata["applied_place_noise"] = place_noise.tolist()
                LOGGER.info(
                    "Applied oracle place noise std=%.4f: before=%s after=%s",
                    place_noise_std, place_xyz_before_noise.tolist(), place_xyz.tolist(),
                )
        elif args.place_target_source == "predicted_place_object":
            if predicted_place_target is None:
                return _pick_not_attempted(
                    "sim_pick_place requires a valid predicted place object when "
                    "--place-target-source predicted_place_object is used."
                )
            place_xyz = predicted_place_target.place_xyz
            place_metadata = dict(predicted_place_target.metadata)
        else:
            return _pick_not_attempted(
                "sim_pick_place requires --place-target-source oracle_cubeB_pose or predicted_place_object."
            )
        if step_callback is None:
            result = scene.execute_pick_place(pick_xyz=target_xyz, place_xyz=place_xyz)
        else:
            result = scene.execute_pick_place(pick_xyz=target_xyz, place_xyz=place_xyz, step_callback=step_callback)
        result.setdefault("metadata", {})
        result["metadata"].update(
            {
                "place_target_source": args.place_target_source,
                "place_query": getattr(args, "place_query", None),
                "place_target_xyz": _array_to_list(place_xyz),
                "place_target_metadata": place_metadata,
                "place_selection_reason": place_metadata.get("selection_reason"),
                "place_pick_xy_distance": place_metadata.get("place_pick_xy_distance"),
            }
        )
        finalize_execution_recorder(recorder, result)
        return result
    if step_callback is None:
        result = scene.execute_pick(target_xyz, executor=args.pick_executor)
    else:
        result = scene.execute_pick(target_xyz, executor=args.pick_executor, step_callback=step_callback)
    finalize_execution_recorder(recorder, result)
    return result


def build_single_view_predicted_place_target(
    args: argparse.Namespace,
    frame: Any,
    pick_xyz: np.ndarray,
    run_dir: Path,
) -> PredictedPlaceTarget | None:
    """Predict a StackCube reference object place target from the same RGB-D view."""

    place_query = str(getattr(args, "place_query", "cube") or "cube")
    parsed_place_query = parse_query(place_query, prefer_llm=args.prefer_llm_parser)
    place_prompts = build_clip_prompts(parsed_place_query)
    place_dir = run_dir / "place_target_prediction"
    place_dir.mkdir(parents=True, exist_ok=True)
    write_json(parsed_place_query, place_dir / "parsed_place_query.json")

    detections = detect_candidates(
        image=frame.rgb,
        text_prompt=parsed_place_query["normalized_prompt"],
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        top_k=args.top_k,
        save_overlay_path=place_dir / "detection_overlay.png",
        backend=args.detector_backend,
        model_config_path=args.model_config_path,
        model_checkpoint_path=args.model_checkpoint_path,
        model_id=args.detector_model_id,
        device=args.device,
        mock_box_position=args.mock_box_position,
    )
    write_json(
        {
            "query": parsed_place_query,
            "candidates": [candidate.to_json_dict() for candidate in detections],
        },
        place_dir / "detections.json",
    )

    if args.skip_clip:
        reranked = rank_detections_without_clip(detections)
    else:
        reranked = rerank_candidates_with_clip(
            image=frame.rgb,
            candidates=detections,
            text_prompt=place_prompts,
            detector_weight=args.detector_weight,
            clip_weight=args.clip_weight,
            crop_output_dir=place_dir / "candidate_crops",
            model_name=args.clip_model_name,
            pretrained=args.clip_pretrained,
            device=args.device,
        )
    write_json(
        {
            "clip_prompts": place_prompts,
            "candidates": [candidate.to_json_dict() for candidate in reranked],
        },
        place_dir / "reranked_candidates.json",
    )

    candidates_3d: list[Candidate3D] = []
    for index, ranked in enumerate(reranked):
        pointcloud_path = (
            place_dir / "candidate_pointclouds" / f"candidate_{index:03d}.ply"
            if args.save_candidate_pointcloud
            else None
        )
        candidates_3d.append(
            lift_box_to_3d(
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
        )
    write_json(
        {
            "candidates_3d": [candidate.to_json_dict() for candidate in candidates_3d],
        },
        place_dir / "candidates_3d.json",
    )

    selected = select_candidate_place_target(
        candidates=candidates_3d,
        pick_xyz=pick_xyz,
        min_xy_distance=float(getattr(args, "place_min_distance_from_pick", 0.05)),
        place_query=place_query,
        place_target_z=float(getattr(args, "place_target_z", 0.02)),
    )
    if selected is not None:
        selected_index = selected.metadata.get("selected_candidate_index")
        if isinstance(selected_index, int) and 0 <= selected_index < len(reranked):
            selected.metadata["selected_phrase"] = reranked[selected_index].phrase
            selected.metadata["selected_det_score"] = float(reranked[selected_index].det_score)
            selected.metadata["selected_clip_score"] = float(reranked[selected_index].clip_score)
            selected.metadata["selected_fused_2d_score"] = float(reranked[selected_index].fused_2d_score)
    write_json(
        {
            "place_target_source": "predicted_place_object",
            "place_query": place_query,
            "min_xy_distance_from_pick": float(getattr(args, "place_min_distance_from_pick", 0.05)),
            "selected": None
            if selected is None
            else {
                "place_xyz": selected.place_xyz.astype(float).tolist(),
                "source": selected.source,
                "metadata": selected.metadata,
            },
        },
        place_dir / "place_target_prediction.json",
    )
    return selected


def make_execution_recorder(
    scene: ManiSkillScene,
    args: argparse.Namespace,
    run_dir: Path | None,
    metadata: dict[str, Any],
) -> ExecutionVideoRecorder | None:
    """Create an opt-in execution recorder for demo runs."""

    if not getattr(args, "capture_execution_video", False):
        return None
    if args.pick_executor not in {"sim_topdown", "sim_pick_place"}:
        return None
    if run_dir is None:
        return None
    return ExecutionVideoRecorder(
        output_dir=run_dir / "execution_video",
        fps=float(args.execution_video_fps),
        camera_name=args.execution_video_camera_name,
        every_n_steps=int(args.execution_video_every_n_steps),
        output_width=args.execution_video_width,
        output_height=args.execution_video_height,
        fallback_observation_fn=scene.capture_sensor_observation,
        metadata=metadata,
    )


def finalize_execution_recorder(recorder: ExecutionVideoRecorder | None, result: dict[str, Any]) -> None:
    if recorder is None:
        return
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        result["metadata"] = metadata
    metadata["execution_video"] = recorder.finalize()


def build_summary(
    args: argparse.Namespace,
    parsed_query: dict[str, Any],
    num_detections: int,
    num_ranked: int,
    candidate_3d: Candidate3D | None,
    pick_result: dict[str, Any],
    run_dir: Path,
    runtime_seconds: float,
    detector_top_phrase: str | None,
    final_top_phrase: str | None,
    top1_changed_by_rerank: bool,
) -> dict[str, Any]:
    """Build a concise run summary."""

    metadata = pick_result.get("metadata") if isinstance(pick_result.get("metadata"), dict) else {}
    return {
        "query": args.query,
        "normalized_prompt": parsed_query["normalized_prompt"],
        "grasp_target_mode": getattr(args, "grasp_target_mode", "semantic"),
        "place_target_source": metadata.get("place_target_source", getattr(args, "place_target_source", "none")),
        "place_query": metadata.get("place_query", getattr(args, "place_query", None)),
        "raw_num_detections": num_detections,
        "num_detections": num_detections,
        "num_ranked_candidates": num_ranked,
        "top1_changed_by_rerank": bool(top1_changed_by_rerank),
        "detector_top_phrase": detector_top_phrase,
        "final_top_phrase": final_top_phrase,
        "camera_xyz": None if candidate_3d is None or candidate_3d.camera_xyz is None else candidate_3d.camera_xyz.tolist(),
        "world_xyz": None if candidate_3d is None or candidate_3d.world_xyz is None else candidate_3d.world_xyz.tolist(),
        "semantic_world_xyz": (
            None if candidate_3d is None or candidate_3d.world_xyz is None else candidate_3d.world_xyz.tolist()
        ),
        "grasp_world_xyz": (
            None if candidate_3d is None or candidate_3d.grasp_world_xyz is None else candidate_3d.grasp_world_xyz.tolist()
        ),
        "target_used_for_pick": metadata.get("target_used_for_pick"),
        "grasp_metadata": {} if candidate_3d is None else candidate_3d.grasp_metadata,
        "num_3d_points": 0 if candidate_3d is None else candidate_3d.num_points,
        "pick_success": bool(pick_result.get("pick_success", pick_result.get("success", False))),
        "grasp_attempted": bool(pick_result.get("grasp_attempted", False)),
        "place_attempted": bool(pick_result.get("place_attempted", False)),
        "place_success": pick_result.get("place_success"),
        "place_target_xyz": pick_result.get("place_xyz", metadata.get("place_target_xyz")),
        "place_selection_reason": metadata.get("place_selection_reason"),
        "place_pick_xy_distance": metadata.get("place_pick_xy_distance"),
        "task_success": pick_result.get("task_success"),
        "is_grasped": pick_result.get("is_grasped"),
        "pick_stage": pick_result.get("stage"),
        "place_message": pick_result.get("message") if pick_result.get("place_attempted") else None,
        "pick_message": pick_result.get("message"),
        "execution_video": metadata.get("execution_video"),
        "runtime_seconds": float(runtime_seconds),
        "artifacts": str(run_dir),
    }


def print_pick_summary(summary: dict[str, Any]) -> None:
    """Print a concise terminal summary."""

    print("Single-view pick pipeline complete")
    print(f"  Query:        {summary['query']}")
    print(f"  Prompt:       {summary['normalized_prompt']}")
    print(f"  Detections:   {summary['num_detections']}")
    print(f"  Ranked:       {summary['num_ranked_candidates']}")
    print(f"  Rerank top-1: {summary['top1_changed_by_rerank']}")
    print(f"  Camera XYZ:   {summary['camera_xyz']}")
    print(f"  World XYZ:    {summary['world_xyz']}")
    print(f"  Grasp mode:   {summary.get('grasp_target_mode', 'semantic')}")
    print(f"  Grasp XYZ:    {summary.get('grasp_world_xyz')}")
    print(f"  Pick success: {summary['pick_success']}")
    print(f"  Place success:{summary.get('place_success')}")
    print(f"  Attempted:    {summary.get('grasp_attempted', False)}")
    print(f"  Task success: {summary.get('task_success')}")
    print(f"  Pick stage:   {summary['pick_stage']}")
    print(f"  Runtime:      {summary.get('runtime_seconds', 0.0):.3f}s")
    print(f"  Artifacts:    {summary['artifacts']}")


def _pick_not_attempted(message: str) -> dict[str, Any]:
    return {
        "success": False,
        "pick_success": False,
        "grasp_attempted": False,
        "place_attempted": False,
        "place_success": False,
        "task_success": None,
        "is_grasped": None,
        "stage": "not_attempted",
        "target_xyz": [],
        "place_xyz": [],
        "message": message,
        "trajectory_summary": {"planned_stages": [], "executed_stages": [], "num_env_steps": 0},
        "metadata": {"executor": None},
    }


def _array_to_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float).tolist()


def _top_phrase(candidates: list[Any]) -> str | None:
    if not candidates:
        return None
    return str(candidates[0].phrase)


def _top1_changed_by_rerank(detections: list[Any], reranked: list[Any], skip_clip: bool) -> bool:
    if skip_clip or not detections or not reranked:
        return False
    detector_top = detections[0]
    final_top = reranked[0]
    if str(detector_top.phrase) != str(final_top.phrase):
        return True
    return not np.allclose(
        np.asarray(detector_top.box_xyxy, dtype=float),
        np.asarray(final_top.box_xyxy, dtype=float),
    )


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    return slug[:40] or "query"


if __name__ == "__main__":
    main()
