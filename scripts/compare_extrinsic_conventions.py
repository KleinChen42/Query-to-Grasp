"""Compare ManiSkill camera extrinsic conventions for RGB-D lifting."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_multiview_fusion_debug import VIEW_PRESETS  # noqa: E402
from scripts.run_single_view_query import build_clip_prompts, rank_detections_without_clip  # noqa: E402
from src.env.camera_utils import (  # noqa: E402
    extract_observation_frame,
    extract_observation_matrix_by_leaf,
)
from src.env.maniskill_env import ManiSkillScene  # noqa: E402
from src.io.export_utils import write_json  # noqa: E402
from src.perception.clip_rerank import RankedCandidate, rerank_candidates_with_clip  # noqa: E402
from src.perception.grounding_dino import detect_candidates  # noqa: E402
from src.perception.mask_projector import lift_box_to_3d  # noqa: E402
from src.perception.query_parser import parse_query  # noqa: E402


CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


@dataclass(frozen=True)
class ExtrinsicConvention:
    """One candidate matrix convention for lifting OpenCV-style camera points."""

    name: str
    matrix: np.ndarray
    source_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare extrinsic conventions on identical multi-view detections.")
    parser.add_argument("--queries", nargs="*", default=["red cube"], help="Queries to evaluate.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0], help="Environment reset seeds.")
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", default="rgbd")
    parser.add_argument("--control-mode", default=None)
    parser.add_argument("--camera-name", default="base_camera")
    parser.add_argument("--view-preset", default="tabletop_3", choices=sorted(VIEW_PRESETS))
    parser.add_argument(
        "--detector-backend",
        default="mock",
        choices=["auto", "hf", "transformers", "groundingdino", "original", "mock"],
    )
    parser.add_argument("--mock-box-position", default="center", choices=["center", "left", "right", "all"])
    parser.add_argument("--detector-model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--skip-clip", dest="skip_clip", action="store_true", default=True)
    parser.add_argument("--use-clip", dest="skip_clip", action="store_false")
    parser.add_argument("--clip-model-name", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--detector-weight", type=float, default=0.5)
    parser.add_argument("--clip-weight", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--fallback-fov-degrees", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "extrinsic_convention_comparison")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_reports = []
    for query in args.queries:
        for seed in args.seeds:
            run_reports.append(run_one_query_seed(args, query=query, seed=seed))

    report = {
        "created_at": datetime.now().isoformat(),
        "runtime_seconds": time.perf_counter() - started_at,
        "queries": args.queries,
        "seeds": args.seeds,
        "view_preset": args.view_preset,
        "camera_name": args.camera_name,
        "detector_backend": args.detector_backend,
        "skip_clip": bool(args.skip_clip),
        "aggregate": aggregate_runs(run_reports),
        "runs": run_reports,
    }
    report["aggregate"]["conclusion"] = build_conclusion(report["aggregate"])

    output_json = args.output_dir / "extrinsic_convention_report.json"
    output_md = args.output_dir / "extrinsic_convention_report.md"
    write_json(report, output_json)
    output_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"Wrote extrinsic convention JSON: {output_json}")
    print(f"Wrote extrinsic convention MD:   {output_md}")
    print(render_console_summary(report))
    return 0


def run_one_query_seed(args: argparse.Namespace, query: str, seed: int) -> dict[str, Any]:
    """Run one query/seed and compare conventions on the same detections."""

    parsed_query = parse_query(query)
    clip_prompts = build_clip_prompts(parsed_query)
    view_rows: list[dict[str, Any]] = []

    scene = ManiSkillScene(
        env_name=args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        camera_name=args.camera_name,
    )
    try:
        scene.reset(seed=seed)
        for view in VIEW_PRESETS[args.view_preset]:
            scene.set_camera_look_at(camera_name=args.camera_name, eye=view.eye, target=view.target, up=view.up)
            raw_observation = scene.capture_sensor_observation()
            frame = extract_observation_frame(raw_observation, camera_name=args.camera_name)
            if frame.rgb is None or frame.depth is None:
                raise RuntimeError(f"View {view.label!r} is missing RGB or depth.")

            detections = detect_candidates(
                image=frame.rgb,
                text_prompt=parsed_query["normalized_prompt"],
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                top_k=args.top_k,
                backend=args.detector_backend,
                model_id=args.detector_model_id,
                device=args.device,
                mock_box_position=args.mock_box_position,
            )
            ranked = rank_candidates(args=args, frame_rgb=frame.rgb, detections=detections, clip_prompts=clip_prompts)
            conventions = build_extrinsic_conventions(raw_observation, camera_name=args.camera_name)
            view_rows.extend(
                lift_ranked_candidates_for_conventions(
                    args=args,
                    view_id=view.label,
                    frame=frame,
                    ranked=ranked,
                    conventions=conventions,
                )
            )
    finally:
        scene.close()

    return {
        "query": query,
        "seed": int(seed),
        "num_rows": len(view_rows),
        "convention_metrics": metrics_by_convention(view_rows),
        "rows": view_rows,
    }


def rank_candidates(
    args: argparse.Namespace,
    frame_rgb: np.ndarray,
    detections: list[Any],
    clip_prompts: list[str],
) -> list[RankedCandidate]:
    """Rank detector candidates with optional CLIP."""

    if args.skip_clip:
        return rank_detections_without_clip(detections)
    return rerank_candidates_with_clip(
        image=frame_rgb,
        candidates=detections,
        text_prompt=clip_prompts,
        detector_weight=args.detector_weight,
        clip_weight=args.clip_weight,
        model_name=args.clip_model_name,
        pretrained=args.clip_pretrained,
        device=args.device,
    )


def build_extrinsic_conventions(observation: dict[str, Any], camera_name: str) -> list[ExtrinsicConvention]:
    """Build candidate camera-to-world transforms for OpenCV-style lifted points."""

    conventions: list[ExtrinsicConvention] = []
    cam2world_gl, gl_key = extract_observation_matrix_by_leaf(
        observation,
        leaf_key="cam2world_gl",
        valid_shape=(4, 4),
        camera_name=camera_name,
    )
    extrinsic_cv, cv_key = extract_observation_matrix_by_leaf(
        observation,
        leaf_key="extrinsic_cv",
        valid_shape=(4, 4),
        camera_name=camera_name,
    )

    if cam2world_gl is not None and gl_key is not None:
        conventions.append(ExtrinsicConvention("cam2world_gl_direct", cam2world_gl, gl_key))
        conventions.append(ExtrinsicConvention("cam2world_gl_cv_to_gl", cam2world_gl @ CV_TO_GL, gl_key))
    if extrinsic_cv is not None and cv_key is not None:
        conventions.append(ExtrinsicConvention("extrinsic_cv_direct", extrinsic_cv, cv_key))
        try:
            conventions.append(ExtrinsicConvention("extrinsic_cv_inverse", np.linalg.inv(extrinsic_cv), cv_key))
        except np.linalg.LinAlgError:
            pass
    return conventions


def lift_ranked_candidates_for_conventions(
    args: argparse.Namespace,
    view_id: str,
    frame: Any,
    ranked: list[RankedCandidate],
    conventions: list[ExtrinsicConvention],
) -> list[dict[str, Any]]:
    """Lift each ranked candidate under every candidate extrinsic convention."""

    rows: list[dict[str, Any]] = []
    for convention in conventions:
        for candidate in ranked:
            lifted = lift_box_to_3d(
                rgb=frame.rgb,
                depth=frame.depth,
                box_xyxy=candidate.box_xyxy,
                intrinsic=frame.camera_info.intrinsic,
                extrinsic=convention.matrix,
                depth_scale=args.depth_scale,
                fallback_fov_degrees=args.fallback_fov_degrees,
            )
            if lifted.world_xyz is None:
                continue
            rows.append(
                {
                    "view_id": view_id,
                    "convention": convention.name,
                    "source_key": convention.source_key,
                    "rank": int(candidate.rank),
                    "phrase": candidate.phrase,
                    "det_score": float(candidate.det_score),
                    "fused_2d_score": float(candidate.fused_2d_score),
                    "box_xyxy": np.asarray(candidate.box_xyxy, dtype=float).tolist(),
                    "camera_xyz": None if lifted.camera_xyz is None else np.asarray(lifted.camera_xyz, dtype=float).tolist(),
                    "world_xyz": np.asarray(lifted.world_xyz, dtype=float).tolist(),
                    "num_points": int(lifted.num_points),
                    "depth_valid_ratio": float(lifted.depth_valid_ratio),
                }
            )
    return rows


def metrics_by_convention(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute per-convention geometry metrics."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["convention"]), []).append(row)

    return {
        convention: {
            "num_candidates": len(items),
            "mean_top_rank_pairwise_distance": mean(top_rank_pairwise_distances(items)),
            "mean_same_label_pairwise_distance": mean(same_label_pairwise_distances(items)),
            "mean_world_z": mean(row["world_xyz"][2] for row in items),
            "source_keys": sorted({str(row["source_key"]) for row in items}),
        }
        for convention, items in sorted(grouped.items())
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate convention metrics across query/seed runs."""

    convention_names = sorted(
        {
            convention
            for run in runs
            for convention in run.get("convention_metrics", {})
        }
    )
    by_convention: dict[str, dict[str, Any]] = {}
    for convention in convention_names:
        metrics = [run["convention_metrics"][convention] for run in runs if convention in run["convention_metrics"]]
        by_convention[convention] = {
            "num_runs": len(metrics),
            "mean_num_candidates": mean(metric["num_candidates"] for metric in metrics),
            "mean_top_rank_pairwise_distance": mean(metric["mean_top_rank_pairwise_distance"] for metric in metrics),
            "mean_same_label_pairwise_distance": mean(metric["mean_same_label_pairwise_distance"] for metric in metrics),
            "mean_world_z": mean(metric["mean_world_z"] for metric in metrics),
            "source_keys": sorted({key for metric in metrics for key in metric["source_keys"]}),
        }

    best = min(
        by_convention,
        key=lambda name: by_convention[name]["mean_same_label_pairwise_distance"],
        default=None,
    )
    return {
        "total_runs": len(runs),
        "by_convention": by_convention,
        "best_convention_by_same_label_distance": best,
    }


def build_conclusion(aggregate: dict[str, Any]) -> str:
    """Build a compact rule-based conclusion."""

    best = aggregate.get("best_convention_by_same_label_distance")
    metrics = aggregate.get("by_convention", {})
    if not best or best not in metrics:
        return "No valid convention comparison was produced."

    best_distance = float(metrics[best]["mean_same_label_pairwise_distance"])
    current_distance = float(metrics.get("cam2world_gl_direct", {}).get("mean_same_label_pairwise_distance", best_distance))
    if best != "cam2world_gl_direct" and best_distance + 1e-6 < current_distance:
        return (
            f"`{best}` gives the lowest same-label cross-view distance "
            f"({best_distance:.4f} vs current cam2world_gl_direct {current_distance:.4f}). "
            "Patch RGB-D lifting to use this convention if the visual sanity check agrees."
        )
    return (
        f"`{best}` gives the lowest same-label cross-view distance ({best_distance:.4f}), "
        "but it does not improve over the current convention enough to explain fusion fragmentation."
    )


def render_markdown(report: dict[str, Any]) -> str:
    """Render the convention comparison as Markdown."""

    aggregate = report["aggregate"]
    lines = [
        "# Extrinsic Convention Comparison",
        "",
        f"- Queries: `{', '.join(report['queries'])}`",
        f"- Seeds: `{', '.join(str(seed) for seed in report['seeds'])}`",
        f"- View preset: `{report['view_preset']}`",
        f"- Camera: `{report['camera_name']}`",
        f"- Detector backend: `{report['detector_backend']}`",
        "",
        "## Aggregate",
        "",
        "| convention | runs | candidates/run | top_rank_dist | same_label_dist | mean_world_z | source_keys |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for convention, metrics in aggregate["by_convention"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    convention,
                    str(metrics["num_runs"]),
                    format_float(metrics["mean_num_candidates"]),
                    format_float(metrics["mean_top_rank_pairwise_distance"]),
                    format_float(metrics["mean_same_label_pairwise_distance"]),
                    format_float(metrics["mean_world_z"]),
                    ", ".join(f"`{key}`" for key in metrics["source_keys"]),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Per-Run", ""])
    for run in report["runs"]:
        lines.extend(render_run_markdown(run))
    lines.extend(["## Conclusion", "", aggregate["conclusion"], ""])
    return "\n".join(lines)


def render_run_markdown(run: dict[str, Any]) -> list[str]:
    """Render one query/seed block."""

    lines = [
        f"### {run['query']} seed={run['seed']}",
        "",
        "| convention | candidates | top_rank_dist | same_label_dist | mean_world_z |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for convention, metrics in run["convention_metrics"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    convention,
                    str(metrics["num_candidates"]),
                    format_float(metrics["mean_top_rank_pairwise_distance"]),
                    format_float(metrics["mean_same_label_pairwise_distance"]),
                    format_float(metrics["mean_world_z"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def render_console_summary(report: dict[str, Any]) -> str:
    """Return a concise terminal summary."""

    aggregate = report["aggregate"]
    best = aggregate["best_convention_by_same_label_distance"]
    return "\n".join(
        [
            "Extrinsic convention comparison complete",
            f"  Runs:             {aggregate['total_runs']}",
            f"  Best convention:  {best}",
            f"  Conclusion:       {aggregate['conclusion']}",
        ]
    )


def top_rank_pairwise_distances(rows: list[dict[str, Any]]) -> list[float]:
    """Return pairwise distances among top-ranked candidates from different views."""

    top_rows = [row for row in rows if int(row.get("rank", -1)) == 0]
    return pairwise_distances([row["world_xyz"] for row in top_rows])


def same_label_pairwise_distances(rows: list[dict[str, Any]]) -> list[float]:
    """Return same-label pairwise distances within a convention."""

    distances: list[float] = []
    for index, row in enumerate(rows):
        for other in rows[index + 1 :]:
            if row["phrase"] and row["phrase"] == other["phrase"]:
                distances.append(euclidean_distance(row["world_xyz"], other["world_xyz"]))
    return distances


def pairwise_distances(points: list[list[float]]) -> list[float]:
    """Return all pairwise distances."""

    distances: list[float] = []
    for index, point in enumerate(points):
        for other in points[index + 1 :]:
            distances.append(euclidean_distance(point, other))
    return distances


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return 3D Euclidean distance."""

    return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)))


def mean(values: Iterable[float]) -> float:
    """Return mean with zero fallback."""

    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return sum(values_list) / len(values_list)


def format_float(value: float) -> str:
    return f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
