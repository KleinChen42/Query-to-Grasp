"""Recapture representative Query-to-Grasp demo stories as continuous execution videos."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_demo_video_pack import DemoStorySpec, build_demo_video_pack  # noqa: E402
from src.io.export_utils import write_json  # noqa: E402


@dataclass(frozen=True)
class DemoExecutionStory:
    label: str
    runner: str
    seed: int
    caption: str
    args: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small H200 demo recapture pack with continuous execution videos.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "h200_60071_demo_execution_capture_latest")
    parser.add_argument("--demo-pack-output-dir", type=Path, default=Path("outputs") / "demo_video_pack_latest")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--camera-name", default="base_camera")
    parser.add_argument("--every-n-steps", type=int, default=1)
    parser.add_argument("--width", type=int, default=1920, help="Output width for high-resolution execution videos.")
    parser.add_argument("--height", type=int, default=1080, help="Output height for high-resolution execution videos.")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_demo_execution_capture_pack(args)
    print(f"Wrote execution capture pack: {args.output_dir}")
    print(f"  Stories: {len(manifest['stories'])}")
    print(f"  Failed:  {manifest['failed_count']}")
    print(f"  Demo pack: {args.demo_pack_output_dir}")
    return 0 if manifest["failed_count"] == 0 else 1


def run_demo_execution_capture_pack(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stories = build_story_specs(args)
    results: list[dict[str, Any]] = []
    for story in stories:
        story_dir = args.output_dir / story.label
        story_dir.mkdir(parents=True, exist_ok=True)
        command = build_story_command(story, args=args, story_dir=story_dir)
        start_time = time.time()
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        newest_run = find_newest_summary_dir(story_dir, start_time=start_time)
        result = {
            "label": story.label,
            "seed": story.seed,
            "runner": story.runner,
            "caption": story.caption,
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "story_dir": str(story_dir),
            "run_dir": None if newest_run is None else str(newest_run),
            "summary_path": None if newest_run is None else str(newest_run / "summary.json"),
            "execution_video_path": find_execution_video(newest_run),
        }
        results.append(result)
        write_json(result, story_dir / "capture_result.json")
        if completed.returncode != 0 and not args.continue_on_error:
            break

    demo_manifest = build_demo_video_pack(
        specs=[
            DemoStorySpec(
                label=story.label,
                source_dir=args.output_dir / story.label,
                caption=story.caption,
                desired_outcomes=("success", "failure"),
            )
            for story in stories
        ],
        output_dir=args.demo_pack_output_dir,
        skip_missing=False,
        max_media_per_story=24,
        make_slideshows=False,
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.output_dir),
        "demo_pack_output_dir": str(args.demo_pack_output_dir),
        "failed_count": sum(1 for result in results if int(result["returncode"]) != 0),
        "stories": results,
        "demo_pack_manifest": demo_manifest,
    }
    write_json(manifest, args.output_dir / "manifest.json")
    (args.output_dir / "README.md").write_text(render_readme(manifest), encoding="utf-8")
    return manifest


def build_story_specs(args: argparse.Namespace) -> tuple[DemoExecutionStory, ...]:
    common_multiview = (
        "--query",
        "red cube",
        "--obs-mode",
        "rgbd",
        "--view-preset",
        "tabletop_3",
        "--detector-backend",
        "hf",
        "--skip-clip",
        "--depth-scale",
        "1000",
        "--control-mode",
        "pd_ee_delta_pos",
        "--grasp-target-mode",
        "refined",
    )
    closed_loop_flags = (
        "--enable-closed-loop-reobserve",
        "--enable-selected-object-continuity",
        "--enable-post-reobserve-selection-continuity",
    )
    stack_place = (
        "--env-id",
        "StackCube-v1",
        "--pick-executor",
        "sim_pick_place",
        "--place-target-source",
        "oracle_cubeB_pose",
    )
    return (
        DemoExecutionStory(
            label="pickcube_full_query_success",
            runner="multiview",
            seed=2,
            caption="PickCube full-query multi-view success with memory_grasp_world_xyz driving sim_topdown.",
            args=(
                *common_multiview,
                "--env-id",
                "PickCube-v1",
                "--pick-executor",
                "sim_topdown",
                *closed_loop_flags,
            ),
        ),
        DemoExecutionStory(
            label="stackcube_query_place_success",
            runner="single",
            seed=0,
            caption="StackCube single-view query-derived cubeA pick plus oracle_cubeB_pose placement bridge.",
            args=(
                "--query",
                "red cube",
                "--env-id",
                "StackCube-v1",
                "--obs-mode",
                "rgbd",
                "--detector-backend",
                "hf",
                "--depth-scale",
                "1000",
                "--control-mode",
                "pd_ee_delta_pos",
                "--pick-executor",
                "sim_pick_place",
                "--grasp-target-mode",
                "refined",
                "--place-target-source",
                "oracle_cubeB_pose",
            ),
        ),
        DemoExecutionStory(
            label="stackcube_tabletop_success",
            runner="multiview",
            seed=0,
            caption="StackCube tabletop success with task_guard_selected_object_world_xyz and oracle_cubeB_pose.",
            args=(*common_multiview, *stack_place),
        ),
        DemoExecutionStory(
            label="stackcube_tabletop_failure",
            runner="multiview",
            seed=1,
            caption="StackCube tabletop failure illustrating target-source and execution sensitivity.",
            args=(*common_multiview, *stack_place),
        ),
        DemoExecutionStory(
            label="stackcube_closed_loop_failure",
            runner="multiview",
            seed=0,
            caption="StackCube closed-loop limitation: re-observation remains diagnostic rather than universally helpful.",
            args=(*common_multiview, *stack_place, *closed_loop_flags),
        ),
        DemoExecutionStory(
            label="stackcube_closed_loop_success",
            runner="multiview",
            seed=5,
            caption="StackCube closed-loop success contrast under the same diagnostic execution path.",
            args=(*common_multiview, *stack_place, *closed_loop_flags),
        ),
    )


def build_story_command(story: DemoExecutionStory, args: argparse.Namespace, story_dir: Path) -> list[str]:
    runner = "run_single_view_pick.py" if story.runner == "single" else "run_multiview_fusion_debug.py"
    return [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / runner),
        *story.args,
        "--seed",
        str(story.seed),
        "--output-dir",
        str(story_dir),
        "--capture-execution-video",
        "--execution-video-fps",
        str(args.fps),
        "--execution-video-camera-name",
        args.camera_name,
        "--execution-video-every-n-steps",
        str(args.every_n_steps),
        "--execution-video-width",
        str(args.width),
        "--execution-video-height",
        str(args.height),
    ]


def find_newest_summary_dir(root: Path, start_time: float) -> Path | None:
    candidates = [
        path.parent
        for path in root.rglob("summary.json")
        if path.stat().st_mtime >= start_time - 1.0
    ]
    if not candidates:
        candidates = [path.parent for path in root.rglob("summary.json")]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def find_execution_video(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    candidates = sorted(run_dir.rglob("execution_video.mp4"))
    return str(candidates[0]) if candidates else None


def render_readme(manifest: dict[str, Any]) -> str:
    lines = [
        "# Continuous Demo Execution Capture Pack",
        "",
        "Representative ManiSkill execution videos for the Query-to-Grasp supplemental video.",
        "",
        f"- Failed stories: {manifest['failed_count']}",
        f"- Demo pack: `{manifest['demo_pack_output_dir']}`",
        "- Video capture: continuous execution frames, letterboxed to the requested output resolution.",
        "",
        "| story | seed | returncode | execution video |",
        "| --- | ---: | ---: | --- |",
    ]
    for story in manifest["stories"]:
        lines.append(
            f"| {story['label']} | {story['seed']} | {story['returncode']} | "
            f"`{story.get('execution_video_path') or 'missing'}` |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
