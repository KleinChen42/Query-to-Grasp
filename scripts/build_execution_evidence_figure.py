"""Build a paper-ready execution evidence montage from representative videos."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class MontageStory:
    label: str
    title: str
    subtitle: str
    run_dir: Path
    expected_task_success: bool
    frame_fraction: float = 0.72


DEFAULT_STORIES = (
    MontageStory(
        label="pickcube_success",
        title="PickCube success",
        subtitle="target: memory_grasp_world_xyz",
        run_dir=Path("outputs/h200_60071_demo_execution_capture_native720_latest/pickcube_full_query_success/20260501_163250_red_cube"),
        expected_task_success=True,
        frame_fraction=0.82,
    ),
    MontageStory(
        label="stackcube_single_bridge_success",
        title="StackCube bridge success",
        subtitle="query pick + diagnostic place",
        run_dir=Path("outputs/h200_60071_demo_execution_capture_native720_latest/stackcube_query_place_success/20260501_163530_red_cube"),
        expected_task_success=True,
        frame_fraction=0.88,
    ),
    MontageStory(
        label="stackcube_tabletop_success",
        title="Tabletop success",
        subtitle="task-guarded target",
        run_dir=Path("outputs/h200_60071_demo_execution_capture_native720_latest/stackcube_tabletop_success/20260501_163902_red_cube"),
        expected_task_success=True,
        frame_fraction=0.88,
    ),
    MontageStory(
        label="stackcube_tabletop_failure",
        title="Tabletop failure",
        subtitle="place not confirmed",
        run_dir=Path("outputs/h200_60071_demo_execution_capture_native720_failure_probe/tabletop_seed2/20260501_192139_red_cube"),
        expected_task_success=False,
        frame_fraction=0.88,
    ),
    MontageStory(
        label="stackcube_closed_loop_failure",
        title="Closed-loop failure",
        subtitle="place not confirmed",
        run_dir=Path("outputs/h200_60071_demo_execution_capture_native720_failure_probe/closed_loop_seed2/20260501_192801_red_cube"),
        expected_task_success=False,
        frame_fraction=0.88,
    ),
    MontageStory(
        label="stackcube_closed_loop_success",
        title="Closed-loop success",
        subtitle="diagnostic path can succeed",
        run_dir=Path("outputs/h200_60071_demo_execution_capture_native720_latest/stackcube_closed_loop_success/20260501_164754_red_cube"),
        expected_task_success=True,
        frame_fraction=0.88,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build execution evidence montage for the paper.")
    parser.add_argument("--output-dir", type=Path, default=Path("paper") / "figures")
    parser.add_argument("--stem", default="execution_evidence_montage")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_execution_evidence_figure(output_dir=args.output_dir, stem=args.stem)
    print(f"Wrote execution evidence montage to {args.output_dir / args.stem}.pdf")
    return 0


def build_execution_evidence_figure(
    output_dir: Path,
    stem: str = "execution_evidence_montage",
    stories: tuple[MontageStory, ...] = DEFAULT_STORIES,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _import_matplotlib()

    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.25))
    for axis, story in zip(axes.flat, stories):
        run_dir = resolve_path(story.run_dir)
        summary = read_json(run_dir / "summary.json")
        validate_story(summary=summary, story=story)
        frame = read_representative_frame(run_dir=run_dir, fraction=story.frame_fraction)

        axis.imshow(frame)
        axis.set_axis_off()
        outcome = "success" if story.expected_task_success else "failure"
        color = "#16a34a" if story.expected_task_success else "#dc2626"
        axis.text(
            0.02,
            0.98,
            story.title,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            color="white",
            bbox=dict(facecolor="black", alpha=0.62, edgecolor="none", pad=3.0),
        )
        axis.text(
            0.02,
            0.06,
            f"{story.subtitle}\n{outcome}",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.0,
            color="white",
            bbox=dict(facecolor=color, alpha=0.72, edgecolor="none", pad=2.6),
        )

    fig.suptitle("Representative ManiSkill diagnostic executions", fontsize=10.5, y=0.99)
    fig.text(
        0.5,
        0.01,
        "Visualization only: RAS claims are reported in the controlled tables; oracle clips are privileged diagnostics.",
        ha="center",
        fontsize=7.4,
        color="#475569",
    )
    fig.tight_layout(rect=[0.01, 0.035, 0.99, 0.96], w_pad=0.08, h_pad=0.08)
    save_figure(fig, output_dir / stem)


def validate_story(summary: dict[str, Any], story: MontageStory) -> None:
    task_success = bool(summary.get("task_success"))
    if task_success != story.expected_task_success:
        raise ValueError(
            f"Story {story.label} expected task_success={story.expected_task_success} "
            f"but summary reports {task_success}."
        )


def read_representative_frame(run_dir: Path, fraction: float) -> Any:
    video_path = run_dir / "execution_video" / "execution_video.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Missing execution video: {video_path}")

    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required to sample execution videos.") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open execution video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f"Execution video has no frames: {video_path}")
        index = max(0, min(frame_count - 1, int(round((frame_count - 1) * fraction))))
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {index} from {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig: Any, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("matplotlib is required to generate the montage.") from exc


if __name__ == "__main__":
    raise SystemExit(main())
