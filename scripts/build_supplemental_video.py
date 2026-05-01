"""Assemble a conference supplemental video from the demo video pack."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.export_utils import write_json  # noqa: E402


DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 24.0
FORBIDDEN_CLAIMS = (
    "real robot",
    "learned controller",
    "learned grasp",
    "full non-oracle stackcube",
    "language-conditioned stacking completion",
)


@dataclass(frozen=True)
class Segment:
    """One timed section in the supplemental video."""

    kind: str
    label: str
    title: str
    caption: str
    duration_seconds: float
    video_path: Path | None = None
    media_kind: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Query-to-Grasp supplemental video v0.1.")
    parser.add_argument("--input", type=Path, default=Path("outputs") / "demo_video_pack_latest" / "manifest.json")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "supplemental_video_latest")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_supplemental_video(
            input_manifest=args.input,
            output_dir=args.output_dir,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote supplemental video: {args.output_dir / 'query_to_grasp_supplemental_video.mp4'}")
    print(f"  Storyboard: {args.output_dir / 'storyboard.md'}")
    print(f"  Captions:   {args.output_dir / 'captions.json'}")
    print(f"  Duration:   {manifest['duration_seconds']:.1f}s")
    return 0


def build_supplemental_video(
    input_manifest: Path,
    output_dir: Path,
    fps: float = DEFAULT_FPS,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> dict[str, Any]:
    """Assemble a single MP4 plus storyboard metadata."""

    try:
        import cv2  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("OpenCV is required to assemble the supplemental video.") from exc

    if fps <= 0:
        raise ValueError("--fps must be positive.")
    if width <= 0 or height <= 0:
        raise ValueError("--width and --height must be positive.")

    input_manifest = resolve_path(input_manifest)
    if not input_manifest.exists():
        raise FileNotFoundError(f"Missing demo video manifest: {input_manifest}")
    demo_root = input_manifest.parent
    demo_manifest = json.loads(input_manifest.read_text(encoding="utf-8"))
    stories = {story["label"]: story for story in demo_manifest.get("stories", []) if story.get("exists")}
    segments = build_segments(stories=stories, demo_root=demo_root)
    validate_claim_boundaries(segments)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "query_to_grasp_supplemental_video.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_video}")

    frame_count = 0
    try:
        for segment in segments:
            if segment.kind == "card":
                frame_count += write_card_segment(writer, segment=segment, fps=fps, width=width, height=height, cv2=cv2)
            elif segment.kind == "clip":
                frame_count += write_clip_segment(writer, segment=segment, fps=fps, width=width, height=height, cv2=cv2)
            else:
                raise ValueError(f"Unknown segment kind: {segment.kind}")
    finally:
        writer.release()

    if not output_video.exists() or output_video.stat().st_size <= 0:
        raise RuntimeError(f"Supplemental video was not written: {output_video}")

    captions = [segment_to_caption(segment) for segment in segments]
    duration_seconds = frame_count / fps
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_manifest": str(input_manifest),
        "output_video": str(output_video),
        "duration_seconds": duration_seconds,
        "fps": fps,
        "width": width,
        "height": height,
        "segments": captions,
    }
    write_json(captions, output_dir / "captions.json")
    write_json(manifest, output_dir / "manifest.json")
    (output_dir / "storyboard.md").write_text(render_storyboard(manifest), encoding="utf-8")
    return manifest


def build_segments(stories: dict[str, dict[str, Any]], demo_root: Path) -> list[Segment]:
    """Build the fixed IROS/ICRA-style supplemental video storyboard."""

    return [
        Segment(
            kind="card",
            label="title",
            title="Query-to-Grasp",
            caption=(
                "From open-vocabulary RGB-D retrieval to graspable 3D action targets. "
                "High-fidelity simulation only; no real-robot claim."
            ),
            duration_seconds=6.0,
        ),
        story_segment(
            stories,
            demo_root,
            label="pickcube_full_query_success",
            title="PickCube: Executable 3D Target",
            caption=(
                "PickCube-v1, query red cube, seed 2. The selected target source is "
                "memory_grasp_world_xyz and the simulated pick succeeds."
            ),
            duration_seconds=10.0,
        ),
        story_segment(
            stories,
            demo_root,
            label="stackcube_query_place_success",
            title="StackCube: Query Pick + Oracle Place",
            caption=(
                "StackCube-v1, query red cube, seed 0. The cubeA pick target is query-derived; "
                "the destination is oracle_cubeB_pose. This is a partial oracle-place bridge."
            ),
            duration_seconds=10.0,
        ),
        Segment(
            kind="card",
            label="tabletop_contrast_card",
            title="StackCube Target-Source Contrast",
            caption="Same query, same task family: target-source quality changes task completion.",
            duration_seconds=3.0,
        ),
        story_segment(
            stories,
            demo_root,
            label="stackcube_tabletop_success",
            title="Tabletop Multi-View Success",
            caption=(
                "StackCube-v1 tabletop_3, seed 0. The task guard uses "
                "task_guard_selected_object_world_xyz and oracle_cubeB_pose; task succeeds."
            ),
            duration_seconds=7.0,
        ),
        story_segment(
            stories,
            demo_root,
            label="stackcube_tabletop_failure",
            title="Tabletop Multi-View Failure",
            caption=(
                "StackCube-v1 tabletop_3, seed 2. The place stage is not confirmed, "
                "illustrating the retrieval-to-execution gap."
            ),
            duration_seconds=7.0,
        ),
        Segment(
            kind="card",
            label="closed_loop_card",
            title="Closed-Loop Re-Observation Is Diagnostic",
            caption=(
                "Re-observation can reduce uncertainty, but it does not guarantee higher "
                "physical task success."
            ),
            duration_seconds=3.0,
        ),
        story_segment(
            stories,
            demo_root,
            label="stackcube_closed_loop_failure",
            title="Closed-Loop Limitation",
            caption=(
                "StackCube-v1 closed-loop, seed 2. The selected target source is "
                "task_guard_selected_object_world_xyz, place source is oracle_cubeB_pose, "
                "and task success is false."
            ),
            duration_seconds=7.0,
        ),
        story_segment(
            stories,
            demo_root,
            label="stackcube_closed_loop_success",
            title="Closed-Loop Success Contrast",
            caption=(
                "StackCube-v1 closed-loop, seed 5. The same diagnostic path can succeed, "
                "but success is not universal."
            ),
            duration_seconds=7.0,
        ),
        Segment(
            kind="card",
            label="claim_boundary",
            title="Claim Boundary",
            caption=(
                "This video supports simulated retrieval-to-execution diagnostics. It does "
                "not claim real-robot execution, learned grasping, or full non-oracle "
                "StackCube stacking."
            ),
            duration_seconds=6.0,
        ),
    ]


def story_segment(
    stories: dict[str, dict[str, Any]],
    demo_root: Path,
    label: str,
    title: str,
    caption: str,
    duration_seconds: float,
) -> Segment:
    story = stories.get(label)
    if not story:
        raise FileNotFoundError(f"Demo story missing from manifest: {label}")
    slideshow_path = story.get("slideshow_path")
    if not slideshow_path:
        raise FileNotFoundError(f"Demo story has no slideshow_path: {label}")
    video_path = demo_root / slideshow_path
    if not video_path.exists():
        raise FileNotFoundError(f"Demo story slideshow missing: {video_path}")
    return Segment(
        kind="clip",
        label=label,
        title=title,
        caption=caption,
        duration_seconds=duration_seconds,
        video_path=video_path,
        media_kind=story.get("media_kind"),
    )


def write_card_segment(writer: Any, segment: Segment, fps: float, width: int, height: int, cv2: Any) -> int:
    frame = make_text_frame(segment.title, segment.caption, width=width, height=height, cv2=cv2)
    frames = int(round(segment.duration_seconds * fps))
    for _ in range(frames):
        writer.write(frame)
    return frames


def write_clip_segment(writer: Any, segment: Segment, fps: float, width: int, height: int, cv2: Any) -> int:
    assert segment.video_path is not None
    frames = read_video_frames(segment.video_path, width=width, height=height, cv2=cv2)
    if not frames:
        raise RuntimeError(f"No readable frames in {segment.video_path}")
    total = int(round(segment.duration_seconds * fps))
    transition_frames = max(1, int(round(0.35 * fps)))
    for index in range(total):
        if segment.media_kind == "execution_video":
            frame = continuous_story_frame(frames, index=index, total_frames=total)
        else:
            frame = smooth_story_frame(
                frames,
                index=index,
                total_frames=total,
                transition_frames=transition_frames,
                cv2=cv2,
            )
        writer.write(overlay_caption(frame, segment.title, segment.caption, cv2=cv2))
    return total


def continuous_story_frame(frames: list[Any], index: int, total_frames: int) -> Any:
    """Map output time to source time for real execution videos."""

    if not frames:
        raise ValueError("frames must be non-empty.")
    if len(frames) == 1 or total_frames <= 1:
        return frames[0]
    source_index = int(round(index * (len(frames) - 1) / max(total_frames - 1, 1)))
    return frames[min(max(source_index, 0), len(frames) - 1)]


def smooth_story_frame(
    frames: list[Any],
    index: int,
    total_frames: int,
    transition_frames: int,
    cv2: Any,
) -> Any:
    """Return a stable story frame with optional crossfade between stills.

    Demo story clips are often lightweight slideshows made from a few PNGs. Holding
    each source frame for its share of the segment avoids rapid modulo cycling.
    """

    if not frames:
        raise ValueError("frames must be non-empty.")
    if len(frames) == 1 or total_frames <= 1:
        return frames[0]

    frames_per_source = max(total_frames / len(frames), 1.0)
    source_index = min(int(index / frames_per_source), len(frames) - 1)
    if source_index >= len(frames) - 1 or transition_frames <= 0:
        return frames[source_index]

    next_boundary = (source_index + 1) * frames_per_source
    max_transition = max(1, int(frames_per_source / 2))
    effective_transition = min(max_transition, transition_frames)
    fade_start = next_boundary - effective_transition
    if index < fade_start:
        return frames[source_index]

    alpha = min(max((index - fade_start) / effective_transition, 0.0), 1.0)
    return cv2.addWeighted(frames[source_index], 1.0 - alpha, frames[source_index + 1], alpha, 0)


def read_video_frames(video_path: Path, width: int, height: int, cv2: Any) -> list[Any]:
    capture = cv2.VideoCapture(str(video_path))
    frames: list[Any] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    return frames


def make_text_frame(title: str, body: str, width: int, height: int, cv2: Any) -> Any:
    import numpy as np

    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (20, 26, 32)
    cv2.rectangle(frame, (0, 0), (width, height), (34, 43, 52), thickness=24)
    draw_wrapped_text(frame, title, x=80, y=170, max_width=width - 160, scale=1.55, color=(245, 248, 250), cv2=cv2)
    draw_wrapped_text(frame, body, x=82, y=310, max_width=width - 164, scale=0.82, color=(210, 220, 228), cv2=cv2)
    return frame


def overlay_caption(frame: Any, title: str, caption: str, cv2: Any) -> Any:
    output = frame.copy()
    height, width = output.shape[:2]
    overlay = output.copy()
    cv2.rectangle(overlay, (0, height - 178), (width, height), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.58, output, 0.42, 0, output)
    draw_wrapped_text(output, title, x=42, y=height - 126, max_width=width - 84, scale=0.82, color=(255, 255, 255), cv2=cv2)
    draw_wrapped_text(output, caption, x=42, y=height - 78, max_width=width - 84, scale=0.54, color=(224, 232, 238), cv2=cv2)
    return output


def draw_wrapped_text(
    frame: Any,
    text: str,
    x: int,
    y: int,
    max_width: int,
    scale: float,
    color: tuple[int, int, int],
    cv2: Any,
) -> None:
    words = text.split()
    line = ""
    line_height = int(40 * scale) + 12
    current_y = y
    for word in words:
        candidate = word if not line else f"{line} {word}"
        size = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        if size[0] <= max_width or not line:
            line = candidate
            continue
        cv2.putText(frame, line, (x, current_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        current_y += line_height
        line = word
    if line:
        cv2.putText(frame, line, (x, current_y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def segment_to_caption(segment: Segment) -> dict[str, Any]:
    return {
        "label": segment.label,
        "kind": segment.kind,
        "title": segment.title,
        "caption": segment.caption,
        "duration_seconds": segment.duration_seconds,
        "video_path": None if segment.video_path is None else str(segment.video_path),
        "media_kind": segment.media_kind,
    }


def validate_claim_boundaries(segments: list[Segment]) -> None:
    """Reject unsupported claim wording while allowing explicit non-claim boundaries."""

    for segment in segments:
        text = f"{segment.title} {segment.caption}".lower()
        for forbidden in FORBIDDEN_CLAIMS:
            if forbidden in text and "does not claim" not in text and "no real-robot claim" not in text:
                raise ValueError(f"Unsupported claim wording in {segment.label}: {forbidden}")


def render_storyboard(manifest: dict[str, Any]) -> str:
    lines = [
        "# Query-to-Grasp Supplemental Video Storyboard",
        "",
        "This storyboard is generated from the demo video pack and is intended for IROS/ICRA-style supplemental material.",
        "It uses representative simulation runs only and does not introduce new experimental claims.",
        "",
        f"- Output video: `{Path(manifest['output_video']).name}`",
        f"- Duration: {manifest['duration_seconds']:.1f} seconds",
        f"- Resolution: {manifest['width']}x{manifest['height']} @ {manifest['fps']} fps",
        "",
        "| order | label | duration | caption |",
        "| ---: | --- | ---: | --- |",
    ]
    for index, segment in enumerate(manifest["segments"], start=1):
        lines.append(
            f"| {index} | {escape_table_cell(segment['label'])} | "
            f"{segment['duration_seconds']:.1f}s | {escape_table_cell(segment['caption'])} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- No real-robot execution is claimed.",
            "- No learned controller or learned grasping is claimed.",
            "- StackCube bridge clips use a privileged `oracle_cubeB_pose` placement target.",
            "- Closed-loop clips are diagnostic; they are not claimed to universally improve task success.",
            "",
        ]
    )
    return "\n".join(lines)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def escape_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
