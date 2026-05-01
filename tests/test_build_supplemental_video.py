from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from scripts.build_supplemental_video import (
    Segment,
    build_supplemental_video,
    continuous_story_frame,
    smooth_story_frame,
    validate_claim_boundaries,
)


cv2 = pytest.importorskip("cv2")


def write_tiny_video(path: Path) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(3):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:] = (index * 40, 60, 120)
            writer.write(frame)
    finally:
        writer.release()


def write_demo_manifest(root: Path) -> Path:
    labels = [
        "pickcube_full_query_success",
        "stackcube_query_place_success",
        "stackcube_tabletop_success",
        "stackcube_tabletop_failure",
        "stackcube_closed_loop_failure",
        "stackcube_closed_loop_success",
    ]
    stories = []
    for label in labels:
        relative = Path("media") / label / f"{label}.mp4"
        write_tiny_video(root / relative)
        stories.append(
            {
                "label": label,
                "exists": True,
                "slideshow_path": relative.as_posix(),
                "media_kind": "execution_video",
            }
        )
    manifest = {
        "created_at": "2026-05-01T00:00:00+00:00",
        "stories": stories,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_build_supplemental_video_writes_video_and_metadata(tmp_path: Path) -> None:
    input_manifest = write_demo_manifest(tmp_path / "demo")
    output_dir = tmp_path / "supplemental"

    manifest = build_supplemental_video(
        input_manifest=input_manifest,
        output_dir=output_dir,
        fps=2.0,
        width=320,
        height=180,
    )

    assert (output_dir / "query_to_grasp_supplemental_video.mp4").stat().st_size > 0
    assert (output_dir / "storyboard.md").exists()
    assert (output_dir / "captions.json").exists()
    assert (output_dir / "manifest.json").exists()
    assert manifest["duration_seconds"] == pytest.approx(66.0)
    captions = json.loads((output_dir / "captions.json").read_text(encoding="utf-8"))
    assert captions[0]["label"] == "title"
    assert any("oracle_cubeB_pose" in item["caption"] for item in captions)


def test_smooth_story_frame_holds_source_frames_before_transition() -> None:
    import numpy as np

    frames = []
    for value in (20, 100, 180):
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:] = value
        frames.append(frame)

    assert smooth_story_frame(frames, index=0, total_frames=30, transition_frames=0, cv2=cv2)[0, 0, 0] == 20
    assert smooth_story_frame(frames, index=8, total_frames=30, transition_frames=0, cv2=cv2)[0, 0, 0] == 20
    assert smooth_story_frame(frames, index=10, total_frames=30, transition_frames=0, cv2=cv2)[0, 0, 0] == 100
    assert smooth_story_frame(frames, index=20, total_frames=30, transition_frames=0, cv2=cv2)[0, 0, 0] == 180


def test_continuous_story_frame_preserves_temporal_order() -> None:
    import numpy as np

    frames = []
    for value in (10, 20, 30, 40):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        frame[:] = value
        frames.append(frame)

    values = [continuous_story_frame(frames, index=index, total_frames=4)[0, 0, 0] for index in range(4)]

    assert values == [10, 20, 30, 40]


def test_claim_boundary_rejects_unsupported_positive_claims() -> None:
    with pytest.raises(ValueError, match="real robot"):
        validate_claim_boundaries(
            [
                Segment(
                    kind="card",
                    label="bad",
                    title="Real Robot Success",
                    caption="This is real robot execution.",
                    duration_seconds=1.0,
                )
            ]
        )
