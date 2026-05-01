from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_execution_evidence_figure import MontageStory, build_execution_evidence_figure


cv2 = pytest.importorskip("cv2")
pytest.importorskip("matplotlib")


def write_tiny_execution_run(run_dir: Path, task_success: bool) -> None:
    import numpy as np

    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"query": "red cube", "task_success": task_success}),
        encoding="utf-8",
    )
    video_path = run_dir / "execution_video" / "execution_video.mp4"
    video_path.parent.mkdir()
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 3.0, (80, 60))
    assert writer.isOpened()
    try:
        for index in range(4):
            frame = np.zeros((60, 80, 3), dtype=np.uint8)
            frame[:] = (20 + index * 20, 80, 160)
            writer.write(frame)
    finally:
        writer.release()


def make_story(tmp_path: Path, index: int, task_success: bool) -> MontageStory:
    run_dir = tmp_path / f"run_{index}"
    write_tiny_execution_run(run_dir, task_success=task_success)
    return MontageStory(
        label=f"story_{index}",
        title=f"Story {index}",
        subtitle="target source",
        run_dir=run_dir,
        expected_task_success=task_success,
    )


def test_build_execution_evidence_figure_writes_pdf_and_png(tmp_path: Path) -> None:
    stories = tuple(make_story(tmp_path, index, task_success=index % 2 == 0) for index in range(6))
    output_dir = tmp_path / "figures"

    build_execution_evidence_figure(output_dir=output_dir, stories=stories)

    assert (output_dir / "execution_evidence_montage.pdf").stat().st_size > 0
    assert (output_dir / "execution_evidence_montage.png").stat().st_size > 0


def test_build_execution_evidence_figure_rejects_mislabeled_story(tmp_path: Path) -> None:
    stories = list(make_story(tmp_path, index, task_success=True) for index in range(6))
    bad_story = stories[0]
    stories[0] = MontageStory(
        label=bad_story.label,
        title=bad_story.title,
        subtitle=bad_story.subtitle,
        run_dir=bad_story.run_dir,
        expected_task_success=False,
    )

    with pytest.raises(ValueError, match="expected task_success=False"):
        build_execution_evidence_figure(output_dir=tmp_path / "figures", stories=tuple(stories))
