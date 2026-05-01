from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_demo_execution_capture_pack import DemoExecutionStory, build_story_command


def test_demo_execution_capture_command_forwards_high_resolution_video_flags(tmp_path: Path) -> None:
    story = DemoExecutionStory(
        label="demo",
        runner="single",
        seed=7,
        caption="Demo.",
        args=("--query", "red cube"),
    )
    args = argparse.Namespace(fps=24.0, camera_name="base_camera", every_n_steps=1, width=1920, height=1080)

    command = build_story_command(story, args=args, story_dir=tmp_path / "story")

    assert "--capture-execution-video" in command
    assert command[command.index("--execution-video-width") + 1] == "1920"
    assert command[command.index("--execution-video-height") + 1] == "1080"
