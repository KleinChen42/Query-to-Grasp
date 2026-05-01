from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.io.execution_video import ExecutionVideoRecorder


cv2 = pytest.importorskip("cv2")


def maniskill_like_observation(value: int = 120) -> dict[str, object]:
    rgb = np.zeros((1, 4, 5, 3), dtype=np.uint8)
    rgb[:] = value
    return {
        "sensor_data": {
            "base_camera": {
                "rgb": rgb,
                "depth": np.ones((1, 4, 5, 1), dtype=np.float32),
            }
        },
        "sensor_param": {
            "base_camera": {
                "intrinsic_cv": np.eye(3, dtype=np.float32),
                "cam2world_gl": np.eye(4, dtype=np.float32),
            }
        },
    }


def test_execution_video_recorder_writes_frames_video_and_metadata(tmp_path: Path) -> None:
    recorder = ExecutionVideoRecorder(output_dir=tmp_path / "video", fps=2.0, camera_name="base_camera")

    for index in range(3):
        recorder.record_step(
            stage="lift",
            action=np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32),
            observation=maniskill_like_observation(40 + index),
            info={"success": False},
        )
    manifest = recorder.finalize()

    assert manifest["status"] == "ok"
    assert manifest["frame_count"] == 3
    assert (tmp_path / "video" / "execution_video.mp4").stat().st_size > 0
    metadata = json.loads((tmp_path / "video" / "execution_video_metadata.json").read_text(encoding="utf-8"))
    assert metadata["records"][0]["stage"] == "lift"
    assert metadata["records"][0]["action"] == [0.0, 0.0, 1.0, -1.0]


def test_execution_video_recorder_reports_missing_rgb(tmp_path: Path) -> None:
    recorder = ExecutionVideoRecorder(output_dir=tmp_path / "video")

    recorder.record_step(stage="descend", action=np.zeros(4, dtype=np.float32), observation={}, info={})
    manifest = recorder.finalize()

    assert manifest["status"] == "not_written_no_frames"
    assert manifest["frame_count"] == 0
    assert manifest["capture_failures"][0]["reason"] == "capture_failed_missing_rgb"
