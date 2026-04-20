from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

import scripts.run_single_view_query as runner
from src.env.camera_utils import CameraInfo, ObservationFrame
from src.perception.grounding_dino import detect_candidates


class FakeManiSkillScene:
    def __init__(self, *args, **kwargs) -> None:
        self.env_name = kwargs.get("env_name", "FakeEnv-v0")

    def reset(self, seed=None):
        return {}

    def get_observation(self, camera_name=None) -> ObservationFrame:
        height, width = 32, 40
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, :, 0] = 128
        depth = np.ones((height, width), dtype=np.float32)
        intrinsic = np.array(
            [
                [30.0, 0.0, (width - 1) / 2.0],
                [0.0, 30.0, (height - 1) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return ObservationFrame(
            rgb=rgb,
            depth=depth,
            segmentation=None,
            camera_info=CameraInfo(camera_name=camera_name, intrinsic=intrinsic),
            observation_keys=["sensor_data.fake.rgb", "sensor_data.fake.depth"],
            observation_summary={},
            source_keys={"rgb": "sensor_data.fake.rgb", "depth": "sensor_data.fake.depth"},
        )

    def close(self) -> None:
        return None


def test_mock_detector_returns_image_size_aware_boxes() -> None:
    image = np.zeros((20, 30, 3), dtype=np.uint8)

    center = detect_candidates(image, "red cube", backend="mock", mock_box_position="center", top_k=3)
    all_boxes = detect_candidates(image, "red cube", backend="mock", mock_box_position="all", top_k=3)

    assert len(center) == 1
    assert len(all_boxes) == 3
    assert center[0].source == "mock"
    assert center[0].box_xyxy[0] >= 0
    assert center[0].box_xyxy[2] <= image.shape[1] - 1
    assert all(candidate.box_xyxy[3] <= image.shape[0] - 1 for candidate in all_boxes)


def test_single_view_runner_mock_skip_clip_creates_json_outputs(monkeypatch) -> None:
    output_root = Path("outputs/test_single_view_runner")
    before = set(output_root.glob("*")) if output_root.exists() else set()

    monkeypatch.setattr(runner, "ManiSkillScene", FakeManiSkillScene)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_single_view_query.py",
            "--query",
            "red cube",
            "--detector-backend",
            "mock",
            "--mock-box-position",
            "center",
            "--skip-clip",
            "--top-k",
            "1",
            "--output-dir",
            str(output_root),
        ],
    )

    runner.main()

    after = set(output_root.glob("*"))
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    run_dir = created[-1] if created else max(after, key=lambda path: path.stat().st_mtime)
    for file_name in [
        "parsed_query.json",
        "detections.json",
        "reranked_candidates.json",
        "top_candidate_3d.json",
        "summary.json",
    ]:
        assert (run_dir / file_name).exists()
