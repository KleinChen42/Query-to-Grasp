from __future__ import annotations

import numpy as np

import scripts.check_maniskill_camera_views as camera_probe


def test_infer_candidate_camera_names_from_maniskill_style_keys() -> None:
    keys = [
        "sensor_data.base_camera.rgb",
        "sensor_data.base_camera.depth",
        "sensor_data.hand_camera.rgb",
        "sensor_param.base_camera.intrinsic_cv",
        "sensor_param.hand_camera.cam2world_gl",
        "extra.tcp_pose",
        "extra.goal_pos",
    ]

    assert camera_probe.infer_candidate_camera_names(keys) == ["base_camera", "hand_camera"]


def test_build_camera_probe_report_finds_usable_named_views() -> None:
    observation = {
        "sensor_data": {
            "base_camera": {
                "rgb": np.zeros((4, 5, 3), dtype=np.uint8),
                "depth": np.ones((4, 5, 1), dtype=np.float32),
            },
            "hand_camera": {
                "rgb": np.zeros((4, 5, 3), dtype=np.uint8),
                "depth": np.ones((4, 5, 1), dtype=np.float32),
            },
        },
        "sensor_param": {
            "base_camera": {
                "intrinsic_cv": np.eye(3, dtype=np.float32),
                "cam2world_gl": np.eye(4, dtype=np.float32),
            },
            "hand_camera": {
                "intrinsic_cv": np.eye(3, dtype=np.float32),
                "cam2world_gl": np.eye(4, dtype=np.float32),
            },
        },
    }

    report = camera_probe.build_camera_probe_report(observation)

    assert report["inferred_camera_names"] == ["base_camera", "hand_camera"]
    assert report["usable_named_views"] == ["base_camera", "hand_camera"]
    probes_by_name = {probe["display_name"]: probe for probe in report["probes"]}
    assert probes_by_name["default"]["usable_rgbd"] is True
    assert probes_by_name["base_camera"]["source_matches_camera_name"] is True
    assert probes_by_name["hand_camera"]["intrinsic_present"] is True


def test_requested_camera_reports_source_mismatch_when_name_is_absent() -> None:
    observation = {
        "sensor_data": {
            "base_camera": {
                "rgb": np.zeros((4, 5, 3), dtype=np.uint8),
                "depth": np.ones((4, 5, 1), dtype=np.float32),
            }
        }
    }

    report = camera_probe.build_camera_probe_report(observation, requested_camera_names=["missing_camera"])
    missing_probe = next(probe for probe in report["probes"] if probe["camera_name"] == "missing_camera")

    assert missing_probe["usable_rgbd"] is True
    assert missing_probe["source_matches_camera_name"] is False
    assert report["usable_named_views"] == ["base_camera"]
