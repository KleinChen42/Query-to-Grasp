from __future__ import annotations

import numpy as np

from scripts.compare_extrinsic_conventions import (
    CV_TO_GL,
    aggregate_runs,
    build_conclusion,
    build_extrinsic_conventions,
    metrics_by_convention,
)


def test_build_extrinsic_conventions_includes_cv_gl_and_inverse() -> None:
    observation = {
        "sensor_param": {
            "base_camera": {
                "cam2world_gl": np.eye(4, dtype=np.float32),
                "extrinsic_cv": np.array(
                    [
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 2.0],
                        [0.0, 0.0, 1.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            }
        }
    }

    conventions = build_extrinsic_conventions(observation, camera_name="base_camera")
    by_name = {convention.name: convention for convention in conventions}

    assert set(by_name) == {
        "cam2world_gl_direct",
        "cam2world_gl_cv_to_gl",
        "extrinsic_cv_direct",
        "extrinsic_cv_inverse",
    }
    np.testing.assert_allclose(by_name["cam2world_gl_cv_to_gl"].matrix, CV_TO_GL)
    np.testing.assert_allclose(
        by_name["extrinsic_cv_inverse"].matrix[:3, 3],
        np.array([-1.0, -2.0, -3.0], dtype=np.float32),
    )


def test_metrics_by_convention_prefers_tighter_same_label_points() -> None:
    rows = [
        _row("bad", "front", [0.0, 0.0, 0.0]),
        _row("bad", "left", [1.0, 0.0, 0.0]),
        _row("good", "front", [0.0, 0.0, 0.0]),
        _row("good", "left", [0.1, 0.0, 0.0]),
    ]

    metrics = metrics_by_convention(rows)
    aggregate = aggregate_runs([{"convention_metrics": metrics}])
    conclusion = build_conclusion(aggregate)

    assert metrics["bad"]["mean_same_label_pairwise_distance"] == 1.0
    assert metrics["good"]["mean_same_label_pairwise_distance"] == 0.1
    assert aggregate["best_convention_by_same_label_distance"] == "good"
    assert "`good` gives the lowest" in conclusion


def _row(convention: str, view_id: str, world_xyz: list[float]) -> dict:
    return {
        "view_id": view_id,
        "convention": convention,
        "source_key": "sensor_param.base_camera.mock",
        "rank": 0,
        "phrase": "red cube",
        "world_xyz": world_xyz,
    }
