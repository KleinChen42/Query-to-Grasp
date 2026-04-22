from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_cross_view_geometry_report import (
    build_conclusion,
    build_report,
    extract_geometry_candidates,
    render_markdown,
    transform_point,
)


def test_transform_point_applies_homogeneous_matrix() -> None:
    matrix = [
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, -0.25],
        [0.0, 0.0, 0.0, 1.0],
    ]

    assert transform_point(matrix, [0.1, 0.2, 0.3]) == pytest.approx([0.6, 1.2, 0.05])


def test_transform_point_uses_cam2world_gl_camera_conversion() -> None:
    matrix = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    assert transform_point(matrix, [0.0, 0.0, 2.0], "sensor_param.base_camera.cam2world_gl") == [0.0, 0.0, -2.0]


def test_extract_geometry_candidates_reads_metadata_and_recompute_error(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path / "run", world_points=[[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]])
    memory_state = _load_json(run_dir / "memory_state.json")

    candidates = extract_geometry_candidates(memory_state, run_dir)

    assert len(candidates) == 2
    assert candidates[0]["extrinsic_key"] == "sensor_param.base_camera.cam2world_gl"
    assert candidates[0]["camera_position"] == [0.0, 0.0, 0.0]
    assert candidates[0]["world_recompute_error"] == pytest.approx(0.0, abs=1e-6)
    assert candidates[0]["box_xyxy"] == [1.0, 2.0, 3.0, 4.0]


def test_build_report_from_benchmark_and_render_markdown(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    run_dir = _write_run(benchmark_dir / "runs" / "run_0001_seed_0" / "fake")
    _write_json(
        benchmark_dir / "benchmark_rows.json",
        [
            {
                "query": "red cube",
                "seed": 0,
                "artifacts": str(run_dir),
            }
        ],
    )

    report = build_report(benchmark_dir=benchmark_dir)
    markdown = render_markdown(report)

    assert report["aggregate"]["total_runs"] == 1
    assert report["aggregate"]["mean_world_recompute_error"] == pytest.approx(0.0, abs=1e-6)
    assert report["aggregate"]["mean_same_label_pairwise_distance"] > 0.5
    assert any("cam2world_gl" in key for key in report["aggregate"]["extrinsic_source_counts"])
    assert "compare OpenGL cam2world against extrinsic_cv" in report["aggregate"]["conclusion"]
    assert "# Cross-View Geometry Sanity Report" in markdown
    assert "box_xyxy" in markdown


def test_build_conclusion_flags_transform_mismatch() -> None:
    conclusion = build_conclusion(
        {
            "mean_same_label_pairwise_distance": 0.0,
            "mean_world_recompute_error": 0.1,
            "extrinsic_source_counts": {},
        }
    )

    assert "do not match extrinsic * camera_xyz" in conclusion


def _write_run(run_dir: Path, world_points: list[list[float]] | None = None) -> Path:
    world_points = world_points or [[0.1, 0.0, 0.0], [1.0, 0.0, 0.0]]
    for view_id, world_xyz in zip(["front", "left"], world_points):
        view_dir = run_dir / "views" / view_id
        _write_json(
            view_dir / "observation" / "metadata.json",
            {
                "camera_info": {
                    "intrinsic": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "intrinsic_key": "sensor_param.base_camera.intrinsic_cv",
                    "extrinsic": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    "extrinsic_key": "sensor_param.base_camera.cam2world_gl",
                }
            },
        )

    _write_json(
        run_dir / "memory_state.json",
        {
            "query": {"raw_query": "red cube"},
            "selected_object_id": "obj_0000",
            "selection_label": "red cube",
            "views": [
                _view("front", world_points[0]),
                _view("left", world_points[1]),
            ],
        },
    )
    return run_dir


def _view(view_id: str, world_xyz: list[float]) -> dict:
    return {
        "view_id": view_id,
        "artifacts": str(Path("views") / view_id),
        "reranked_candidates": [
            {
                "phrase": "red cube",
                "det_score": 0.9,
                "clip_score": 0.0,
                "fused_2d_score": 0.9,
            }
        ],
        "candidates_3d": [
            {
                "box_xyxy": [1.0, 2.0, 3.0, 4.0],
                "camera_xyz": world_xyz,
                "world_xyz": world_xyz,
                "num_points": 10,
                "depth_valid_ratio": 0.5,
            }
        ],
    }


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
