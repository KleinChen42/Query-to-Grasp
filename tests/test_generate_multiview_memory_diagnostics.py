from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_multiview_memory_diagnostics import (
    build_diagnostics,
    build_conclusion,
    extract_candidate_observations,
    render_markdown,
    simulate_merge_count,
)


def test_extract_candidate_observations_reads_world_xyz_and_labels() -> None:
    memory_state = {
        "views": [
            {
                "view_id": "front",
                "reranked_candidates": [{"phrase": "red cube", "det_score": 0.9, "fused_2d_score": 0.9}],
                "candidates_3d": [
                    {
                        "world_xyz": [0.0, 0.0, 0.1],
                        "num_points": 12,
                        "depth_valid_ratio": 0.5,
                    }
                ],
            }
        ]
    }

    observations = extract_candidate_observations(memory_state)

    assert observations == [
        {
            "view_id": "front",
            "rank": 0,
            "label": "red cube",
            "world_xyz": [0.0, 0.0, 0.1],
            "num_points": 12,
            "depth_valid_ratio": 0.5,
            "det_score": 0.9,
            "fused_2d_score": 0.9,
        }
    ]


def test_simulate_merge_count_matches_greedy_distance_rule() -> None:
    observations = [
        {"world_xyz": [0.0, 0.0, 0.0]},
        {"world_xyz": [0.03, 0.0, 0.0]},
        {"world_xyz": [0.20, 0.0, 0.0]},
    ]

    assert simulate_merge_count(observations, merge_distance=0.05) == 2
    assert simulate_merge_count(observations, merge_distance=0.25) == 1


def test_build_diagnostics_and_markdown(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    run_dir = benchmark_dir / "runs" / "run_0001_seed_0" / "fake"
    run_dir.mkdir(parents=True)
    _write_json(
        benchmark_dir / "benchmark_summary.json",
        {
            "detector_backend": "hf",
            "skip_clip": True,
            "view_preset": "tabletop_3",
            "aggregate_metrics": {
                "mean_num_views": 3.0,
                "mean_num_memory_objects": 3.0,
            },
        },
    )
    _write_json(
        benchmark_dir / "benchmark_rows.json",
        [
            {
                "query": "red cube",
                "seed": 0,
                "num_views": 3,
                "num_observations_added": 3,
                "num_memory_objects": 3,
                "selected_object_id": "obj_0000",
                "selected_overall_confidence": 0.5,
                "artifacts": str(run_dir),
            }
        ],
    )
    _write_json(
        run_dir / "memory_state.json",
        {
            "views": [
                _view("front", [0.0, 0.0, 0.0]),
                _view("left", [0.11, 0.0, 0.0]),
                _view("right", [0.22, 0.0, 0.0]),
            ]
        },
    )

    diagnostics = build_diagnostics(benchmark_dir, merge_distances=[0.08, 0.24])
    markdown = render_markdown(diagnostics)

    assert diagnostics["aggregate"]["total_runs"] == 1
    assert diagnostics["aggregate"]["mean_num_candidate_observations"] == 3.0
    assert diagnostics["aggregate"]["mean_simulated_memory_objects_by_merge_distance"]["0.08"] == 3.0
    assert diagnostics["aggregate"]["mean_simulated_memory_objects_by_merge_distance"]["0.24"] == 1.0
    assert "Multi-view capture is working" in diagnostics["aggregate"]["conclusion"]
    assert "# Multi-View Memory Diagnostics" in markdown
    assert "Merge-Distance Sweep" in markdown


def test_build_conclusion_flags_cross_view_3d_inconsistency() -> None:
    conclusion = build_conclusion(
        {
            "mean_same_label_pairwise_distance": 1.0,
            "mean_simulated_memory_objects_by_merge_distance": {
                "0.08": 3.5,
                "0.32": 3.0,
            },
        },
        {
            "aggregate_metrics": {
                "mean_num_views": 3.0,
                "mean_num_memory_objects": 3.5,
            }
        },
    )

    assert "same-label 3D target estimates are far apart" in conclusion
    assert "not only a conservative merge threshold" in conclusion


def _view(view_id: str, world_xyz: list[float]) -> dict:
    return {
        "view_id": view_id,
        "reranked_candidates": [{"phrase": "red cube", "det_score": 0.9, "fused_2d_score": 0.9}],
        "candidates_3d": [{"world_xyz": world_xyz, "num_points": 10, "depth_valid_ratio": 0.5}],
    }


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
