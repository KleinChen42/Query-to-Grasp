from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.generate_fusion_comparison_table import (
    build_table_rows,
    fusion_row_from_benchmark,
    render_markdown_table,
    single_view_row_from_benchmark,
    write_rows_csv,
)


def test_build_table_rows_combines_single_view_and_fusion(tmp_path: Path) -> None:
    single_dir = tmp_path / "single"
    fusion_dir = tmp_path / "fusion"
    _write_single_summary(single_dir)
    _write_fusion_summary(fusion_dir)

    rows = build_table_rows(
        single_view_specs=[f"HF single={single_dir}"],
        fusion_specs=[f"HF fusion={fusion_dir}"],
    )

    assert [row["label"] for row in rows] == ["HF single", "HF fusion"]
    assert rows[0]["benchmark_type"] == "single_view_pick"
    assert rows[0]["primary_rate_name"] == "fraction_with_3d_target"
    assert rows[0]["primary_rate"] == 1.0
    assert rows[1]["benchmark_type"] == "fusion_debug"
    assert rows[1]["primary_rate_name"] == "fraction_with_selected_object"
    assert rows[1]["primary_rate"] == 0.5


def test_single_view_row_maps_detection_metrics(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "single"
    _write_single_summary(benchmark_dir)

    row = single_view_row_from_benchmark("single", benchmark_dir)

    assert row["mean_raw_num_detections"] == 1.25
    assert row["env_id"] == "PickCube-v1"
    assert row["obs_mode"] == "rgbd"
    assert row["pick_executor"] == "sim_topdown"
    assert row["grasp_target_mode"] == "refined"
    assert row["mean_num_ranked_candidates"] == 1.0
    assert row["mean_num_views"] == 1.0
    assert row["mean_num_memory_objects"] == "n/a"
    assert row["grasp_attempted_rate"] == "n/a"
    assert row["pick_success_rate"] == 0.0
    assert row["task_success_rate"] == "n/a"
    assert row["pick_stage_counts"] == "n/a"


def test_fusion_row_maps_memory_metrics(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "fusion"
    _write_fusion_summary(benchmark_dir)

    row = fusion_row_from_benchmark("fusion", benchmark_dir)

    assert row["mean_raw_num_detections"] == "n/a"
    assert row["env_id"] == "PickCube-v1"
    assert row["obs_mode"] == "rgbd"
    assert row["pick_executor"] == "sim_topdown"
    assert row["grasp_target_mode"] == "refined"
    assert row["mean_num_views"] == 2.0
    assert row["mean_num_memory_objects"] == 3.0
    assert row["mean_num_observations_added"] == 4.0
    assert row["mean_same_label_pairwise_distance"] == 0.12
    assert row["mean_selected_overall_confidence"] == 0.75
    assert row["reobserve_trigger_rate"] == 0.5
    assert row["initial_reobserve_trigger_rate"] == 0.75
    assert row["final_reobserve_trigger_rate"] == 0.25
    assert row["closed_loop_execution_rate"] == 0.5
    assert row["closed_loop_resolution_rate"] == 0.25
    assert row["closed_loop_still_needed_rate"] == 0.25
    assert row["mean_closed_loop_delta_selected_overall_confidence"] == 0.1
    assert row["mean_closed_loop_delta_selected_num_views"] == 0.5
    assert row["mean_closed_loop_delta_num_memory_objects"] == 0.0
    assert row["reobserve_reason_counts"] == "ambiguous_top_candidates: 1; confident_enough: 1"
    assert row["grasp_attempted_rate"] == 1.0
    assert row["pick_success_rate"] == 0.5
    assert row["task_success_rate"] == 0.25
    assert row["pick_stage_counts"] == "grasp_not_confirmed: 1; success: 1"


def test_build_table_rows_missing_behavior(tmp_path: Path) -> None:
    existing = tmp_path / "single"
    missing = tmp_path / "missing"
    _write_single_summary(existing)

    with pytest.raises(FileNotFoundError, match="Missing benchmark_summary"):
        build_table_rows([f"single={existing}"], [f"missing={missing}"])

    rows = build_table_rows([f"single={existing}"], [f"missing={missing}"], skip_missing=True)

    assert len(rows) == 1
    assert rows[0]["label"] == "single"


def test_build_table_rows_requires_at_least_one_summary(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Provide at least one"):
        build_table_rows([], [])

    with pytest.raises(ValueError, match="No benchmark summaries"):
        build_table_rows([], [f"missing={tmp_path / 'missing'}"], skip_missing=True)


def test_render_markdown_table_and_csv(tmp_path: Path) -> None:
    rows = [
        {
            "label": "HF fusion",
            "benchmark_type": "fusion_debug",
            "env_id": "PickCube-v1",
            "obs_mode": "rgbd",
            "pick_executor": "sim_topdown",
            "grasp_target_mode": "refined",
            "detector_backend": "hf",
            "skip_clip": "True",
            "total_runs": 2,
            "primary_rate_name": "fraction_with_selected_object",
            "primary_rate": 1.0,
            "mean_runtime_seconds": 10.5,
            "mean_raw_num_detections": "n/a",
            "mean_num_ranked_candidates": "n/a",
            "mean_num_views": 1.0,
            "mean_num_memory_objects": 1.0,
            "mean_num_observations_added": 1.0,
            "mean_same_label_pairwise_distance": 0.12,
            "mean_selected_overall_confidence": 0.5,
            "reobserve_trigger_rate": 0.25,
            "initial_reobserve_trigger_rate": 0.5,
            "final_reobserve_trigger_rate": 0.25,
            "closed_loop_execution_rate": 0.25,
            "closed_loop_resolution_rate": 0.25,
            "closed_loop_still_needed_rate": 0.0,
            "mean_closed_loop_delta_selected_overall_confidence": 0.1,
            "mean_closed_loop_delta_selected_num_views": 0.25,
            "mean_closed_loop_delta_num_memory_objects": 0.0,
            "reobserve_reason_counts": "ambiguous_top_candidates: 1; none: 3",
            "grasp_attempted_rate": 1.0,
            "pick_success_rate": 0.5,
            "task_success_rate": 0.25,
            "pick_stage_counts": "success: 1; grasp_not_confirmed: 1",
        }
    ]

    markdown = render_markdown_table(rows)
    csv_path = tmp_path / "comparison.csv"
    write_rows_csv(rows, csv_path)

    assert "# Single-View vs Fusion Comparison Table" in markdown
    assert "HF fusion" in markdown
    assert "env_id" in markdown
    assert "pick_executor" in markdown
    assert "sim_topdown" in markdown
    assert "0.5000" in markdown
    assert "mean_same_label_pairwise_distance" in markdown
    assert "reobserve_trigger_rate" in markdown
    assert "initial_reobserve_trigger_rate" in markdown
    assert "closed_loop_execution_rate" in markdown
    assert "closed_loop_resolution_rate" in markdown
    assert "mean_closed_loop_delta_selected_num_views" in markdown
    assert "grasp_attempted_rate" in markdown
    assert "pick_stage_counts" in markdown
    assert "ambiguous_top_candidates: 1" in markdown
    assert "grasp_not_confirmed: 1" in markdown
    csv_rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert csv_rows[0]["label"] == "HF fusion"
    assert csv_rows[0]["mean_same_label_pairwise_distance"] == "0.12"
    assert csv_rows[0]["mean_selected_overall_confidence"] == "0.5"
    assert csv_rows[0]["reobserve_trigger_rate"] == "0.25"
    assert csv_rows[0]["grasp_attempted_rate"] == "1.0"
    assert csv_rows[0]["pick_success_rate"] == "0.5"


def _write_single_summary(benchmark_dir: Path) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_runs": 4,
        "env_id": "PickCube-v1",
        "obs_mode": "rgbd",
        "pick_executor": "sim_topdown",
        "grasp_target_mode": "refined",
        "detector_backend": "hf",
        "skip_clip": True,
        "aggregate_metrics": {
            "total_runs": 4,
            "mean_raw_num_detections": 1.25,
            "mean_num_ranked_candidates": 1.0,
            "fraction_with_3d_target": 1.0,
            "pick_success_rate": 0.0,
            "mean_runtime_seconds": 12.5,
        },
    }
    (benchmark_dir / "benchmark_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _write_fusion_summary(benchmark_dir: Path) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_runs": 2,
        "env_id": "PickCube-v1",
        "obs_mode": "rgbd",
        "pick_executor": "sim_topdown",
        "grasp_target_mode": "refined",
        "detector_backend": "hf",
        "skip_clip": True,
        "aggregate_metrics": {
            "total_runs": 2,
            "mean_num_views": 2.0,
            "mean_num_memory_objects": 3.0,
            "mean_num_observations_added": 4.0,
            "fraction_with_selected_object": 0.5,
            "mean_selected_overall_confidence": 0.75,
            "reobserve_trigger_rate": 0.5,
            "initial_reobserve_trigger_rate": 0.75,
            "final_reobserve_trigger_rate": 0.25,
            "closed_loop_execution_rate": 0.5,
            "closed_loop_resolution_rate": 0.25,
            "closed_loop_still_needed_rate": 0.25,
            "mean_closed_loop_delta_selected_overall_confidence": 0.1,
            "mean_closed_loop_delta_selected_num_views": 0.5,
            "mean_closed_loop_delta_num_memory_objects": 0.0,
            "reobserve_reason_counts": {
                "ambiguous_top_candidates": 1,
                "confident_enough": 1,
            },
            "grasp_attempted_rate": 1.0,
            "pick_success_rate": 0.5,
            "task_success_rate": 0.25,
            "pick_stage_counts": {
                "success": 1,
                "grasp_not_confirmed": 1,
            },
            "mean_runtime_seconds": 20.0,
        },
    }
    (benchmark_dir / "benchmark_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    diagnostics = {
        "aggregate": {
            "mean_same_label_pairwise_distance": 0.12,
        }
    }
    (benchmark_dir / "memory_diagnostics.json").write_text(json.dumps(diagnostics), encoding="utf-8")
