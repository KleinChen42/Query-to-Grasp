from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.generate_stackcube_grasp_failure_report import (
    build_failure_report,
    classify_failure,
    render_markdown,
)


def test_classify_failure_prioritizes_success_and_third_object() -> None:
    assert classify_failure({"pick_success": "true"}) == "success"
    assert classify_failure({"run_failed": "true"}) == "run_failed"
    assert classify_failure({"has_selected_object": "false"}) == "selected_object_missing"
    assert classify_failure({"has_selected_object": "true", "grasp_attempted": "false"}) == "pick_not_attempted"
    assert (
        classify_failure(
            {
                "has_selected_object": "true",
                "grasp_attempted": "true",
                "closed_loop_extra_view_third_object_involved": "true",
            }
        )
        == "third_object_absorption"
    )


def test_classify_failure_detects_fused_grasp_outliers_and_controller_failures() -> None:
    outlier = {
        "has_selected_object": "true",
        "grasp_attempted": "true",
        "selected_semantic_to_grasp_xy_distance": "0.01",
        "selected_grasp_observation_xy_spread": "0.12",
        "selected_grasp_observation_max_distance_to_fused": "0.02",
    }
    contact = {
        "has_selected_object": "true",
        "grasp_attempted": "true",
        "selected_semantic_to_grasp_xy_distance": "0.01",
        "selected_grasp_observation_xy_spread": "0.01",
        "selected_grasp_observation_max_distance_to_fused": "0.02",
    }

    assert classify_failure(outlier) == "wrong_fused_grasp_observation"
    assert classify_failure({**contact, "final_should_reobserve": "true"}) == "memory_fragmentation_or_low_support"
    assert classify_failure(contact) == "controller_contact_failure"


def test_build_failure_report_summarizes_benchmarks_and_rows(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "stackcube_tabletop"
    _write_summary(benchmark_dir)
    _write_rows(benchmark_dir)

    report = build_failure_report([f"tabletop={benchmark_dir}"])
    markdown = render_markdown(report)

    assert report["benchmarks"][0]["label"] == "tabletop"
    assert report["benchmarks"][0]["total_runs"] == 3
    assert report["benchmarks"][0]["pick_success_rate"] == 1 / 3
    assert report["failure_class_counts"] == {
        "third_object_absorption": 1,
        "wrong_fused_grasp_observation": 1,
    }
    assert "Dominant failure class" in report["conclusion"]
    assert "StackCube Multi-View Grasp Failure Diagnosis" in markdown
    assert "wrong_fused_grasp_observation" in markdown
    assert "third_object_absorption" in markdown


def test_build_failure_report_missing_rows_can_be_skipped(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        build_failure_report([f"missing={missing}"])

    with pytest.raises(ValueError, match="No benchmark summaries"):
        build_failure_report([f"missing={missing}"], skip_missing=True)


def _write_summary(benchmark_dir: Path) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "total_runs": 3,
        "env_id": "StackCube-v1",
        "view_preset": "tabletop_3",
        "closed_loop_reobserve_enabled": False,
        "aggregate_metrics": {
            "total_runs": 3,
            "failed_runs": 0,
            "pick_success_rate": 1 / 3,
            "mean_selected_semantic_to_grasp_xy_distance": 0.04,
            "mean_selected_grasp_observation_xy_spread": 0.05,
            "mean_selected_grasp_observation_max_distance_to_fused": 0.06,
        },
    }
    (benchmark_dir / "benchmark_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _write_rows(benchmark_dir: Path) -> None:
    rows = [
        {
            "query": "red cube",
            "seed": "0",
            "pick_success": "true",
            "run_failed": "false",
            "has_selected_object": "true",
            "grasp_attempted": "true",
            "pick_stage": "success",
            "pick_target_source": "memory_grasp_world_xyz",
        },
        {
            "query": "red cube",
            "seed": "1",
            "pick_success": "false",
            "run_failed": "false",
            "has_selected_object": "true",
            "grasp_attempted": "true",
            "pick_stage": "grasp_not_confirmed",
            "pick_target_source": "memory_grasp_world_xyz",
            "selected_semantic_to_grasp_xy_distance": "0.02",
            "selected_grasp_observation_count": "3",
            "selected_grasp_observation_xy_spread": "0.10",
            "selected_grasp_observation_max_distance_to_fused": "0.04",
            "closed_loop_extra_view_third_object_involved": "false",
            "final_reobserve_reason": "confident_enough",
            "artifacts": "run_1",
        },
        {
            "query": "red cube",
            "seed": "2",
            "pick_success": "false",
            "run_failed": "false",
            "has_selected_object": "true",
            "grasp_attempted": "true",
            "pick_stage": "grasp_not_confirmed",
            "pick_target_source": "memory_grasp_world_xyz",
            "selected_semantic_to_grasp_xy_distance": "0.01",
            "selected_grasp_observation_count": "2",
            "selected_grasp_observation_xy_spread": "0.02",
            "selected_grasp_observation_max_distance_to_fused": "0.02",
            "closed_loop_extra_view_third_object_involved": "true",
            "final_reobserve_reason": "ambiguous_top_candidates",
            "artifacts": "run_2",
        },
    ]
    with (benchmark_dir / "benchmark_rows.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
