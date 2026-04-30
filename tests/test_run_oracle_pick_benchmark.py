from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_oracle_pick_benchmark import (
    aggregate_oracle_rows,
    build_child_summary,
    failed_row,
    row_from_summary,
    write_rows_csv,
)


def test_oracle_child_summary_and_row_are_serializable(tmp_path: Path) -> None:
    args = argparse.Namespace(env_id="PickCube-v1", obs_mode="rgbd", control_mode="pd_ee_delta_pos")
    pick_result = {
        "grasp_attempted": True,
        "pick_success": True,
        "task_success": False,
        "is_grasped": True,
        "stage": "success",
    }

    summary = build_child_summary(
        args=args,
        seed=3,
        run_dir=tmp_path,
        target_xyz=np.asarray([0.1, 0.2, 0.03], dtype=np.float32),
        pick_result=pick_result,
        runtime_seconds=1.5,
    )
    row = row_from_summary(summary)

    json.dumps(summary)
    json.dumps(row)
    assert summary["detector_backend"] == "oracle"
    assert summary["grasp_target_mode"] == "oracle"
    assert summary["pick_target_source"] == "oracle_object_pose"
    assert row["grasp_attempted"] is True
    assert row["pick_success"] is True
    assert row["task_success"] is False
    assert row["pick_stage"] == "success"
    assert row["run_failed"] is False


def test_oracle_aggregate_counts_failures_and_target_sources() -> None:
    rows = [
        {
            "has_3d_target": True,
            "grasp_attempted": True,
            "pick_success": True,
            "task_success": False,
            "is_grasped": True,
            "pick_stage": "success",
            "pick_target_source": "oracle_object_pose",
            "runtime_seconds": 1.0,
            "run_failed": False,
        },
        {
            "has_3d_target": True,
            "grasp_attempted": True,
            "pick_success": False,
            "task_success": False,
            "is_grasped": False,
            "pick_stage": "grasp_not_confirmed",
            "pick_target_source": "oracle_object_pose",
            "runtime_seconds": 2.0,
            "run_failed": False,
        },
        failed_row(seed=5, message="missing oracle pose", artifacts=Path("outputs/run"), runtime_seconds=0.5),
    ]

    metrics = aggregate_oracle_rows(rows)

    assert metrics["total_runs"] == 3
    assert metrics["failed_runs"] == 1
    assert metrics["grasp_attempted_rate"] == 2 / 3
    assert metrics["pick_success_rate"] == 1 / 3
    assert metrics["task_success_rate"] == 0.0
    assert metrics["pick_stage_counts"] == {
        "grasp_not_confirmed": 1,
        "run_failed": 1,
        "success": 1,
    }
    assert metrics["pick_target_source_counts"] == {"oracle_object_pose": 3}


def test_write_oracle_rows_csv_includes_target_source(tmp_path: Path) -> None:
    csv_path = tmp_path / "rows.csv"
    row = failed_row(seed=1, message="no pose", artifacts=tmp_path)

    write_rows_csv([row], csv_path)

    text = csv_path.read_text(encoding="utf-8")
    assert "pick_target_source" in text
    assert "oracle_object_pose" in text
