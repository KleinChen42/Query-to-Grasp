from __future__ import annotations

from src.eval.metrics import aggregate_runs, aggregate_runs_by_query, summarize_run


def test_summarize_run_with_complete_input() -> None:
    summary = {
        "query": "red cube",
        "num_detections": 2,
        "num_ranked_candidates": 2,
        "camera_xyz": [0.1, 0.2, 0.3],
        "world_xyz": None,
        "num_3d_points": 42,
        "pick_success": False,
        "pick_stage": "placeholder_not_executed",
        "runtime_seconds": 1.25,
        "artifacts": "outputs/run",
    }
    pick_result = {"success": False, "stage": "placeholder_not_executed", "target_xyz": [0.1, 0.2, 0.3]}

    row = summarize_run(summary, pick_result)

    assert row["query"] == "red cube"
    assert row["num_detections"] == 2
    assert row["num_ranked_candidates"] == 2
    assert row["has_3d_target"] is True
    assert row["num_3d_points"] == 42
    assert row["pick_success"] is False
    assert row["pick_stage"] == "placeholder_not_executed"
    assert row["runtime_seconds"] == 1.25
    assert row["artifacts"] == "outputs/run"


def test_summarize_run_with_missing_fields() -> None:
    row = summarize_run({}, {})

    assert row["query"] == ""
    assert row["num_detections"] == 0
    assert row["num_ranked_candidates"] == 0
    assert row["has_3d_target"] is False
    assert row["num_3d_points"] == 0
    assert row["pick_success"] is False
    assert row["pick_stage"] == "unknown"
    assert row["runtime_seconds"] == 0.0
    assert row["artifacts"] == ""


def test_aggregate_runs_computes_means_and_rates() -> None:
    rows = [
        {
            "num_detections": 2,
            "num_ranked_candidates": 2,
            "has_3d_target": True,
            "num_3d_points": 10,
            "pick_success": False,
            "pick_stage": "placeholder_not_executed",
            "runtime_seconds": 1.0,
        },
        {
            "num_detections": 0,
            "num_ranked_candidates": 0,
            "has_3d_target": False,
            "num_3d_points": 0,
            "pick_success": False,
            "pick_stage": "run_failed",
            "runtime_seconds": 0.0,
        },
        {
            "num_detections": 1,
            "num_ranked_candidates": 1,
            "has_3d_target": True,
            "num_3d_points": 20,
            "pick_success": True,
            "pick_stage": "success",
            "runtime_seconds": 2.0,
        },
    ]

    metrics = aggregate_runs(rows)

    assert metrics["total_runs"] == 3
    assert metrics["mean_num_detections"] == 1.0
    assert metrics["mean_num_ranked_candidates"] == 1.0
    assert metrics["mean_num_3d_points"] == 10.0
    assert metrics["fraction_with_3d_target"] == 2 / 3
    assert metrics["pick_success_rate"] == 1 / 3
    assert metrics["mean_runtime_seconds"] == 1.0
    assert metrics["pick_stage_counts"] == {
        "placeholder_not_executed": 1,
        "run_failed": 1,
        "success": 1,
    }


def test_aggregate_runs_defaults_missing_runtime_to_zero() -> None:
    metrics = aggregate_runs(
        [
            {
                "num_detections": 1,
                "num_ranked_candidates": 1,
                "has_3d_target": True,
                "num_3d_points": 8,
                "pick_success": False,
                "pick_stage": "placeholder_not_executed",
            }
        ]
    )

    assert metrics["mean_runtime_seconds"] == 0.0


def test_aggregate_runs_by_query_computes_per_query_metrics() -> None:
    rows = [
        {
            "query": "red cube",
            "num_detections": 2,
            "num_ranked_candidates": 2,
            "has_3d_target": True,
            "num_3d_points": 10,
            "pick_success": False,
            "pick_stage": "placeholder_not_executed",
            "runtime_seconds": 1.0,
        },
        {
            "query": "red cube",
            "num_detections": 4,
            "num_ranked_candidates": 3,
            "has_3d_target": True,
            "num_3d_points": 30,
            "pick_success": True,
            "pick_stage": "success",
            "runtime_seconds": 3.0,
        },
        {
            "query": "blue mug",
            "num_detections": 1,
            "num_ranked_candidates": 1,
            "has_3d_target": False,
            "num_3d_points": 0,
            "pick_success": False,
            "pick_stage": "run_failed",
            "runtime_seconds": 0.5,
        },
    ]

    per_query = aggregate_runs_by_query(rows)

    assert sorted(per_query) == ["blue mug", "red cube"]
    assert per_query["red cube"]["total_runs"] == 2
    assert per_query["red cube"]["mean_num_detections"] == 3.0
    assert per_query["red cube"]["mean_num_3d_points"] == 20.0
    assert per_query["red cube"]["pick_success_rate"] == 0.5
    assert per_query["red cube"]["mean_runtime_seconds"] == 2.0
    assert per_query["blue mug"]["fraction_with_3d_target"] == 0.0
