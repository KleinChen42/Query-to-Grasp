from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.run_multiview_fusion_benchmark as benchmark


def test_build_child_command_explicitly_forwards_clip_mode(tmp_path: Path) -> None:
    args = _args(output_dir=tmp_path, skip_clip=False, view_ids=["front", "left"], view_preset="none")

    command = benchmark.build_child_command(args=args, query="red cube", seed=3, output_dir=tmp_path / "child")

    assert "run_multiview_fusion_debug.py" in command[1]
    assert "--query" in command
    assert command[command.index("--query") + 1] == "red cube"
    assert "--seed" in command
    assert command[command.index("--seed") + 1] == "3"
    assert "--view-ids" in command
    assert "front" in command
    assert "left" in command
    assert "--skip-clip" not in command
    assert "--use-clip" in command

    args.skip_clip = True
    command = benchmark.build_child_command(args=args, query="red cube", seed=3, output_dir=tmp_path / "child")

    assert "--skip-clip" in command
    assert "--use-clip" not in command


def test_build_child_command_forwards_view_preset(tmp_path: Path) -> None:
    args = _args(output_dir=tmp_path, skip_clip=True, view_preset="tabletop_3")

    command = benchmark.build_child_command(args=args, query="red cube", seed=3, output_dir=tmp_path / "child")

    assert "--view-preset" in command
    assert command[command.index("--view-preset") + 1] == "tabletop_3"


def test_build_child_command_forwards_closed_loop_reobserve(tmp_path: Path) -> None:
    args = _args(output_dir=tmp_path, skip_clip=True, view_preset="tabletop_3")
    args.enable_closed_loop_reobserve = True
    args.closed_loop_max_extra_views = 2

    command = benchmark.build_child_command(args=args, query="object", seed=0, output_dir=tmp_path / "child")

    assert "--enable-closed-loop-reobserve" in command
    assert "--closed-loop-max-extra-views" in command
    assert command[command.index("--closed-loop-max-extra-views") + 1] == "2"


def test_build_child_command_forwards_selected_object_continuity(tmp_path: Path) -> None:
    args = _args(output_dir=tmp_path, skip_clip=True, view_preset="tabletop_3")
    args.enable_selected_object_continuity = True
    args.selected_object_continuity_distance_scale = 1.5

    command = benchmark.build_child_command(args=args, query="object", seed=0, output_dir=tmp_path / "child")

    assert "--enable-selected-object-continuity" in command
    assert "--selected-object-continuity-distance-scale" in command
    assert command[command.index("--selected-object-continuity-distance-scale") + 1] == "1.5"


def test_summarize_fusion_run_defaults_missing_fields() -> None:
    row = benchmark.summarize_fusion_run(
        {
            "query": "red cube",
            "num_views": 2,
            "num_memory_objects": 1,
            "num_observations_added": 2,
            "selected_object_id": "obj_0000",
            "selected_overall_confidence": 0.75,
            "should_reobserve": True,
            "reobserve_reason": "ambiguous_top_candidates",
            "skip_clip": "true",
        }
    )

    assert row["query"] == "red cube"
    assert row["has_selected_object"] is True
    assert row["selected_overall_confidence"] == 0.75
    assert row["should_reobserve"] is True
    assert row["reobserve_reason"] == "ambiguous_top_candidates"
    assert row["runtime_seconds"] == 0.0
    assert row["skip_clip"] is True
    assert row["view_preset"] == "none"
    assert row["closed_loop_delta_selected_overall_confidence"] == 0.0
    assert row["closed_loop_reobserve_resolved"] is False
    assert row["closed_loop_before_selected_received_observation"] is False
    assert row["closed_loop_before_selected_delta_num_views"] == 0
    assert row["closed_loop_final_selected_absorbed_extra_view"] is False
    assert row["closed_loop_extra_view_absorber_count"] == 0
    assert row["closed_loop_selected_object_continuity_enabled"] is False
    assert row["closed_loop_preferred_merge_count"] == 0
    assert row["closed_loop_preferred_merge_rate"] == 0.0


def test_aggregate_rows_by_query() -> None:
    rows = [
        {
            "query": "red cube",
            "num_views": 1,
            "num_memory_objects": 1,
            "num_observations_added": 1,
            "has_selected_object": True,
            "selected_overall_confidence": 0.6,
            "should_reobserve": True,
            "reobserve_reason": "low_overall_confidence",
            "initial_should_reobserve": True,
            "initial_reobserve_reason": "low_overall_confidence",
            "final_should_reobserve": False,
            "final_reobserve_reason": "confident_enough",
            "closed_loop_reobserve_executed": True,
            "closed_loop_reobserve_resolved": True,
            "closed_loop_reobserve_still_needed": False,
            "closed_loop_selected_object_changed": False,
            "closed_loop_reobserve_reason_changed": True,
            "closed_loop_before_selected_present_after": True,
            "closed_loop_before_selected_still_selected": True,
            "closed_loop_before_selected_received_observation": True,
            "closed_loop_before_selected_gained_view_support": True,
            "closed_loop_before_selected_delta_num_observations": 1,
            "closed_loop_before_selected_delta_num_views": 1,
            "closed_loop_final_selected_absorbed_extra_view": True,
            "closed_loop_extra_view_third_object_involved": False,
            "closed_loop_extra_view_absorber_count": 1,
            "closed_loop_selected_object_continuity_enabled": True,
            "closed_loop_preferred_merge_count": 2,
            "closed_loop_preferred_merge_rate": 1.0,
            "closed_loop_delta_num_views": 1,
            "closed_loop_delta_num_memory_objects": 0,
            "closed_loop_delta_num_observations_added": 1,
            "closed_loop_delta_selected_overall_confidence": 0.2,
            "closed_loop_delta_selected_num_views": 1,
            "closed_loop_delta_selected_num_observations": 1,
            "runtime_seconds": 10.0,
            "run_failed": False,
        },
        {
            "query": "red cube",
            "num_views": 1,
            "num_memory_objects": 0,
            "num_observations_added": 0,
            "has_selected_object": False,
            "selected_overall_confidence": 0.0,
            "should_reobserve": False,
            "reobserve_reason": None,
            "initial_should_reobserve": False,
            "initial_reobserve_reason": "confident_enough",
            "final_should_reobserve": False,
            "final_reobserve_reason": "confident_enough",
            "closed_loop_reobserve_executed": False,
            "closed_loop_reobserve_resolved": False,
            "closed_loop_reobserve_still_needed": False,
            "closed_loop_selected_object_changed": False,
            "closed_loop_reobserve_reason_changed": False,
            "closed_loop_before_selected_present_after": False,
            "closed_loop_before_selected_still_selected": False,
            "closed_loop_before_selected_received_observation": False,
            "closed_loop_before_selected_gained_view_support": False,
            "closed_loop_before_selected_delta_num_observations": 0,
            "closed_loop_before_selected_delta_num_views": 0,
            "closed_loop_final_selected_absorbed_extra_view": False,
            "closed_loop_extra_view_third_object_involved": False,
            "closed_loop_extra_view_absorber_count": 0,
            "closed_loop_selected_object_continuity_enabled": False,
            "closed_loop_preferred_merge_count": 0,
            "closed_loop_preferred_merge_rate": 0.0,
            "closed_loop_delta_num_views": 0,
            "closed_loop_delta_num_memory_objects": 0,
            "closed_loop_delta_num_observations_added": 0,
            "closed_loop_delta_selected_overall_confidence": 0.0,
            "closed_loop_delta_selected_num_views": 0,
            "closed_loop_delta_selected_num_observations": 0,
            "runtime_seconds": 4.0,
            "run_failed": True,
        },
        {
            "query": "blue mug",
            "num_views": 2,
            "num_memory_objects": 2,
            "num_observations_added": 3,
            "has_selected_object": True,
            "selected_overall_confidence": 0.8,
            "should_reobserve": True,
            "reobserve_reason": "ambiguous_top_candidates",
            "initial_should_reobserve": True,
            "initial_reobserve_reason": "ambiguous_top_candidates",
            "final_should_reobserve": True,
            "final_reobserve_reason": "ambiguous_top_candidates",
            "closed_loop_reobserve_executed": True,
            "closed_loop_reobserve_resolved": False,
            "closed_loop_reobserve_still_needed": True,
            "closed_loop_selected_object_changed": True,
            "closed_loop_reobserve_reason_changed": False,
            "closed_loop_before_selected_present_after": True,
            "closed_loop_before_selected_still_selected": False,
            "closed_loop_before_selected_received_observation": False,
            "closed_loop_before_selected_gained_view_support": False,
            "closed_loop_before_selected_delta_num_observations": 0,
            "closed_loop_before_selected_delta_num_views": 0,
            "closed_loop_final_selected_absorbed_extra_view": False,
            "closed_loop_extra_view_third_object_involved": True,
            "closed_loop_extra_view_absorber_count": 2,
            "closed_loop_selected_object_continuity_enabled": True,
            "closed_loop_preferred_merge_count": 1,
            "closed_loop_preferred_merge_rate": 0.5,
            "closed_loop_delta_num_views": 1,
            "closed_loop_delta_num_memory_objects": 1,
            "closed_loop_delta_num_observations_added": 2,
            "closed_loop_delta_selected_overall_confidence": -0.1,
            "closed_loop_delta_selected_num_views": 0,
            "closed_loop_delta_selected_num_observations": 0,
            "runtime_seconds": 12.0,
            "run_failed": False,
        },
    ]

    aggregate = benchmark.aggregate_rows(rows)
    per_query = benchmark.aggregate_rows_by_query(rows)

    assert aggregate["total_runs"] == 3
    assert aggregate["failed_runs"] == 1
    assert aggregate["fraction_with_selected_object"] == 2 / 3
    assert aggregate["reobserve_trigger_rate"] == 2 / 3
    assert aggregate["initial_reobserve_trigger_rate"] == 2 / 3
    assert aggregate["final_reobserve_trigger_rate"] == 1 / 3
    assert aggregate["closed_loop_execution_rate"] == 2 / 3
    assert aggregate["closed_loop_resolution_rate"] == 1 / 3
    assert aggregate["closed_loop_still_needed_rate"] == 1 / 3
    assert aggregate["closed_loop_selected_object_change_rate"] == 1 / 3
    assert aggregate["closed_loop_reobserve_reason_change_rate"] == 1 / 3
    assert aggregate["closed_loop_before_selected_still_selected_rate"] == 1 / 3
    assert aggregate["closed_loop_before_selected_received_observation_rate"] == 1 / 3
    assert aggregate["closed_loop_before_selected_gained_view_support_rate"] == 1 / 3
    assert aggregate["closed_loop_final_selected_absorbed_extra_view_rate"] == 1 / 3
    assert aggregate["closed_loop_extra_view_third_object_involved_rate"] == 1 / 3
    assert aggregate["mean_closed_loop_preferred_merge_count"] == 1.0
    assert aggregate["mean_closed_loop_preferred_merge_rate"] == 0.5
    assert aggregate["mean_closed_loop_delta_num_views"] == 2 / 3
    assert aggregate["mean_closed_loop_delta_num_memory_objects"] == 1 / 3
    assert aggregate["mean_closed_loop_delta_num_observations_added"] == 1.0
    assert aggregate["mean_closed_loop_delta_selected_overall_confidence"] == (0.2 + 0.0 - 0.1) / 3
    assert aggregate["mean_closed_loop_delta_selected_num_views"] == 1 / 3
    assert aggregate["mean_closed_loop_before_selected_delta_num_observations"] == 1 / 3
    assert aggregate["mean_closed_loop_before_selected_delta_num_views"] == 1 / 3
    assert aggregate["mean_closed_loop_extra_view_absorber_count"] == 1.0
    assert aggregate["reobserve_reason_counts"] == {
        "ambiguous_top_candidates": 1,
        "low_overall_confidence": 1,
        "none": 1,
    }
    assert aggregate["mean_selected_overall_confidence"] == (0.6 + 0.0 + 0.8) / 3
    assert per_query["red cube"]["total_runs"] == 2
    assert per_query["red cube"]["fraction_run_failed"] == 0.5
    assert per_query["blue mug"]["mean_num_views"] == 2.0


def test_failed_row_preserves_requested_benchmark_context(tmp_path: Path) -> None:
    args = _args(output_dir=tmp_path, skip_clip=True, view_preset="tabletop_3")
    args.detector_backend = "hf"
    args.camera_name = "base_camera"

    row = benchmark.failed_row(
        query="object",
        seed=0,
        message="subprocess returned 1",
        artifacts=str(tmp_path / "child"),
        args=args,
    )

    assert row["run_failed"] is True
    assert row["detector_backend"] == "hf"
    assert row["skip_clip"] is True
    assert row["view_preset"] == "tabletop_3"
    assert row["camera_name"] == "base_camera"


def test_multiview_fusion_benchmark_writes_outputs(monkeypatch, tmp_path: Path) -> None:
    seen_commands = []

    def fake_run(command, cwd, capture_output, text, check):
        seen_commands.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        query = command[command.index("--query") + 1]
        seed = int(command[command.index("--seed") + 1])
        run_dir = output_dir / f"fake_run_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "query": query,
            "num_views": 1,
            "num_memory_objects": 2,
            "num_observations_added": 2,
            "selected_object_id": "obj_0000",
            "selected_top_label": query,
            "selection_label": query,
            "selected_overall_confidence": 0.7,
            "should_reobserve": False,
            "reobserve_reason": "confident_enough",
            "initial_should_reobserve": True,
            "initial_reobserve_reason": "ambiguous_top_candidates",
            "final_should_reobserve": False,
            "final_reobserve_reason": "confident_enough",
            "closed_loop_reobserve_enabled": True,
            "closed_loop_reobserve_executed": True,
            "closed_loop_reobserve_view_ids": ["top_down"],
            "closed_loop_delta_num_views": 1,
            "closed_loop_delta_num_memory_objects": 0,
            "closed_loop_delta_num_observations_added": 1,
            "closed_loop_delta_selected_overall_confidence": 0.1,
            "closed_loop_delta_selected_num_views": 1,
            "closed_loop_delta_selected_num_observations": 1,
            "closed_loop_selected_object_changed": False,
            "closed_loop_reobserve_reason_changed": True,
            "closed_loop_reobserve_resolved": True,
            "closed_loop_reobserve_still_needed": False,
            "closed_loop_before_selected_present_after": True,
            "closed_loop_before_selected_still_selected": True,
            "closed_loop_before_selected_received_observation": True,
            "closed_loop_before_selected_gained_view_support": True,
            "closed_loop_before_selected_merged_extra_view_ids": ["top_down"],
            "closed_loop_before_selected_delta_num_observations": 1,
            "closed_loop_before_selected_delta_num_views": 1,
            "closed_loop_extra_view_absorber_object_ids": ["obj_0000"],
            "closed_loop_extra_view_absorber_count": 1,
            "closed_loop_final_selected_absorbed_extra_view": True,
            "closed_loop_extra_view_third_object_ids": [],
            "closed_loop_extra_view_third_object_involved": False,
            "closed_loop_selected_object_continuity_enabled": True,
            "closed_loop_preferred_merge_count": 1,
            "closed_loop_preferred_merge_rate": 1.0,
            "runtime_seconds": 2.5,
            "detector_backend": "mock",
            "skip_clip": True,
            "view_preset": "none",
            "camera_name": None,
            "artifacts": str(run_dir),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    output_dir = tmp_path / "benchmark"
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_multiview_fusion_benchmark.py",
            "--queries",
            "red cube",
            "blue mug",
            "--seeds",
            "0",
            "1",
            "--detector-backend",
            "mock",
            "--mock-box-position",
            "all",
            "--skip-clip",
            "--depth-scale",
            "1000",
            "--enable-closed-loop-reobserve",
            "--enable-selected-object-continuity",
            "--selected-object-continuity-distance-scale",
            "1.25",
            "--output-dir",
            str(output_dir),
        ],
    )

    benchmark.main()

    rows = json.loads((output_dir / "benchmark_rows.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    csv_header = (output_dir / "benchmark_rows.csv").read_text(encoding="utf-8").splitlines()[0]

    assert len(rows) == 4
    assert rows[0]["seed"] == 0
    assert rows[0]["has_selected_object"] is True
    assert summary["total_runs"] == 4
    assert summary["skip_clip"] is True
    assert summary["closed_loop_selected_object_continuity_enabled"] is True
    assert summary["selected_object_continuity_distance_scale"] == 1.25
    assert summary["aggregate_metrics"]["fraction_with_selected_object"] == 1.0
    assert summary["aggregate_metrics"]["reobserve_trigger_rate"] == 0.0
    assert summary["aggregate_metrics"]["initial_reobserve_trigger_rate"] == 1.0
    assert summary["aggregate_metrics"]["final_reobserve_trigger_rate"] == 0.0
    assert summary["aggregate_metrics"]["closed_loop_execution_rate"] == 1.0
    assert summary["aggregate_metrics"]["closed_loop_resolution_rate"] == 1.0
    assert summary["aggregate_metrics"]["closed_loop_still_needed_rate"] == 0.0
    assert summary["aggregate_metrics"]["closed_loop_before_selected_received_observation_rate"] == 1.0
    assert summary["aggregate_metrics"]["closed_loop_before_selected_gained_view_support_rate"] == 1.0
    assert summary["aggregate_metrics"]["closed_loop_final_selected_absorbed_extra_view_rate"] == 1.0
    assert summary["aggregate_metrics"]["closed_loop_extra_view_third_object_involved_rate"] == 0.0
    assert summary["aggregate_metrics"]["mean_closed_loop_preferred_merge_count"] == 1.0
    assert summary["aggregate_metrics"]["mean_closed_loop_preferred_merge_rate"] == 1.0
    assert summary["aggregate_metrics"]["mean_closed_loop_delta_selected_num_views"] == 1.0
    assert summary["aggregate_metrics"]["mean_closed_loop_delta_selected_overall_confidence"] == 0.1
    assert summary["aggregate_metrics"]["mean_closed_loop_before_selected_delta_num_views"] == 1.0
    assert summary["aggregate_metrics"]["mean_closed_loop_extra_view_absorber_count"] == 1.0
    assert summary["aggregate_metrics"]["reobserve_reason_counts"] == {"confident_enough": 4}
    assert summary["aggregate_metrics"]["mean_num_memory_objects"] == 2.0
    assert "selected_overall_confidence" in csv_header
    assert "should_reobserve" in csv_header
    assert "closed_loop_delta_selected_num_views" in csv_header
    assert "closed_loop_reobserve_resolved" in csv_header
    assert "closed_loop_before_selected_received_observation" in csv_header
    assert "closed_loop_final_selected_absorbed_extra_view" in csv_header
    assert "closed_loop_selected_object_continuity_enabled" in csv_header
    assert "closed_loop_preferred_merge_rate" in csv_header
    assert all("--skip-clip" in command for command in seen_commands)
    assert all("--enable-closed-loop-reobserve" in command for command in seen_commands)
    assert all("--enable-selected-object-continuity" in command for command in seen_commands)
    assert all("--selected-object-continuity-distance-scale" in command for command in seen_commands)


def test_multiview_fusion_benchmark_can_fail_on_child_error(monkeypatch, tmp_path: Path) -> None:
    def fake_run(command, cwd, capture_output, text, check):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="boom")

    output_dir = tmp_path / "benchmark"
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_multiview_fusion_benchmark.py",
            "--queries",
            "object",
            "--seeds",
            "0",
            "--detector-backend",
            "mock",
            "--skip-clip",
            "--view-preset",
            "tabletop_3",
            "--camera-name",
            "base_camera",
            "--fail-on-child-error",
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        benchmark.main()

    summary = json.loads((output_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    rows = json.loads((output_dir / "benchmark_rows.json").read_text(encoding="utf-8"))

    assert exc_info.value.code == 1
    assert summary["aggregate_metrics"]["failed_runs"] == 1
    assert rows[0]["run_failed"] is True
    assert rows[0]["view_preset"] == "tabletop_3"
    assert rows[0]["camera_name"] == "base_camera"


def _args(
    output_dir: Path,
    skip_clip: bool,
    view_ids: list[str] | None = None,
    view_preset: str = "none",
):
    return type(
        "Args",
        (),
        {
            "env_id": "PickCube-v1",
            "obs_mode": "rgbd",
            "detector_backend": "mock",
            "mock_box_position": "center",
            "depth_scale": 1000.0,
            "merge_distance": 0.08,
            "camera_name": None,
            "view_preset": view_preset,
            "view_ids": view_ids or [],
            "skip_clip": skip_clip,
            "output_dir": output_dir,
            "enable_closed_loop_reobserve": False,
            "closed_loop_max_extra_views": 1,
            "enable_selected_object_continuity": False,
            "selected_object_continuity_distance_scale": 1.0,
        },
    )()
