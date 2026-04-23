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
            "should_reobserve": True,
            "reobserve_reason": "ambiguous_top_candidates",
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
    assert summary["aggregate_metrics"]["fraction_with_selected_object"] == 1.0
    assert summary["aggregate_metrics"]["reobserve_trigger_rate"] == 1.0
    assert summary["aggregate_metrics"]["reobserve_reason_counts"] == {"ambiguous_top_candidates": 4}
    assert summary["aggregate_metrics"]["mean_num_memory_objects"] == 2.0
    assert "selected_overall_confidence" in csv_header
    assert "should_reobserve" in csv_header
    assert all("--skip-clip" in command for command in seen_commands)


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
        },
    )()
