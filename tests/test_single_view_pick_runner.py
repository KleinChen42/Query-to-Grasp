from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import scripts.run_single_view_pick_benchmark as benchmark


def test_load_queries_ignores_blank_lines_and_comments(tmp_path: Path) -> None:
    queries_file = tmp_path / "queries.txt"
    queries_file.write_text("# exact objects\n\nred cube\n  blue mug  \n# broad queries\nobject\n", encoding="utf-8")

    queries = benchmark.load_queries(queries_file, ["  cup  ", ""])

    assert queries == ["red cube", "blue mug", "object", "cup"]


def test_single_view_pick_benchmark_writes_outputs(monkeypatch, tmp_path) -> None:
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
            "num_detections": 1,
            "num_ranked_candidates": 1,
            "camera_xyz": [0.1, 0.2, 0.3],
            "world_xyz": None,
            "num_3d_points": 12,
            "pick_success": False,
            "pick_stage": "placeholder_not_executed",
            "runtime_seconds": 1.5,
            "artifacts": str(run_dir),
        }
        pick_result = {
            "success": False,
            "stage": "placeholder_not_executed",
            "target_xyz": [0.1, 0.2, 0.3],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "pick_result.json").write_text(json.dumps(pick_result), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    output_dir = tmp_path / "benchmark"
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_single_view_pick_benchmark.py",
            "--queries",
            "red cube",
            "blue mug",
            "--seeds",
            "0",
            "1",
            "--detector-backend",
            "mock",
            "--mock-box-position",
            "center",
            "--skip-clip",
            "--depth-scale",
            "1000",
            "--grasp-target-mode",
            "refined",
            "--output-dir",
            str(output_dir),
        ],
    )

    benchmark.main()

    rows_json = output_dir / "benchmark_rows.json"
    rows_csv = output_dir / "benchmark_rows.csv"
    summary_json = output_dir / "benchmark_summary.json"
    assert rows_json.exists()
    assert rows_csv.exists()
    assert summary_json.exists()

    rows = json.loads(rows_json.read_text(encoding="utf-8"))
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert len(rows) == 4
    assert rows[0]["seed"] == 0
    assert rows[0]["run_failed"] is False
    assert rows[0]["runtime_seconds"] == 1.5
    assert summary["total_runs"] == 4
    assert summary["env_id"] == "PickCube-v1"
    assert summary["obs_mode"] == "rgbd"
    assert summary["skip_clip"] is True
    assert summary["grasp_target_mode"] == "refined"
    assert summary["aggregate_metrics"]["total_runs"] == 4
    assert summary["aggregate_metrics"]["failed_runs"] == 0
    assert summary["aggregate_metrics"]["mean_runtime_seconds"] == 1.5
    assert "red cube" in summary["per_query_metrics"]
    assert "runtime_seconds" in rows_csv.read_text(encoding="utf-8").splitlines()[0]
    assert all("--skip-clip" in command for command in seen_commands)
    assert all("--grasp-target-mode" in command for command in seen_commands)
    assert all("refined" in command for command in seen_commands)


def test_single_view_pick_benchmark_forwards_place_target_source(monkeypatch, tmp_path) -> None:
    seen_commands = []

    def fake_run(command, cwd, capture_output, text, check):
        seen_commands.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        run_dir = output_dir / "fake_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "query": command[command.index("--query") + 1],
            "num_detections": 1,
            "num_ranked_candidates": 1,
            "camera_xyz": [0.1, 0.2, 0.3],
            "world_xyz": [0.1, 0.2, 0.3],
            "num_3d_points": 12,
            "grasp_attempted": True,
            "pick_success": True,
            "place_attempted": True,
            "place_success": False,
            "place_target_xyz": [0.0, 0.0, 0.05],
            "place_target_source": "oracle_cubeB_pose",
            "task_success": False,
            "pick_stage": "place_not_confirmed",
            "runtime_seconds": 1.5,
            "artifacts": str(run_dir),
        }
        pick_result = {
            "success": False,
            "pick_success": True,
            "place_success": False,
            "stage": "place_not_confirmed",
            "target_xyz": [0.1, 0.2, 0.3],
            "place_xyz": [0.0, 0.0, 0.05],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "pick_result.json").write_text(json.dumps(pick_result), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    output_dir = tmp_path / "benchmark"
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_single_view_pick_benchmark.py",
            "--queries",
            "red cube",
            "--seeds",
            "0",
            "--detector-backend",
            "mock",
            "--skip-clip",
            "--pick-executor",
            "sim_pick_place",
            "--grasp-target-mode",
            "refined",
            "--place-target-source",
            "oracle_cubeB_pose",
            "--output-dir",
            str(output_dir),
        ],
    )

    benchmark.main()

    assert seen_commands
    command = seen_commands[0]
    assert command[command.index("--pick-executor") + 1] == "sim_pick_place"
    assert command[command.index("--place-target-source") + 1] == "oracle_cubeB_pose"
    summary = json.loads((output_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["place_target_source"] == "oracle_cubeB_pose"
    assert summary["aggregate_metrics"]["place_attempted_rate"] == 1.0


def test_single_view_pick_benchmark_forwards_sensor_resolution(monkeypatch, tmp_path) -> None:
    seen_commands = []

    def fake_run(command, cwd, capture_output, text, check):
        seen_commands.append(command)
        output_dir = Path(command[command.index("--output-dir") + 1])
        run_dir = output_dir / "fake_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "query": "red cube",
            "num_detections": 1,
            "num_ranked_candidates": 1,
            "camera_xyz": [0.1, 0.2, 0.3],
            "world_xyz": [0.1, 0.2, 0.3],
            "num_3d_points": 12,
            "pick_success": False,
            "pick_stage": "placeholder_not_executed",
            "runtime_seconds": 1.5,
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
            "run_single_view_pick_benchmark.py",
            "--queries",
            "red cube",
            "--seeds",
            "0",
            "--detector-backend",
            "mock",
            "--skip-clip",
            "--sensor-width",
            "720",
            "--sensor-height",
            "720",
            "--output-dir",
            str(output_dir),
        ],
    )

    benchmark.main()

    assert seen_commands
    command = seen_commands[0]
    assert command[command.index("--sensor-width") + 1] == "720"
    assert command[command.index("--sensor-height") + 1] == "720"


def test_single_view_pick_benchmark_defaults_to_clip_enabled(monkeypatch, tmp_path) -> None:
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
            "num_detections": 1,
            "num_ranked_candidates": 1,
            "camera_xyz": [0.1, 0.2, 0.3],
            "world_xyz": None,
            "num_3d_points": 12,
            "pick_success": False,
            "pick_stage": "placeholder_not_executed",
            "runtime_seconds": 1.5,
            "artifacts": str(run_dir),
        }
        pick_result = {
            "success": False,
            "stage": "placeholder_not_executed",
            "target_xyz": [0.1, 0.2, 0.3],
        }
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (run_dir / "pick_result.json").write_text(json.dumps(pick_result), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    output_dir = tmp_path / "benchmark"
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_single_view_pick_benchmark.py",
            "--queries",
            "red cube",
            "--seeds",
            "0",
            "--detector-backend",
            "mock",
            "--mock-box-position",
            "center",
            "--depth-scale",
            "1000",
            "--output-dir",
            str(output_dir),
        ],
    )

    benchmark.main()

    summary = json.loads((output_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["skip_clip"] is False
    assert seen_commands
    assert all("--skip-clip" not in command for command in seen_commands)
