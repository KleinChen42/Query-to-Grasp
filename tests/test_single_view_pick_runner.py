from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import scripts.run_single_view_pick_benchmark as benchmark


def test_single_view_pick_benchmark_writes_outputs(monkeypatch, tmp_path) -> None:
    def fake_run(command, cwd, capture_output, text, check):
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
    assert summary["total_runs"] == 4
    assert summary["aggregate_metrics"]["total_runs"] == 4
