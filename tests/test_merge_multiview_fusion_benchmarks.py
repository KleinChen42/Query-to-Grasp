from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.merge_multiview_fusion_benchmarks import merge_benchmark_dirs


def test_merge_benchmark_dirs_combines_rows_and_metrics(tmp_path: Path) -> None:
    first = tmp_path / "seed0"
    second = tmp_path / "seed12"
    _write_benchmark(first, seed=0, confidence=0.5, should_reobserve=True)
    _write_benchmark(second, seed=1, confidence=0.7, should_reobserve=False)
    output_dir = tmp_path / "merged"

    summary = merge_benchmark_dirs([first, second], output_dir)

    rows = json.loads((output_dir / "benchmark_rows.json").read_text(encoding="utf-8"))
    csv_header = (output_dir / "benchmark_rows.csv").read_text(encoding="utf-8").splitlines()[0]
    saved_summary = json.loads((output_dir / "benchmark_summary.json").read_text(encoding="utf-8"))

    assert summary["total_runs"] == 2
    assert saved_summary["total_runs"] == 2
    assert summary["detector_backend"] == "hf"
    assert summary["skip_clip"] is True
    assert summary["view_preset"] == "tabletop_3"
    assert summary["aggregate_metrics"]["fraction_with_selected_object"] == 1.0
    assert summary["aggregate_metrics"]["reobserve_trigger_rate"] == 0.5
    assert summary["aggregate_metrics"]["mean_selected_overall_confidence"] == 0.6
    assert summary["per_query_metrics"]["object"]["total_runs"] == 2
    assert rows[0]["source_benchmark_dir"] == str(first)
    assert rows[1]["source_benchmark_dir"] == str(second)
    assert "source_benchmark_dir" not in csv_header


def test_merge_benchmark_dirs_marks_mixed_metadata(tmp_path: Path) -> None:
    first = tmp_path / "skip"
    second = tmp_path / "clip"
    _write_benchmark(first, seed=0, confidence=0.5, should_reobserve=True, skip_clip=True)
    _write_benchmark(second, seed=1, confidence=0.7, should_reobserve=False, skip_clip=False)

    summary = merge_benchmark_dirs([first, second], tmp_path / "merged")

    assert summary["skip_clip"] == "mixed"


def test_merge_benchmark_dirs_missing_behavior(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    missing = tmp_path / "missing"
    _write_benchmark(existing, seed=0, confidence=0.5, should_reobserve=True)

    with pytest.raises(FileNotFoundError, match="Missing benchmark"):
        merge_benchmark_dirs([existing, missing], tmp_path / "merged")

    summary = merge_benchmark_dirs([existing, missing], tmp_path / "partial", skip_missing=True)

    assert summary["total_runs"] == 1


def _write_benchmark(
    benchmark_dir: Path,
    seed: int,
    confidence: float,
    should_reobserve: bool,
    skip_clip: bool = True,
) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "query": "object",
            "seed": seed,
            "num_views": 3,
            "num_memory_objects": 2,
            "num_observations_added": 4,
            "has_selected_object": True,
            "selected_object_id": "obj_0000",
            "selected_top_label": "object",
            "selected_overall_confidence": confidence,
            "selection_label": "object",
            "should_reobserve": should_reobserve,
            "reobserve_reason": "insufficient_view_support" if should_reobserve else "confident_enough",
            "runtime_seconds": 10.0,
            "detector_backend": "hf",
            "skip_clip": skip_clip,
            "view_preset": "tabletop_3",
            "camera_name": "base_camera",
            "artifacts": f"outputs/run_{seed}",
            "run_failed": False,
            "error_message": "",
        }
    ]
    summary = {
        "total_runs": 1,
        "unique_queries": ["object"],
        "view_ids": [],
        "camera_name": "base_camera",
        "view_preset": "tabletop_3",
        "detector_backend": "hf",
        "skip_clip": skip_clip,
        "depth_scale": 1000.0,
        "merge_distance": 0.08,
        "aggregate_metrics": {},
        "per_query_metrics": {},
    }
    (benchmark_dir / "benchmark_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (benchmark_dir / "benchmark_summary.json").write_text(json.dumps(summary), encoding="utf-8")
