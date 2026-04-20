from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_benchmark_report import generate_report


def test_generate_benchmark_report_writes_outputs(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    _write_fake_benchmark(benchmark_dir, query="red cube", mean_detections=1.0, pick_rate=0.0)

    output_md = benchmark_dir / "report.md"
    output_json = benchmark_dir / "report_summary.json"
    report_summary = generate_report(benchmark_dir=benchmark_dir, output_md=output_md, output_json=output_json)

    assert output_md.exists()
    assert output_json.exists()
    markdown = output_md.read_text(encoding="utf-8")
    saved_summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert "# Benchmark Report" in markdown
    assert "Aggregate Metrics" in markdown
    assert "mean_runtime_seconds" in markdown
    assert "mean_raw_num_detections" in markdown
    assert "fraction_top1_changed_by_rerank" in markdown
    assert "## Ambiguity Conclusion" not in markdown
    assert "## Per-Query Breakdown" in markdown
    assert report_summary["benchmark_dir"] == str(benchmark_dir)
    assert saved_summary["primary_summary"]["detector_backend"] == "mock"


def test_generate_benchmark_report_comparison_mode(tmp_path: Path) -> None:
    primary_dir = tmp_path / "primary"
    secondary_dir = tmp_path / "secondary"
    _write_fake_benchmark(primary_dir, query="red cube", mean_detections=2.0, pick_rate=0.25)
    _write_fake_benchmark(secondary_dir, query="blue mug", mean_detections=1.0, pick_rate=0.0)

    output_md = primary_dir / "report.md"
    output_json = primary_dir / "report_summary.json"
    generate_report(
        benchmark_dir=primary_dir,
        compare_benchmark_dir=secondary_dir,
        output_md=output_md,
        output_json=output_json,
    )

    markdown = output_md.read_text(encoding="utf-8")
    saved_summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert "## Comparison" in markdown
    assert "## Per-Query Breakdown" in markdown
    assert "| Metric | Primary | Secondary | Delta |" in markdown
    assert "comparison_metrics" in saved_summary
    assert saved_summary["comparison_metrics"]["mean_num_detections"]["delta"] == 1.0
    assert "mean_runtime_seconds" in saved_summary["comparison_metrics"]
    assert "mean_raw_num_detections" in saved_summary["comparison_metrics"]
    assert "fraction_top1_changed_by_rerank" in saved_summary["comparison_metrics"]
    assert saved_summary["compare_benchmark_dir"] == str(secondary_dir)


def test_generate_benchmark_report_notes_reranking_opportunity(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "ambiguity_benchmark"
    _write_fake_benchmark(
        benchmark_dir,
        query="object",
        mean_detections=2.0,
        pick_rate=0.0,
        top1_changed_rate=0.5,
    )

    output_md = benchmark_dir / "report.md"
    generate_report(benchmark_dir=benchmark_dir, output_md=output_md, output_json=benchmark_dir / "report_summary.json")

    markdown = output_md.read_text(encoding="utf-8")
    assert "## Ambiguity Conclusion" in markdown
    assert "Reranking has measurable opportunity in this benchmark setting." in markdown


def test_generate_benchmark_report_notes_limited_ambiguity_headroom(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "ambiguity_benchmark"
    _write_fake_benchmark(benchmark_dir, query="object", mean_detections=1.0, pick_rate=0.0)

    output_md = benchmark_dir / "report.md"
    generate_report(benchmark_dir=benchmark_dir, output_md=output_md, output_json=benchmark_dir / "report_summary.json")

    markdown = output_md.read_text(encoding="utf-8")
    assert "## Ambiguity Conclusion" in markdown
    assert "Current ambiguity benchmark still does not provide useful reranking headroom." in markdown


def test_generate_benchmark_report_supports_old_summary_without_runtime(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "query": "red cube",
            "seed": 0,
            "num_detections": 1,
            "num_ranked_candidates": 1,
            "has_3d_target": True,
            "num_3d_points": 10,
            "pick_success": False,
            "pick_stage": "placeholder_not_executed",
            "artifacts": str(benchmark_dir / "runs" / "run_0001"),
        }
    ]
    summary = {
        "timestamp": "2026-04-20T00:00:00+00:00",
        "total_runs": 1,
        "unique_queries": ["red cube"],
        "detector_backend": "mock",
        "skip_clip": True,
        "depth_scale": 1000.0,
        "aggregate_metrics": {
            "total_runs": 1,
            "mean_num_detections": 1.0,
            "mean_num_ranked_candidates": 1.0,
            "mean_num_3d_points": 10.0,
            "fraction_with_3d_target": 1.0,
            "pick_success_rate": 0.0,
            "pick_stage_counts": {"placeholder_not_executed": 1},
        },
    }
    (benchmark_dir / "benchmark_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (benchmark_dir / "benchmark_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    output_md = benchmark_dir / "report.md"
    generate_report(benchmark_dir=benchmark_dir, output_md=output_md, output_json=benchmark_dir / "report_summary.json")

    markdown = output_md.read_text(encoding="utf-8")
    assert "mean_runtime_seconds" in markdown
    assert "mean_raw_num_detections" in markdown
    assert "fraction_top1_changed_by_rerank" in markdown
    assert "## Per-Query Breakdown" in markdown
    assert "| red cube | 1 | 1 | 1 | 1 | 10 | 1 | 0 | 0 | 0 |" in markdown


def _write_fake_benchmark(
    benchmark_dir: Path,
    query: str,
    mean_detections: float,
    pick_rate: float,
    top1_changed_rate: float = 0.0,
) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "query": query,
            "seed": 0,
            "raw_num_detections": int(mean_detections),
            "num_detections": int(mean_detections),
            "num_ranked_candidates": 1,
            "top1_changed_by_rerank": top1_changed_rate > 0.0,
            "detector_top_phrase": query,
            "final_top_phrase": query,
            "has_3d_target": True,
            "num_3d_points": 10,
            "pick_success": pick_rate > 0.0,
            "pick_stage": "placeholder_not_executed",
            "runtime_seconds": 1.5,
            "artifacts": str(benchmark_dir / "runs" / "run_0001"),
        }
    ]
    summary = {
        "timestamp": "2026-04-20T00:00:00+00:00",
        "total_runs": 1,
        "unique_queries": [query],
        "detector_backend": "mock",
        "skip_clip": True,
        "depth_scale": 1000.0,
        "aggregate_metrics": {
            "total_runs": 1,
            "mean_raw_num_detections": mean_detections,
            "mean_num_detections": mean_detections,
            "mean_num_ranked_candidates": 1.0,
            "mean_num_3d_points": 10.0,
            "fraction_with_3d_target": 1.0,
            "pick_success_rate": pick_rate,
            "fraction_top1_changed_by_rerank": top1_changed_rate,
            "mean_runtime_seconds": 1.5,
            "pick_stage_counts": {"placeholder_not_executed": 1},
        },
        "per_query_metrics": {
            query: {
                "total_runs": 1,
                "mean_raw_num_detections": mean_detections,
                "mean_num_detections": mean_detections,
                "mean_num_ranked_candidates": 1.0,
                "mean_num_3d_points": 10.0,
                "fraction_with_3d_target": 1.0,
                "pick_success_rate": pick_rate,
                "fraction_top1_changed_by_rerank": top1_changed_rate,
                "mean_runtime_seconds": 1.5,
                "pick_stage_counts": {"placeholder_not_executed": 1},
            }
        },
    }
    (benchmark_dir / "benchmark_rows.json").write_text(json.dumps(rows), encoding="utf-8")
    (benchmark_dir / "benchmark_summary.json").write_text(json.dumps(summary), encoding="utf-8")
