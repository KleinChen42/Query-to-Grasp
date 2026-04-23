from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_reobserve_policy_report import (
    build_policy_report,
    format_reason_counts,
    render_markdown,
)


def test_build_policy_report_maps_aggregate_and_examples(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "fusion"
    _write_fusion_benchmark(benchmark_dir)

    report = build_policy_report([f"HF fusion={benchmark_dir}"], max_examples=1)

    benchmark = report["benchmarks"][0]
    assert benchmark["label"] == "HF fusion"
    assert benchmark["total_runs"] == 2
    assert benchmark["reobserve_trigger_rate"] == 0.5
    assert benchmark["initial_reobserve_trigger_rate"] == 1.0
    assert benchmark["final_reobserve_trigger_rate"] == 0.5
    assert benchmark["closed_loop_execution_rate"] == 0.5
    assert benchmark["closed_loop_resolution_rate"] == 0.5
    assert benchmark["closed_loop_still_needed_rate"] == 0.5
    assert benchmark["closed_loop_before_selected_received_observation_rate"] == 0.5
    assert benchmark["closed_loop_before_selected_gained_view_support_rate"] == 0.5
    assert benchmark["mean_closed_loop_delta_selected_num_views"] == 0.5
    assert benchmark["reobserve_reason_counts"] == {"ambiguous_top_candidates": 1, "confident_enough": 1}
    assert benchmark["per_query"][0]["query"] == "blue mug"
    assert benchmark["per_query"][1]["query"] == "red cube"
    assert benchmark["trigger_examples"] == [
        {
            "query": "red cube",
            "seed": 0,
            "reason": "ambiguous_top_candidates",
            "selected_overall_confidence": 0.62,
            "artifacts": "outputs/run_red",
        }
    ]
    assert "reduced the mean policy trigger rate" in report["conclusion"]


def test_render_markdown_contains_policy_sections(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "fusion"
    _write_fusion_benchmark(benchmark_dir)
    report = build_policy_report([f"HF fusion={benchmark_dir}"])

    markdown = render_markdown(report)

    assert "# Re-Observation Policy Diagnostics" in markdown
    assert "## Benchmarks" in markdown
    assert "## Per-Query Breakdown" in markdown
    assert "## Trigger Examples" in markdown
    assert "reobserve_trigger_rate" in markdown
    assert "initial_trigger_rate" in markdown
    assert "final_trigger_rate" in markdown
    assert "closed_loop_execution_rate" in markdown
    assert "resolution_rate" in markdown
    assert "selected_assoc_rate" in markdown
    assert "selected_support_gain_rate" in markdown
    assert "delta_selected_views" in markdown
    assert "ambiguous_top_candidates: 1" in markdown
    assert "resolution rate 0.5000" in markdown


def test_conclusion_ignores_confident_enough_as_trigger_reason(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "fusion"
    _write_fusion_benchmark(
        benchmark_dir,
        reason_counts={"confident_enough": 7, "insufficient_view_support": 2, "low_overall_confidence": 1},
        trigger_rate=0.3,
        closed_loop_execution_rate=0.0,
    )

    report = build_policy_report([f"HF fusion={benchmark_dir}"])

    assert "insufficient_view_support" in report["conclusion"]
    assert "triggered reason was `confident_enough`" not in report["conclusion"]


def test_build_policy_report_missing_behavior(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Missing benchmark_summary"):
        build_policy_report([f"missing={missing}"])

    with pytest.raises(ValueError, match="No benchmark summaries"):
        build_policy_report([f"missing={missing}"], skip_missing=True)


def test_format_reason_counts_is_stable() -> None:
    text = format_reason_counts({"none": 1, "ambiguous_top_candidates": 3})

    assert text == "ambiguous_top_candidates: 3; none: 1"


def _write_fusion_benchmark(
    benchmark_dir: Path,
    reason_counts: dict[str, int] | None = None,
    trigger_rate: float = 0.5,
    closed_loop_execution_rate: float = 0.5,
) -> None:
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    reason_counts = reason_counts or {
        "ambiguous_top_candidates": 1,
        "confident_enough": 1,
    }
    summary = {
        "total_runs": 2,
        "detector_backend": "hf",
        "skip_clip": True,
        "view_preset": "tabletop_3",
        "aggregate_metrics": {
            "total_runs": 2,
            "fraction_with_selected_object": 1.0,
            "mean_selected_overall_confidence": 0.7,
            "mean_num_views": 3.0,
            "mean_num_memory_objects": 2.0,
            "reobserve_trigger_rate": trigger_rate,
            "initial_reobserve_trigger_rate": 1.0,
            "final_reobserve_trigger_rate": trigger_rate,
            "closed_loop_execution_rate": closed_loop_execution_rate,
            "closed_loop_resolution_rate": trigger_rate,
            "closed_loop_still_needed_rate": trigger_rate,
            "closed_loop_before_selected_still_selected_rate": trigger_rate,
            "closed_loop_before_selected_received_observation_rate": trigger_rate,
            "closed_loop_before_selected_gained_view_support_rate": trigger_rate,
            "mean_closed_loop_delta_selected_overall_confidence": 0.1,
            "mean_closed_loop_delta_selected_num_views": 0.5,
            "mean_closed_loop_delta_num_memory_objects": 0.0,
            "mean_closed_loop_before_selected_delta_num_observations": trigger_rate,
            "mean_closed_loop_before_selected_delta_num_views": trigger_rate,
            "reobserve_reason_counts": reason_counts,
            "initial_reobserve_reason_counts": {"ambiguous_top_candidates": 2},
            "final_reobserve_reason_counts": reason_counts,
        },
        "per_query_metrics": {
            "red cube": {
                "total_runs": 1,
                "fraction_with_selected_object": 1.0,
                "mean_selected_overall_confidence": 0.62,
                "reobserve_trigger_rate": 1.0,
                "initial_reobserve_trigger_rate": 1.0,
                "final_reobserve_trigger_rate": 1.0,
                "closed_loop_execution_rate": 1.0,
                "closed_loop_resolution_rate": 0.0,
                "closed_loop_still_needed_rate": 1.0,
                "closed_loop_before_selected_still_selected_rate": 1.0,
                "closed_loop_before_selected_received_observation_rate": 1.0,
                "closed_loop_before_selected_gained_view_support_rate": 1.0,
                "mean_closed_loop_delta_selected_overall_confidence": 0.2,
                "mean_closed_loop_delta_selected_num_views": 1.0,
                "mean_closed_loop_before_selected_delta_num_observations": 1.0,
                "mean_closed_loop_before_selected_delta_num_views": 1.0,
                "reobserve_reason_counts": {"ambiguous_top_candidates": 1},
            },
            "blue mug": {
                "total_runs": 1,
                "fraction_with_selected_object": 1.0,
                "mean_selected_overall_confidence": 0.78,
                "reobserve_trigger_rate": 0.0,
                "initial_reobserve_trigger_rate": 1.0,
                "final_reobserve_trigger_rate": 0.0,
                "closed_loop_execution_rate": 0.0,
                "closed_loop_resolution_rate": 1.0,
                "closed_loop_still_needed_rate": 0.0,
                "closed_loop_before_selected_still_selected_rate": 0.0,
                "closed_loop_before_selected_received_observation_rate": 0.0,
                "closed_loop_before_selected_gained_view_support_rate": 0.0,
                "mean_closed_loop_delta_selected_overall_confidence": 0.0,
                "mean_closed_loop_delta_selected_num_views": 0.0,
                "mean_closed_loop_before_selected_delta_num_observations": 0.0,
                "mean_closed_loop_before_selected_delta_num_views": 0.0,
                "reobserve_reason_counts": {"confident_enough": 1},
            },
        },
    }
    rows = [
        {
            "query": "red cube",
            "seed": 0,
            "should_reobserve": True,
            "reobserve_reason": "ambiguous_top_candidates",
            "selected_overall_confidence": 0.62,
            "artifacts": "outputs/run_red",
        },
        {
            "query": "blue mug",
            "seed": 1,
            "should_reobserve": False,
            "reobserve_reason": "confident_enough",
            "selected_overall_confidence": 0.78,
            "artifacts": "outputs/run_blue",
        },
    ]
    (benchmark_dir / "benchmark_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (benchmark_dir / "benchmark_rows.json").write_text(json.dumps(rows), encoding="utf-8")
