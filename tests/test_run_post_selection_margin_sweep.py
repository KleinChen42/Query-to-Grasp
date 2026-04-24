from __future__ import annotations

import json
from pathlib import Path

from scripts.run_post_selection_margin_sweep import (
    build_benchmark_command,
    build_conclusion,
    build_policy_report_command,
    build_table_rows,
    margin_label,
    margin_tag,
    render_markdown,
)


class _Args:
    queries_file = Path("configs") / "ambiguity_queries.txt"
    seeds = [0]
    num_runs = 1
    detector_backend = "hf"
    mock_box_position = "center"
    skip_clip = False
    depth_scale = 1000.0
    env_id = "PickCube-v1"
    obs_mode = "rgbd"
    view_preset = "tabletop_3"
    camera_name = "base_camera"
    merge_distance = 0.08
    closed_loop_max_extra_views = 1
    selected_object_continuity_distance_scale = 1.0
    fail_on_child_error = True


def test_build_benchmark_command_defaults_to_use_clip() -> None:
    command = build_benchmark_command(_Args(), margin=0.05, output_dir=Path("outputs") / "sweep")

    assert "--use-clip" in command
    assert "--skip-clip" not in command
    assert "--enable-closed-loop-reobserve" in command
    assert "--enable-selected-object-continuity" in command
    assert "--enable-post-reobserve-selection-continuity" in command
    assert "--post-reobserve-selection-margin" in command
    assert "0.05" in command


def test_build_benchmark_command_respects_skip_clip() -> None:
    args = _Args()
    args.skip_clip = True

    command = build_benchmark_command(args, margin=0.08, output_dir=Path("outputs") / "sweep")

    assert "--skip-clip" in command
    assert "--use-clip" not in command


def test_build_policy_report_command_collects_specs() -> None:
    command = build_policy_report_command(["m0.03=outputs/a", "m0.05=outputs/b"], Path("outputs") / "sweep")

    assert command.count("--benchmark") == 2
    assert "m0.03=outputs/a" in command
    assert any(str(item).endswith("reobserve_policy_report_margin_sweep.md") for item in command)


def test_build_table_rows_and_markdown(tmp_path: Path) -> None:
    _write_summary(tmp_path / "margin_0p03", margin=0.03, apply_rate=0.0, resolution_rate=0.0)
    _write_summary(tmp_path / "margin_0p08", margin=0.08, apply_rate=0.25, resolution_rate=0.0)

    rows = build_table_rows([0.03, 0.08], tmp_path)
    conclusion = build_conclusion(rows)
    markdown = render_markdown(rows, conclusion)

    assert rows[0]["label"] == "0.03"
    assert rows[1]["closed_loop_post_selection_continuity_apply_rate"] == 0.25
    assert "Margin 0.08 increased post-selection continuity application to 0.2500" in conclusion
    assert "# Post-Selection Margin Sweep" in markdown
    assert "closed_loop_post_selection_continuity_apply_rate" in markdown
    assert "0.2500" in markdown


def test_margin_helpers_are_stable() -> None:
    assert margin_label(0.03) == "0.03"
    assert margin_label(0.1) == "0.1"
    assert margin_tag(0.03) == "margin_0p03"


def _write_summary(path: Path, margin: float, apply_rate: float, resolution_rate: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "detector_backend": "hf",
        "skip_clip": False,
        "total_runs": 4,
        "post_reobserve_selection_margin": margin,
        "aggregate_metrics": {
            "total_runs": 4,
            "closed_loop_execution_rate": 0.75,
            "closed_loop_resolution_rate": resolution_rate,
            "closed_loop_still_needed_rate": 0.75,
            "closed_loop_selected_object_change_rate": 0.5,
            "closed_loop_before_selected_received_observation_rate": 0.25,
            "closed_loop_before_selected_gained_view_support_rate": 0.25,
            "closed_loop_final_selected_absorbed_extra_view_rate": 0.25,
            "closed_loop_extra_view_third_object_involved_rate": 0.5,
            "mean_closed_loop_preferred_merge_rate": 0.25,
            "closed_loop_post_selection_continuity_eligibility_rate": 0.25,
            "closed_loop_post_selection_continuity_apply_rate": apply_rate,
            "mean_closed_loop_delta_selected_overall_confidence": 0.01,
            "mean_closed_loop_delta_selected_num_views": 0.25,
            "mean_selected_overall_confidence": 0.66,
            "mean_runtime_seconds": 123.4,
        },
    }
    (path / "benchmark_summary.json").write_text(json.dumps(summary), encoding="utf-8")
