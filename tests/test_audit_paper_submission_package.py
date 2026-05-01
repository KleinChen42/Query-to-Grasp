from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_paper_submission_package import (
    ResultSpec,
    audit_paper_submission_package,
    build_final_result_rows,
    check_claim_boundaries,
)


def write_summary(path: Path, pick: float, task: float, total: int = 50, place: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "total_runs": total,
        "failed_runs": 0,
        "pick_success_rate": pick,
        "task_success_rate": task,
    }
    if place is not None:
        metrics["place_success_rate"] = place
    path.write_text(
        json.dumps(
            {
                "env_id": "StackCube-v1",
                "total_runs": total,
                "aggregate_metrics": metrics,
            }
        ),
        encoding="utf-8",
    )


def write_required_artifacts(root: Path) -> None:
    required = [
        "paper/main.tex",
        "paper/references.bib",
        "docs/paper_draft_outline.md",
        "docs/paper_manuscript_draft.md",
        "docs/paper_multitask_sim_grasp_section.md",
        "outputs/paper_figure_pack_latest/manifest.json",
        "outputs/demo_video_pack_latest/manifest.json",
        "outputs/supplemental_video_latest/manifest.json",
        "outputs/supplemental_video_latest/query_to_grasp_supplemental_video.mp4",
    ]
    for rel in required:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = "No real-robot execution is claimed.\n"
        path.write_bytes(b"video") if path.suffix == ".mp4" else path.write_text(text, encoding="utf-8")


def test_build_final_result_rows_validates_expected_metrics(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    write_summary(summary, pick=0.72, place=0.52, task=0.48, total=50)
    spec = ResultSpec(
        benchmark="Query bridge",
        summary_path=summary,
        mode="closed-loop",
        target_source="query target",
        place_source="oracle_cubeB_pose",
        claim_boundary="Partial bridge.",
        expected_pick=0.72,
        expected_place=0.52,
        expected_task=0.48,
        expected_total_runs=50,
    )

    rows, issues = build_final_result_rows((spec,), project_root=tmp_path)

    assert issues == []
    assert rows[0]["pick_success_rate"] == 0.72
    assert rows[0]["place_success_rate"] == 0.52
    assert rows[0]["place_source"] == "oracle_cubeB_pose"


def test_audit_writes_pass_report_and_final_tables(tmp_path: Path) -> None:
    write_required_artifacts(tmp_path)
    summary = tmp_path / "outputs" / "accepted" / "benchmark_summary.json"
    write_summary(summary, pick=1.0, task=0.0, total=5)
    spec = ResultSpec(
        benchmark="Accepted",
        summary_path=Path("outputs/accepted/benchmark_summary.json"),
        mode="demo",
        target_source="memory_grasp_world_xyz",
        place_source="none",
        claim_boundary="Simulated pick.",
        expected_pick=1.0,
        expected_task=0.0,
        expected_total_runs=5,
    )

    report = audit_paper_submission_package(
        output_dir=tmp_path / "audit",
        project_root=tmp_path,
        specs=(spec,),
    )

    assert report["status"] == "pass"
    assert (tmp_path / "audit" / "audit_report.md").exists()
    assert (tmp_path / "audit" / "final_main_results_table.md").exists()
    assert "memory_grasp_world_xyz" in (tmp_path / "audit" / "final_main_results_table.md").read_text(encoding="utf-8")


def test_claim_boundary_flags_positive_unsupported_claim(tmp_path: Path) -> None:
    path = tmp_path / "paper" / "main.tex"
    path.parent.mkdir(parents=True)
    path.write_text("We report real-robot success on StackCube.\n", encoding="utf-8")

    issues = check_claim_boundaries(tmp_path)

    assert issues
    assert "real" in issues[0]
    assert "success" in issues[0]
