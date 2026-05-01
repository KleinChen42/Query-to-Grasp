"""Audit the paper submission package and freeze the final results table."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.export_utils import write_json  # noqa: E402


TOLERANCE = 1e-4


@dataclass(frozen=True)
class ResultSpec:
    """One accepted result that should appear in the final paper-facing table."""

    benchmark: str
    summary_path: Path
    mode: str
    target_source: str
    place_source: str
    claim_boundary: str
    expected_pick: float | None = None
    expected_place: float | None = None
    expected_task: float | None = None
    expected_total_runs: int | None = None
    expected_failed_runs: int | None = 0


DEFAULT_RESULT_SPECS = (
    ResultSpec(
        benchmark="PickCube full-query tabletop",
        summary_path=Path("outputs/h200_60071_multiview_memory_grasp_point_full_ambiguity_seed01234/tabletop_no_clip/benchmark_summary.json"),
        mode="tabletop_3 no CLIP",
        target_source="memory_grasp_world_xyz",
        place_source="none",
        claim_boundary="Simulated pick benchmark; no real-robot claim.",
        expected_pick=1.0,
        expected_task=0.14545454545454545,
        expected_total_runs=55,
    ),
    ResultSpec(
        benchmark="PickCube full-query closed-loop",
        summary_path=Path("outputs/h200_60071_multiview_memory_grasp_point_full_ambiguity_seed01234/closed_loop_no_clip/benchmark_summary.json"),
        mode="closed-loop no CLIP",
        target_source="memory_grasp_world_xyz",
        place_source="none",
        claim_boundary="Simulated pick benchmark; closed-loop is diagnostic.",
        expected_pick=1.0,
        expected_task=0.14545454545454545,
        expected_total_runs=55,
    ),
    ResultSpec(
        benchmark="StackCube guarded pick-only tabletop",
        summary_path=Path("outputs/h200_60071_stackcube_task_guard_expanded_seed0_49/tabletop_no_clip/benchmark_summary.json"),
        mode="tabletop_3 no CLIP",
        target_source="task_guard_selected_object_world_xyz",
        place_source="none",
        claim_boundary="Pick-only compatibility; not stack completion.",
        expected_pick=0.62,
        expected_task=0.0,
        expected_total_runs=50,
    ),
    ResultSpec(
        benchmark="StackCube guarded pick-only closed-loop",
        summary_path=Path("outputs/h200_60071_stackcube_task_guard_expanded_seed0_49/closed_loop_no_clip/benchmark_summary.json"),
        mode="closed-loop no CLIP",
        target_source="task_guard_selected_object_world_xyz",
        place_source="none",
        claim_boundary="Pick-only compatibility; closed-loop limitation.",
        expected_pick=0.52,
        expected_task=0.0,
        expected_total_runs=50,
    ),
    ResultSpec(
        benchmark="Oracle pick",
        summary_path=Path("outputs/h200_60071_oracle_pick_ablation/pickcube_oracle/benchmark_summary.json"),
        mode="PickCube oracle",
        target_source="oracle_object_pose",
        place_source="none",
        claim_boundary="Privileged target-source upper bound.",
        expected_pick=1.0,
        expected_task=0.04,
        expected_total_runs=50,
    ),
    ResultSpec(
        benchmark="Oracle pick",
        summary_path=Path("outputs/h200_60071_oracle_pick_ablation/stackcube_oracle/benchmark_summary.json"),
        mode="StackCube oracle",
        target_source="oracle_object_pose",
        place_source="none",
        claim_boundary="Privileged target-source upper bound.",
        expected_pick=0.94,
        expected_task=0.0,
        expected_total_runs=50,
    ),
    ResultSpec(
        benchmark="Oracle pick-place",
        summary_path=Path("outputs/h200_60071_oracle_stackcube_place_seed0_49/benchmark_summary.json"),
        mode="StackCube oracle cubeA to cubeB",
        target_source="oracle_cubeA_pose",
        place_source="oracle_cubeB_pose",
        claim_boundary="Privileged pick-place upper bound.",
        expected_pick=0.94,
        expected_place=0.88,
        expected_task=0.88,
        expected_total_runs=50,
    ),
    ResultSpec(
        benchmark="Query-pick plus oracle-place bridge",
        summary_path=Path("outputs/h200_60071_query_stackcube_place_bridge_seed0_49/single_with_clip/benchmark_summary.json"),
        mode="single-view with CLIP",
        target_source="query-derived grasp_world_xyz",
        place_source="oracle_cubeB_pose",
        claim_boundary="Partial bridge; destination is privileged.",
        expected_pick=0.88,
        expected_place=0.72,
        expected_task=0.72,
        expected_total_runs=50,
    ),
    ResultSpec(
        benchmark="Query-pick plus oracle-place bridge",
        summary_path=Path("outputs/h200_60071_query_stackcube_place_bridge_seed0_49/tabletop_no_clip/benchmark_summary.json"),
        mode="tabletop_3 no CLIP",
        target_source="query-derived task_guard_selected_object_world_xyz",
        place_source="oracle_cubeB_pose",
        claim_boundary="Partial bridge; destination is privileged.",
        expected_pick=0.62,
        expected_place=0.52,
        expected_task=0.52,
        expected_total_runs=50,
    ),
    ResultSpec(
        benchmark="Query-pick plus oracle-place bridge",
        summary_path=Path("outputs/h200_60071_query_stackcube_place_bridge_seed0_49/closed_loop_no_clip/benchmark_summary.json"),
        mode="closed-loop no CLIP",
        target_source="query-derived task_guard_selected_object_world_xyz",
        place_source="oracle_cubeB_pose",
        claim_boundary="Partial bridge; destination is privileged; closed-loop diagnostic.",
        expected_pick=0.52,
        expected_place=0.48,
        expected_task=0.48,
        expected_total_runs=50,
    ),
)


REQUIRED_ARTIFACTS = (
    Path("paper/main.tex"),
    Path("paper/references.bib"),
    Path("docs/paper_draft_outline.md"),
    Path("docs/paper_manuscript_draft.md"),
    Path("outputs/paper_figure_pack_latest/manifest.json"),
    Path("outputs/demo_video_pack_latest/manifest.json"),
    Path("outputs/supplemental_video_latest/manifest.json"),
    Path("outputs/supplemental_video_latest/query_to_grasp_supplemental_video.mp4"),
)


UNSUPPORTED_PATTERNS = (
    "real-robot success",
    "real robot success",
    "learned controller success",
    "learned grasping success",
    "full non-oracle stackcube stacking completion",
    "fully non-oracle stackcube task completion",
    "language-conditioned stackcube stacking completion",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Query-to-Grasp paper submission artifacts.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "paper_submission_audit_latest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_paper_submission_package(output_dir=args.output_dir)
    print(f"Wrote paper submission audit: {args.output_dir}")
    print(f"  Status: {report['status']}")
    print(f"  Report: {args.output_dir / 'audit_report.md'}")
    print(f"  Table:  {args.output_dir / 'final_main_results_table.md'}")
    return 0 if report["status"] == "pass" else 1


def audit_paper_submission_package(
    output_dir: Path,
    project_root: Path = PROJECT_ROOT,
    specs: tuple[ResultSpec, ...] = DEFAULT_RESULT_SPECS,
) -> dict[str, Any]:
    """Run artifact, metric, and claim-boundary checks and write reports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows, metric_issues = build_final_result_rows(specs=specs, project_root=project_root)
    artifact_issues = check_required_artifacts(project_root)
    claim_issues = check_claim_boundaries(project_root)
    status = "pass" if not metric_issues and not artifact_issues and not claim_issues else "fail"

    write_markdown_table(rows, output_dir / "final_main_results_table.md")
    write_csv_table(rows, output_dir / "final_main_results_table.csv")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "metric_issues": metric_issues,
        "artifact_issues": artifact_issues,
        "claim_issues": claim_issues,
        "result_rows": rows,
    }
    write_json(report, output_dir / "audit_report.json")
    (output_dir / "audit_report.md").write_text(render_audit_report(report), encoding="utf-8")
    return report


def build_final_result_rows(specs: tuple[ResultSpec, ...], project_root: Path = PROJECT_ROOT) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for spec in specs:
        summary_path = resolve_path(spec.summary_path, project_root)
        if not summary_path.exists():
            issues.append(f"Missing summary for {spec.benchmark} / {spec.mode}: {spec.summary_path}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary.get("aggregate_metrics", {})
        row = {
            "benchmark": spec.benchmark,
            "env_id": summary.get("env_id", "unknown"),
            "mode": spec.mode,
            "runs": int(summary.get("total_runs", metrics.get("total_runs", 0))),
            "failed_runs": metrics.get("failed_runs", "n/a"),
            "pick_success_rate": metrics.get("pick_success_rate"),
            "place_success_rate": metrics.get("place_success_rate", "n/a"),
            "task_success_rate": metrics.get("task_success_rate"),
            "target_source": spec.target_source,
            "place_source": spec.place_source,
            "claim_boundary": spec.claim_boundary,
            "source_summary": spec.summary_path.as_posix(),
        }
        rows.append(row)
        issues.extend(validate_row_against_spec(row, spec))
    return rows, issues


def validate_row_against_spec(row: dict[str, Any], spec: ResultSpec) -> list[str]:
    issues: list[str] = []
    if spec.expected_total_runs is not None and row["runs"] != spec.expected_total_runs:
        issues.append(f"{spec.benchmark} / {spec.mode}: expected runs {spec.expected_total_runs}, got {row['runs']}")
    if spec.expected_failed_runs is not None and row["failed_runs"] not in {spec.expected_failed_runs, "n/a", None}:
        issues.append(
            f"{spec.benchmark} / {spec.mode}: expected failed_runs {spec.expected_failed_runs}, got {row['failed_runs']}"
        )
    for key, expected in [
        ("pick_success_rate", spec.expected_pick),
        ("place_success_rate", spec.expected_place),
        ("task_success_rate", spec.expected_task),
    ]:
        if expected is None:
            continue
        actual = row.get(key)
        if actual in {"n/a", None}:
            issues.append(f"{spec.benchmark} / {spec.mode}: missing {key}, expected {expected:.4f}")
            continue
        if abs(float(actual) - expected) > TOLERANCE:
            issues.append(f"{spec.benchmark} / {spec.mode}: expected {key}={expected:.4f}, got {float(actual):.4f}")
    return issues


def check_required_artifacts(project_root: Path = PROJECT_ROOT) -> list[str]:
    issues: list[str] = []
    for path in REQUIRED_ARTIFACTS:
        resolved = resolve_path(path, project_root)
        if not resolved.exists():
            issues.append(f"Missing required artifact: {path.as_posix()}")
        elif resolved.is_file() and resolved.stat().st_size <= 0:
            issues.append(f"Required artifact is empty: {path.as_posix()}")
    return issues


def check_claim_boundaries(project_root: Path = PROJECT_ROOT) -> list[str]:
    issues: list[str] = []
    paths = [
        Path("paper/main.tex"),
        Path("docs/paper_draft_outline.md"),
        Path("docs/paper_manuscript_draft.md"),
        Path("docs/paper_multitask_sim_grasp_section.md"),
    ]
    for path in paths:
        resolved = resolve_path(path, project_root)
        if not resolved.exists():
            continue
        text = resolved.read_text(encoding="utf-8").lower()
        lines = text.splitlines()
        for pattern in UNSUPPORTED_PATTERNS:
            for line_index, line in enumerate(lines):
                if pattern not in line:
                    continue
                context = "\n".join(lines[max(0, line_index - 12) : line_index + 1])
                if is_negated_claim(context):
                    continue
                issues.append(f"Unsupported positive claim in {path.as_posix()}: {pattern}")
    return issues


def is_negated_claim(text_window: str) -> bool:
    negators = (
        "no ",
        "not ",
        "does not ",
        "do not ",
        "must not ",
        "must avoid ",
        "avoid claim",
        "avoid claiming",
        "should not ",
        "without ",
        "rather than ",
        "not as ",
        "not claim",
        "not claiming",
        "does not claim",
        "do not claim",
        "must not claim",
        "must not be claimed",
        "non-claim",
        "non-claims",
        "claim boundary",
        "claim boundaries",
        "unsupported claim",
        "unsupported claims",
        "no real-robot claim",
    )
    return any(negator in text_window for negator in negators)


def write_markdown_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Final Main Results Table",
        "",
        "Paper-facing target-source and manipulation result table. Privileged oracle sources are labeled explicitly.",
        "",
        "| benchmark | env | mode | runs | failed | pick | place | task | target source | place source | claim boundary |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_table_cell(row["benchmark"]),
                    escape_table_cell(row["env_id"]),
                    escape_table_cell(row["mode"]),
                    str(row["runs"]),
                    escape_table_cell(row["failed_runs"]),
                    format_rate(row["pick_success_rate"]),
                    format_rate(row["place_success_rate"]),
                    format_rate(row["task_success_rate"]),
                    f"`{escape_table_cell(row['target_source'])}`",
                    f"`{escape_table_cell(row['place_source'])}`",
                    escape_table_cell(row["claim_boundary"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_table(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "benchmark",
        "env_id",
        "mode",
        "runs",
        "failed_runs",
        "pick_success_rate",
        "place_success_rate",
        "task_success_rate",
        "target_source",
        "place_source",
        "claim_boundary",
        "source_summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_audit_report(report: dict[str, Any]) -> str:
    lines = [
        "# Query-to-Grasp Paper Submission Audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Created at: `{report['created_at']}`",
        f"- Result rows: {len(report['result_rows'])}",
        "",
    ]
    for title, key in [
        ("Metric Issues", "metric_issues"),
        ("Artifact Issues", "artifact_issues"),
        ("Claim Issues", "claim_issues"),
    ]:
        issues = report[key]
        lines.extend([f"## {title}", ""])
        if issues:
            lines.extend(f"- {issue}" for issue in issues)
        else:
            lines.append("- none")
        lines.append("")
    lines.extend(
        [
            "## Generated Files",
            "",
            "- `final_main_results_table.md`",
            "- `final_main_results_table.csv`",
            "- `audit_report.json`",
            "",
        ]
    )
    return "\n".join(lines)


def format_rate(value: Any) -> str:
    if value in {"n/a", None, ""}:
        return "n/a"
    return f"{float(value):.4f}"


def resolve_path(path: Path, project_root: Path = PROJECT_ROOT) -> Path:
    return path if path.is_absolute() else project_root / path


def escape_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
