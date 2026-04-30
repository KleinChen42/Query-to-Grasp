"""Collect paper/demo-ready artifacts into one captioned figure pack."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.export_utils import write_json  # noqa: E402


@dataclass(frozen=True)
class ArtifactSpec:
    """One source artifact to copy into the paper pack."""

    label: str
    source_path: Path
    category: str
    caption: str


DEFAULT_ARTIFACTS = (
    ArtifactSpec(
        label="implemented_architecture",
        source_path=Path("docs/architecture_query_to_grasp.md"),
        category="notes",
        caption=(
            "Implemented architecture diagram and artifact map for the current "
            "single-view, multi-view fusion, target-selection, and policy "
            "diagnostic pipeline."
        ),
    ),
    ArtifactSpec(
        label="single_view_and_ambiguity_ablation",
        source_path=Path("outputs/h200_60071_paper_baseline/outputs/paper_ablation_table.md"),
        category="tables",
        caption=(
            "HF single-view and ambiguity benchmark table. Shows that exact-object "
            "queries have one candidate on average and that ambiguity increases "
            "candidate count only modestly."
        ),
    ),
    ArtifactSpec(
        label="corrected_multiview_clip_ablation",
        source_path=Path(
            "outputs/h200_60071_reobserve_policy_v2/outputs/"
            "fusion_comparison_table_tabletop3_hf_reobserve_v2.md"
        ),
        category="tables",
        caption=(
            "Corrected single-view vs tabletop_3 fusion ablation. Shows the "
            "no-CLIP/with-CLIP comparison after the camera convention fix, "
            "including re-observation policy metrics."
        ),
    ),
    ArtifactSpec(
        label="extrinsic_convention_report",
        source_path=Path("outputs/h200_60071_extrinsic_convention/extrinsic_convention_report.md"),
        category="geometry",
        caption=(
            "Camera convention comparison. Demonstrates that applying an "
            "OpenCV-to-OpenGL camera-frame conversion before cam2world_gl reduces "
            "cross-view spread."
        ),
    ),
    ArtifactSpec(
        label="corrected_cross_view_geometry",
        source_path=Path("outputs/h200_60071_tabletop3_cvfix/cross_view_geometry_report_tabletop3_cvfix.md"),
        category="geometry",
        caption=(
            "Post-fix cross-view geometry sanity report. Confirms that corrected "
            "tabletop_3 world coordinates are geometrically consistent."
        ),
    ),
    ArtifactSpec(
        label="corrected_memory_diagnostics",
        source_path=Path(
            "outputs/h200_60071_reobserve_policy_v2/outputs/"
            "multiview_fusion_tabletop3_hf_no_clip_reobserve_v2/memory_diagnostics.md"
        ),
        category="diagnostics",
        caption=(
            "Post-fix memory diagnostics and merge-distance sweep. Shows compact "
            "multi-view object memory after fixing the camera convention."
        ),
    ),
    ArtifactSpec(
        label="selection_trace_example",
        source_path=Path("outputs/h200_60071_selection_trace_red_cube_seed0/selection_trace.md"),
        category="traces",
        caption=(
            "Selection trace for red cube seed 0. Explains the selected memory "
            "object through label votes, confidence, views, and deterministic "
            "tie-breaks."
        ),
    ),
    ArtifactSpec(
        label="reobserve_decision_example",
        source_path=Path("outputs/h200_60071_reobserve_smoke/reobserve_decision.json"),
        category="policy",
        caption=(
            "Rule-based re-observation decision example. Shows whether another "
            "view is requested, why, and which view ids are suggested."
        ),
    ),
    ArtifactSpec(
        label="reobserve_policy_report",
        source_path=Path(
            "outputs/h200_60071_reobserve_policy_v2/outputs/"
            "reobserve_policy_report_tabletop3_hf_reobserve_v2.md"
        ),
        category="policy",
        caption=(
            "Aggregate re-observation policy diagnostics. Summarizes trigger "
            "rate, reason counts, per-query policy behavior, and example runs."
        ),
    ),
    ArtifactSpec(
        label="ambiguity_fusion_stress_seed012",
        source_path=Path(
            "outputs/h200_60071_ambiguity_tabletop3_seed012/outputs/"
            "fusion_comparison_table_ambiguity_tabletop3_hf_seed012.md"
        ),
        category="tables",
        caption=(
            "Ambiguity-query corrected tabletop_3 fusion stress test for seeds 0-2. "
            "Shows that broader queries increase memory fragmentation and that "
            "CLIP raises selected confidence while reducing policy trigger rate."
        ),
    ),
    ArtifactSpec(
        label="ambiguity_reobserve_policy_seed012",
        source_path=Path(
            "outputs/h200_60071_ambiguity_tabletop3_seed012/outputs/"
            "reobserve_policy_report_ambiguity_tabletop3_hf_seed012.md"
        ),
        category="policy",
        caption=(
            "Per-query re-observation policy report for the ambiguity fusion "
            "stress test across seeds 0-2, including trigger reasons and examples."
        ),
    ),
    ArtifactSpec(
        label="paper_milestone_log",
        source_path=Path("docs/paper_milestone_log.md"),
        category="notes",
        caption="Running paper-oriented milestone log with quantitative findings and next decisions.",
    ),
    ArtifactSpec(
        label="paper_draft_outline",
        source_path=Path("docs/paper_draft_outline.md"),
        category="notes",
        caption="Current paper outline with claims, experiment plan, limitations, and next milestones.",
    ),
    ArtifactSpec(
        label="fused_memory_grasp_targeted_summary",
        source_path=Path(
            "outputs/h200_60071_multiview_memory_grasp_point_targeted/"
            "compact_broad_no_clip/benchmark_summary.json"
        ),
        category="grasp",
        caption=(
            "Targeted fused-memory grasp-point validation. Compact broad tabletop_3 "
            "no-CLIP run reaches 20/20 simulated pick success using memory_grasp_world_xyz."
        ),
    ),
    ArtifactSpec(
        label="fused_memory_grasp_targeted_rows",
        source_path=Path(
            "outputs/h200_60071_multiview_memory_grasp_point_targeted/"
            "compact_broad_no_clip/benchmark_rows.csv"
        ),
        category="grasp",
        caption="Per-run targeted fused-memory grasp rows, including pick target source diagnostics.",
    ),
    ArtifactSpec(
        label="fused_memory_grasp_ablation_no_clip",
        source_path=Path(
            "outputs/h200_60071_multiview_memory_grasp_point_ablation_seed01234/"
            "tabletop_no_clip/benchmark_summary.json"
        ),
        category="grasp",
        caption=(
            "Accepted compact tabletop_3 no-CLIP fused-memory grasp benchmark summary "
            "with pick_success_rate = 1.0000."
        ),
    ),
    ArtifactSpec(
        label="fused_memory_grasp_ablation_closed_loop",
        source_path=Path(
            "outputs/h200_60071_multiview_memory_grasp_point_ablation_seed01234/"
            "closed_loop_no_clip/benchmark_summary.json"
        ),
        category="grasp",
        caption=(
            "Accepted compact closed-loop no-CLIP fused-memory grasp benchmark summary "
            "with closed-loop resolution and downstream pick metrics."
        ),
    ),
    ArtifactSpec(
        label="stackcube_pick_smoke_no_clip_summary",
        source_path=Path(
            "outputs/h200_60071_stackcube_single_view_red_cube_seed01234/"
            "no_clip/benchmark_summary.json"
        ),
        category="grasp",
        caption=(
            "StackCube-v1 query-driven red cube no-CLIP pick-only smoke. "
            "Shows task-specific grasp detection without claiming stack placement."
        ),
    ),
    ArtifactSpec(
        label="stackcube_pick_smoke_with_clip_summary",
        source_path=Path(
            "outputs/h200_60071_stackcube_single_view_red_cube_seed01234/"
            "with_clip/benchmark_summary.json"
        ),
        category="grasp",
        caption=(
            "StackCube-v1 query-driven red cube with-CLIP pick-only smoke. "
            "Reports pick success separately from task success."
        ),
    ),
    ArtifactSpec(
        label="stackcube_pick_smoke_no_clip_rows",
        source_path=Path(
            "outputs/h200_60071_stackcube_single_view_red_cube_seed01234/"
            "no_clip/benchmark_rows.csv"
        ),
        category="grasp",
        caption="Per-run StackCube-v1 no-CLIP pick-only smoke rows.",
    ),
    ArtifactSpec(
        label="stackcube_pick_smoke_with_clip_rows",
        source_path=Path(
            "outputs/h200_60071_stackcube_single_view_red_cube_seed01234/"
            "with_clip/benchmark_rows.csv"
        ),
        category="grasp",
        caption="Per-run StackCube-v1 with-CLIP pick-only smoke rows.",
    ),
    ArtifactSpec(
        label="cross_task_sim_pick_table",
        source_path=Path("outputs/h200_60071_cross_task_sim_pick_report/cross_task_pick_table.md"),
        category="tables",
        caption=(
            "Cross-task simulated pick comparison table spanning PickCube-v1 "
            "and the StackCube-v1 pick-only smoke."
        ),
    ),
    ArtifactSpec(
        label="full_ambiguity_grasp_comparison",
        source_path=Path(
            "outputs/h200_60071_multiview_memory_grasp_point_full_ambiguity_seed01234/"
            "reports/full_ambiguity_grasp_comparison.md"
        ),
        category="grasp",
        caption=(
            "Full ambiguity PickCube-v1 fused-memory grasp comparison. Shows "
            "55-run tabletop_3 and closed-loop pick metrics for no-CLIP and with-CLIP."
        ),
    ),
    ArtifactSpec(
        label="stackcube_broader_pick_summary",
        source_path=Path(
            "outputs/h200_60071_stackcube_broader_pick_seed0_19/"
            "reports/stackcube_broader_summary.md"
        ),
        category="grasp",
        caption=(
            "StackCube-v1 broader pick-only validation summary across seeds 0-19, "
            "including single-view and tabletop_3 results."
        ),
    ),
    ArtifactSpec(
        label="overnight_followup_summary",
        source_path=Path("outputs/h200_60071_overnight_followup_queue/reports/overnight_summary.md"),
        category="grasp",
        caption=(
            "Overnight H200 follow-up summary covering full PickCube ambiguity "
            "grasp validation and StackCube closed-loop diagnostics."
        ),
    ),
    ArtifactSpec(
        label="stackcube_task_guard_expanded_summary",
        source_path=Path(
            "outputs/h200_60071_stackcube_task_guard_expanded_seed0_49/"
            "reports/stackcube_guard_expanded_summary.md"
        ),
        category="grasp",
        caption=(
            "Expanded StackCube-v1 task-aware guard validation summary across "
            "50 seeds, including pick-stage counts and target-source checks."
        ),
    ),
    ArtifactSpec(
        label="stackcube_task_guard_expanded_tabletop_summary",
        source_path=Path(
            "outputs/h200_60071_stackcube_task_guard_expanded_seed0_49/"
            "tabletop_no_clip/benchmark_summary.json"
        ),
        category="grasp",
        caption=(
            "StackCube-v1 expanded task-aware grasp target guard validation for "
            "tabletop_3 no-CLIP seeds 0-49."
        ),
    ),
    ArtifactSpec(
        label="stackcube_task_guard_expanded_closed_loop_summary",
        source_path=Path(
            "outputs/h200_60071_stackcube_task_guard_expanded_seed0_49/"
            "closed_loop_no_clip/benchmark_summary.json"
        ),
        category="grasp",
        caption=(
            "StackCube-v1 expanded task-aware grasp target guard validation for "
            "closed-loop no-CLIP seeds 0-49."
        ),
    ),
    ArtifactSpec(
        label="stackcube_task_guard_expanded_pickcube_regression_summary",
        source_path=Path(
            "outputs/h200_60071_stackcube_task_guard_expanded_seed0_49/"
            "pickcube_regression_no_clip/benchmark_summary.json"
        ),
        category="grasp",
        caption=(
            "PickCube-v1 regression check showing that the StackCube-specific guard "
            "does not disturb memory_grasp_world_xyz refined picking."
        ),
    ),
    ArtifactSpec(
        label="stackcube_task_guard_expanded_cross_task_table",
        source_path=Path(
            "outputs/h200_60071_stackcube_task_guard_expanded_seed0_49/"
            "reports/cross_task_pick_table.md"
        ),
        category="tables",
        caption=(
            "Cross-task simulated pick table refreshed after the StackCube task-aware "
            "multi-view grasp target guard expanded validation."
        ),
    ),
    ArtifactSpec(
        label="stackcube_task_guard_expanded_failure_report",
        source_path=Path(
            "outputs/h200_60071_stackcube_task_guard_expanded_seed0_49/"
            "reports/stackcube_guard_failure_report.md"
        ),
        category="grasp",
        caption=(
            "StackCube-v1 expanded failure taxonomy showing wrong fused grasp "
            "observations and closed-loop third-object absorption as residual limits."
        ),
    ),
    ArtifactSpec(
        label="stackcube_task_guard_expanded_failure_rows",
        source_path=Path(
            "outputs/h200_60071_stackcube_task_guard_expanded_seed0_49/"
            "reports/stackcube_guard_failure_report.csv"
        ),
        category="grasp",
        caption="Per-failure StackCube-v1 expanded rows for paper limitation analysis.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact paper/demo artifact pack.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "paper_figure_pack")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help=(
            "Extra artifact as LABEL=PATH or LABEL=PATH::CATEGORY::CAPTION. "
            "Repeat to add multiple files."
        ),
    )
    parser.add_argument("--skip-defaults", action="store_true", help="Only include artifacts passed with --artifact.")
    parser.add_argument("--skip-missing", action="store_true", help="Write a partial pack instead of failing on missing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = build_artifact_specs(extra_specs=args.artifact, include_defaults=not args.skip_defaults)
    try:
        manifest = build_paper_figure_pack(
            specs=specs,
            output_dir=args.output_dir,
            skip_missing=args.skip_missing,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote paper figure pack: {args.output_dir}")
    print(f"  README:   {args.output_dir / 'README.md'}")
    print(f"  Manifest: {args.output_dir / 'manifest.json'}")
    print(f"  Included: {manifest['included_count']} / {manifest['total_count']}")
    return 0


def build_artifact_specs(extra_specs: list[str], include_defaults: bool = True) -> list[ArtifactSpec]:
    """Return default plus user-provided artifact specs."""

    specs = list(DEFAULT_ARTIFACTS) if include_defaults else []
    specs.extend(parse_artifact_spec(item) for item in extra_specs)
    if not specs:
        raise ValueError("No artifacts requested. Use defaults or pass --artifact.")
    return specs


def parse_artifact_spec(value: str) -> ArtifactSpec:
    """Parse ``LABEL=PATH`` or ``LABEL=PATH::CATEGORY::CAPTION``."""

    if "=" not in value:
        raise ValueError(f"Invalid artifact spec {value!r}; expected LABEL=PATH.")
    label, rest = value.split("=", 1)
    label = label.strip()
    parts = [part.strip() for part in rest.split("::")]
    path_text = parts[0] if parts else ""
    if not label or not path_text:
        raise ValueError(f"Invalid artifact spec {value!r}; expected non-empty LABEL and PATH.")
    category = parts[1] if len(parts) >= 2 and parts[1] else "extra"
    caption = parts[2] if len(parts) >= 3 and parts[2] else f"User-provided artifact: {label}."
    return ArtifactSpec(label=label, source_path=Path(path_text), category=category, caption=caption)


def build_paper_figure_pack(
    specs: list[ArtifactSpec],
    output_dir: Path,
    skip_missing: bool = False,
) -> dict[str, Any]:
    """Copy artifacts, write a manifest, and render a captioned README."""

    entries: list[dict[str, Any]] = []
    missing: list[ArtifactSpec] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        source = resolve_path(spec.source_path)
        exists = source.exists()
        copied_path: Path | None = None
        if exists:
            copied_path = copy_artifact(source, spec=spec, output_dir=output_dir)
        else:
            missing.append(spec)
        entries.append(
            {
                "label": spec.label,
                "category": spec.category,
                "caption": spec.caption,
                "source_path": str(spec.source_path),
                "resolved_source_path": str(source),
                "exists": exists,
                "copied_path": None if copied_path is None else copied_path.relative_to(output_dir).as_posix(),
            }
        )

    if missing and not skip_missing:
        lines = ["Missing required artifact(s):"]
        lines.extend(f"- {spec.label}: {spec.source_path}" for spec in missing)
        lines.append("Rerun the missing experiments or pass --skip-missing to build a partial pack.")
        raise FileNotFoundError("\n".join(lines))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "total_count": len(entries),
        "included_count": sum(1 for entry in entries if entry["exists"]),
        "missing_count": sum(1 for entry in entries if not entry["exists"]),
        "entries": entries,
    }
    write_json(manifest, output_dir / "manifest.json")
    (output_dir / "README.md").write_text(render_pack_readme(manifest), encoding="utf-8")
    return manifest


def copy_artifact(source: Path, spec: ArtifactSpec, output_dir: Path) -> Path:
    """Copy one artifact into its category directory with a stable name."""

    category_dir = output_dir / slug(spec.category)
    category_dir.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix or ".txt"
    destination = category_dir / f"{slug(spec.label)}{suffix}"
    shutil.copy2(source, destination)
    return destination


def render_pack_readme(manifest: dict[str, Any]) -> str:
    """Render a captioned README for the paper figure pack."""

    lines = [
        "# Query-to-Grasp Paper Figure Pack",
        "",
        "This folder collects the current paper/demo support artifacts into one place.",
        "Generated files are copies; source artifacts remain in their original output directories.",
        "",
        "## Included Artifacts",
        "",
        "| label | category | copied file | caption |",
        "| --- | --- | --- | --- |",
    ]
    for entry in manifest["entries"]:
        if not entry["exists"]:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_table_cell(entry["label"]),
                    escape_table_cell(entry["category"]),
                    f"`{entry['copied_path']}`",
                    escape_table_cell(entry["caption"]),
                ]
            )
            + " |"
        )

    missing = [entry for entry in manifest["entries"] if not entry["exists"]]
    if missing:
        lines.extend(["", "## Missing Artifacts", "", "| label | source |", "| --- | --- |"])
        for entry in missing:
            lines.append(f"| {escape_table_cell(entry['label'])} | `{entry['source_path']}` |")

    lines.extend(
        [
            "",
            "## Suggested Paper Use",
            "",
            "- Use the implemented architecture note as the method/pipeline diagram source.",
            "- Use the corrected multi-view CLIP ablation table as the main quantitative table.",
            "- Use the extrinsic convention and cross-view geometry reports as system validation evidence.",
            "- Use the selection trace as a qualitative example for explainable target selection.",
            "- Use the re-observation policy report as open-loop uncertainty-policy evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def resolve_path(path: Path) -> Path:
    """Resolve relative paths against the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def slug(value: str) -> str:
    """Return a filesystem-safe lowercase slug."""

    text = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
    return text or "artifact"


def escape_table_cell(value: Any) -> str:
    """Escape Markdown table separators."""

    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    raise SystemExit(main())
