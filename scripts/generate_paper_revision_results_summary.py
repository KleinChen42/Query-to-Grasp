"""Generate the frozen paper-revision result summary from benchmark rows."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = Path("outputs") / "paper_revision_results_summary_latest"


@dataclass(frozen=True)
class RunSet:
    label: str
    group: str
    sources: tuple[Path, ...]
    env_id: str
    view_mode: str
    target_mode: str
    pick_source: str
    place_source: str
    claim_boundary: str


PAPER_REVISION_RUNSETS: tuple[RunSet, ...] = (
    RunSet("PickCube oracle pick noise 1cm", "noisy_oracle_pick", (Path("outputs/h200_60071_noisy_oracle_pick_pickcube_1cm_seed0_49"),), "PickCube-v1", "single", "refined", "oracle+noise", "none", "pick sensitivity"),
    RunSet("PickCube oracle pick noise 2cm", "noisy_oracle_pick", (Path("outputs/h200_60071_noisy_oracle_pick_pickcube_2cm_seed0_49"),), "PickCube-v1", "single", "refined", "oracle+noise", "none", "pick sensitivity"),
    RunSet("PickCube oracle pick noise 5cm", "noisy_oracle_pick", (Path("outputs/h200_60071_noisy_oracle_pick_pickcube_5cm_seed0_49"),), "PickCube-v1", "single", "refined", "oracle+noise", "none", "pick sensitivity"),
    RunSet("StackCube oracle pick noise 1cm", "noisy_oracle_pick", (Path("outputs/h200_60071_noisy_oracle_pick_stackcube_1cm_seed0_49"),), "StackCube-v1", "single", "refined", "oracle+noise", "none", "pick sensitivity"),
    RunSet("StackCube oracle pick noise 2cm", "noisy_oracle_pick", (Path("outputs/h200_60071_noisy_oracle_pick_stackcube_2cm_seed0_49"),), "StackCube-v1", "single", "refined", "oracle+noise", "none", "pick sensitivity"),
    RunSet("StackCube oracle pick noise 5cm", "noisy_oracle_pick", (Path("outputs/h200_60071_noisy_oracle_pick_stackcube_5cm_seed0_49"),), "StackCube-v1", "single", "refined", "oracle+noise", "none", "pick sensitivity"),
    RunSet("StackCube noisy pick-place 1cm", "noisy_pickplace", (Path("outputs/h200_60071_noisy_oracle_pickplace_stackcube_1cm_seed0_49"),), "StackCube-v1", "single", "refined", "oracle+pick-noise", "oracle_cubeB_pose", "oracle place bridge"),
    RunSet("StackCube noisy pick-place 2cm", "noisy_pickplace", (Path("outputs/h200_60071_noisy_oracle_pickplace_stackcube_2cm_seed0_49"),), "StackCube-v1", "single", "refined", "oracle+pick-noise", "oracle_cubeB_pose", "oracle place bridge"),
    RunSet("StackCube noisy pick-place 5cm", "noisy_pickplace", (Path("outputs/h200_60071_noisy_oracle_pickplace_stackcube_5cm_seed0_49"),), "StackCube-v1", "single", "refined", "oracle+pick-noise", "oracle_cubeB_pose", "oracle place bridge"),
    RunSet("StackCube noisy oracle place 1cm", "noisy_oracle_place", (Path("outputs/h200_60071_noisy_oracle_place_stackcube_1cm_seed0_49"),), "StackCube-v1", "single", "refined", "oracle_cubeA_pose", "oracle_cubeB_pose+noise", "place sensitivity"),
    RunSet("StackCube noisy oracle place 2cm", "noisy_oracle_place", (Path("outputs/h200_60071_noisy_oracle_place_stackcube_2cm_seed0_49"),), "StackCube-v1", "single", "refined", "oracle_cubeA_pose", "oracle_cubeB_pose+noise", "place sensitivity"),
    RunSet("StackCube noisy oracle place 5cm", "noisy_oracle_place", (Path("outputs/h200_60071_noisy_oracle_place_stackcube_5cm_seed0_49"),), "StackCube-v1", "single", "refined", "oracle_cubeA_pose", "oracle_cubeB_pose+noise", "place sensitivity"),
    RunSet("PickCube semantic target", "target_point_ablation", (Path("outputs/h200_60071_pickcube_semantic_target_ablation_seed0_54"),), "PickCube-v1", "single", "semantic", "world_xyz", "none", "target point ablation"),
    RunSet("PickCube refined target", "target_point_ablation", (Path("outputs/h200_60071_pickcube_refined_target_ablation_seed0_54"),), "PickCube-v1", "single", "refined", "grasp_world_xyz", "none", "target point ablation"),
    RunSet("PickCube with CLIP", "clip_ablation", (Path("outputs/h200_60071_pickcube_with_clip_ablation_seed0_49"),), "PickCube-v1", "single", "refined", "grasp_world_xyz", "none", "clip ablation"),
    RunSet("StackCube with CLIP", "clip_ablation", (Path("outputs/h200_60071_stackcube_with_clip_ablation_seed0_49"),), "StackCube-v1", "single", "refined", "world_xyz", "none", "clip ablation"),
    RunSet("StackCube refined predicted place single 500", "stackcube_predicted_place_500", (Path("outputs/h200_60071_predicted_place_object_seed0_199/single_no_clip"), Path("outputs/h200_60071_predicted_place_object_single_no_clip_seed200_499")), "StackCube-v1", "single", "refined", "refined_query_target", "predicted_place_object_green_cube", "non-oracle bridge"),
    RunSet("StackCube refined predicted place tabletop 500", "stackcube_predicted_place_500", (Path("outputs/h200_60071_predicted_place_object_seed0_199/tabletop_no_clip"), Path("outputs/h200_60071_predicted_place_object_tabletop_no_clip_seed200_499")), "StackCube-v1", "tabletop_3", "refined", "task_guard_selected_object_world_xyz", "predicted_place_object_green_cube", "non-oracle bridge"),
    RunSet("StackCube refined predicted place closed-loop 500", "stackcube_predicted_place_500", (Path("outputs/h200_60071_predicted_place_object_seed0_199/closed_loop_no_clip"), Path("outputs/h200_60071_predicted_place_object_closed_loop_no_clip_seed200_499")), "StackCube-v1", "closed_loop", "refined", "task_guard_selected_object_world_xyz", "predicted_place_object_green_cube", "non-oracle bridge"),
    RunSet("StackCube semantic predicted place single 500", "stackcube_predicted_place_500", (Path("outputs/h200_60071_stackcube_semantic_predicted_place_single_no_clip_seed0_199"), Path("outputs/h200_60071_stackcube_semantic_predicted_place_single_no_clip_seed200_499")), "StackCube-v1", "single", "semantic", "semantic_center", "predicted_place_object_green_cube", "semantic baseline"),
    RunSet("StackCube semantic predicted place tabletop 500", "stackcube_predicted_place_500", (Path("outputs/h200_60071_stackcube_semantic_predicted_place_tabletop_no_clip_seed0_199"), Path("outputs/h200_60071_stackcube_semantic_predicted_place_tabletop_no_clip_seed200_499")), "StackCube-v1", "tabletop_3", "semantic", "semantic_center", "predicted_place_object_green_cube", "semantic baseline"),
    RunSet("StackCube semantic predicted place closed-loop 500", "stackcube_predicted_place_500", (Path("outputs/h200_60071_stackcube_semantic_predicted_place_closed_loop_no_clip_seed0_199"), Path("outputs/h200_60071_stackcube_semantic_predicted_place_closed_loop_no_clip_seed200_499")), "StackCube-v1", "closed_loop", "semantic", "semantic_center", "predicted_place_object_green_cube", "semantic baseline"),
    RunSet("StackCube broad place query single", "reference_query_ablation", (Path("outputs/h200_60071_predicted_place_broad_cube_seed0_199_v2/single_no_clip"),), "StackCube-v1", "single", "refined", "refined_query_target", "predicted_place_object_broad_cube", "reference query ablation"),
    RunSet("StackCube broad place query tabletop", "reference_query_ablation", (Path("outputs/h200_60071_predicted_place_broad_cube_seed0_199_v2/tabletop_no_clip"),), "StackCube-v1", "tabletop_3", "refined", "task_guard_selected_object_world_xyz", "predicted_place_object_broad_cube", "reference query ablation"),
    RunSet("StackCube broad place query closed-loop", "reference_query_ablation", (Path("outputs/h200_60071_predicted_place_broad_cube_seed0_199_v2/closed_loop_no_clip"),), "StackCube-v1", "closed_loop", "refined", "task_guard_selected_object_world_xyz", "predicted_place_object_broad_cube", "reference query ablation"),
    RunSet("PushCube single refined", "task_diversity_target_source", (Path("outputs/h200_60071_pushcube_targetsource_seed0_199/single_refined_no_clip"),), "PushCube-v1", "single", "refined", "refined_query_target", "none", "target-source formation only"),
    RunSet("PushCube tabletop refined", "task_diversity_target_source", (Path("outputs/h200_60071_pushcube_multiview_targetsource_seed0_199/tabletop_refined_no_clip"),), "PushCube-v1", "tabletop_3", "refined", "memory_grasp_world_xyz", "none", "target-source formation only"),
    RunSet("PushCube closed-loop refined", "task_diversity_target_source", (Path("outputs/h200_60071_pushcube_closed_loop_targetsource_seed0_199/closed_loop_refined_no_clip"),), "PushCube-v1", "closed_loop", "refined", "memory_grasp_world_xyz", "none", "target-source formation only"),
    RunSet("LiftPeg target-source", "task_diversity_target_source", (Path("outputs/h200_60071_maniskill_diverse_targetsource_seed0_199/lift_peg_no_clip"),), "LiftPegUpright-v1", "single", "refined", "refined_query_target", "none", "target-source formation only"),
    RunSet("PegInsertion target-source", "task_diversity_target_source", (Path("outputs/h200_60071_maniskill_diverse_targetsource_seed0_199/peg_insertion_no_clip"),), "PegInsertionSide-v1", "single", "refined", "refined_query_target", "none", "target-source formation only"),
    RunSet("StackPyramid target-source", "task_diversity_target_source", (Path("outputs/h200_60071_maniskill_diverse_targetsource_seed0_199/stack_pyramid_no_clip"),), "StackPyramid-v1", "single", "refined", "refined_query_target", "none", "target-source formation only"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strict", action="store_true", help="Fail if any expected runset artifact is missing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(PAPER_REVISION_RUNSETS, strict=args.strict)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(summary, args.output_dir)
    print(f"Wrote paper revision summary: {args.output_dir}")
    return 0


def build_summary(runsets: Iterable[RunSet], strict: bool = False) -> dict[str, Any]:
    rows = []
    missing: list[str] = []
    for runset in runsets:
        loaded_rows: list[dict[str, str]] = []
        for source in runset.sources:
            rows_path = source / "benchmark_rows.csv"
            if not rows_path.exists():
                missing.append(str(rows_path))
                continue
            loaded_rows.extend(load_csv_rows(rows_path))
        if loaded_rows:
            rows.append(summarize_runset(runset, loaded_rows))
    if missing and strict:
        raise FileNotFoundError("Missing paper-revision artifact(s):\n" + "\n".join(missing))
    return {
        "total_runsets": len(rows),
        "missing_artifacts": missing,
        "rows": rows,
    }


def summarize_runset(runset: RunSet, rows: list[dict[str, str]]) -> dict[str, Any]:
    total = len(rows)
    seeds = sorted({int(row["seed"]) for row in rows if str(row.get("seed", "")).isdigit()})
    return {
        "label": runset.label,
        "group": runset.group,
        "env_id": runset.env_id,
        "view_mode": runset.view_mode,
        "target_mode": runset.target_mode,
        "pick_source": runset.pick_source,
        "place_source": runset.place_source,
        "claim_boundary": runset.claim_boundary,
        "total_runs": total,
        "seed_min": seeds[0] if seeds else None,
        "seed_max": seeds[-1] if seeds else None,
        "pick_success_count": count_true(rows, "pick_success"),
        "pick_success_rate": rate(rows, "pick_success"),
        "place_attempted_count": count_true(rows, "place_attempted"),
        "place_attempted_rate": rate(rows, "place_attempted"),
        "place_success_count": count_true(rows, "place_success"),
        "place_success_rate": rate(rows, "place_success"),
        "task_success_count": count_true(rows, "task_success"),
        "task_success_rate": rate(rows, "task_success"),
        "target_source_count": count_target_source(rows),
        "target_source_rate": count_target_source(rows) / total if total else 0.0,
        "top1_changed_count": count_true(rows, "top1_changed_by_rerank"),
        "top1_changed_rate": rate(rows, "top1_changed_by_rerank"),
        "mean_num_detections": mean_float(rows, "num_detections"),
        "mean_num_ranked_candidates": mean_float(rows, "num_ranked_candidates"),
        "sources": [str(source) for source in runset.sources],
    }


def write_outputs(summary: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "paper_revision_results_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(summary["rows"], output_dir / "paper_revision_results_summary.csv")
    (output_dir / "paper_revision_results_summary.md").write_text(render_markdown(summary), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "group",
        "label",
        "env_id",
        "view_mode",
        "target_mode",
        "pick_source",
        "place_source",
        "total_runs",
        "pick_success_count",
        "pick_success_rate",
        "place_attempted_count",
        "place_attempted_rate",
        "place_success_count",
        "place_success_rate",
        "task_success_count",
        "task_success_rate",
        "target_source_count",
        "target_source_rate",
        "top1_changed_count",
        "top1_changed_rate",
        "mean_num_detections",
        "mean_num_ranked_candidates",
        "claim_boundary",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# H200 Paper-Revision Results Summary",
        "",
        "Generated from frozen benchmark rows. Generated outputs are not committed; this file is a paper-table source artifact.",
        "",
        f"- runsets summarized: {summary['total_runsets']}",
        f"- missing artifacts: {len(summary['missing_artifacts'])}",
        "",
    ]
    for group, title in [
        ("noisy_oracle_pick", "Noisy Oracle Pick Sensitivity"),
        ("noisy_pickplace", "Noisy Oracle Pick-Place"),
        ("noisy_oracle_place", "Noisy Oracle Place Sensitivity"),
        ("target_point_ablation", "PickCube Target Point Ablation"),
        ("clip_ablation", "CLIP Reranking Ablation"),
        ("stackcube_predicted_place_500", "StackCube Predicted-Place 500-Seed Ladder"),
        ("reference_query_ablation", "Reference Query Specificity Ablation"),
        ("task_diversity_target_source", "Task-Diversity Target-Source Formation"),
    ]:
        group_rows = [row for row in summary["rows"] if row["group"] == group]
        if not group_rows:
            continue
        lines.extend(["", f"## {title}", "", "| label | n | pick | place attempted | place | task | target source | top1 changed | mean dets |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in group_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["label"],
                        str(row["total_runs"]),
                        fraction(row, "pick_success"),
                        fraction(row, "place_attempted"),
                        fraction(row, "place_success"),
                        fraction(row, "task_success"),
                        fraction(row, "target_source"),
                        fraction(row, "top1_changed"),
                        fmt(row["mean_num_detections"]),
                    ]
                )
                + " |"
            )
    if summary["missing_artifacts"]:
        lines.extend(["", "## Missing Artifacts", ""])
        lines.extend(f"- `{item}`" for item in summary["missing_artifacts"])
    lines.append("")
    return "\n".join(lines)


def fraction(row: dict[str, Any], prefix: str) -> str:
    count = row[f"{prefix}_count"]
    total = row["total_runs"]
    return f"{count}/{total} = {fmt(row[f'{prefix}_rate'])}"


def fmt(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def count_true(rows: list[dict[str, str]], key: str) -> int:
    return sum(1 for row in rows if as_bool(row.get(key)))


def rate(rows: list[dict[str, str]], key: str) -> float:
    if not rows:
        return 0.0
    return count_true(rows, key) / len(rows)


def mean_float(rows: list[dict[str, str]], key: str) -> float:
    if not rows or all(key not in row for row in rows):
        return 0.0
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(key) or 0.0))
        except ValueError:
            continue
    return sum(values) / len(values) if values else 0.0


def count_target_source(rows: list[dict[str, str]]) -> int:
    if any("has_3d_target" in row for row in rows):
        return count_true(rows, "has_3d_target")
    count = 0
    for row in rows:
        if str(row.get("pick_target_source") or "").strip():
            count += 1
            continue
        if str(row.get("selected_world_xyz") or "").strip():
            count += 1
            continue
        if str(row.get("selected_grasp_world_xyz") or "").strip():
            count += 1
    return count


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


if __name__ == "__main__":
    raise SystemExit(main())
