"""Summarize the 200-seed external crop baseline and non-cube feasibility gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs") / "paper_revision_results_summary_latest"


@dataclass(frozen=True)
class RunSpec:
    label: str
    group: str
    path: Path
    env: str
    mode: str
    target_source: str
    place_source: str
    primary_metric: str
    interpretation: str


RUNS: tuple[RunSpec, ...] = (
    RunSpec("PickCube box center", "external_crop_200", Path("outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508/pickcube_box_center_depth"), "PickCube-v1", "pick", "box_center_depth", "none", "pick_success", "single-pixel center-depth baseline"),
    RunSpec("PickCube crop median", "external_crop_200", Path("outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508/pickcube_crop_median"), "PickCube-v1", "pick", "crop_median", "none", "pick_success", "crop depth aggregation"),
    RunSpec("PickCube crop top-surface", "external_crop_200", Path("outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508/pickcube_crop_top_surface"), "PickCube-v1", "pick", "crop_top_surface", "none", "pick_success", "workspace-aware crop heuristic"),
    RunSpec("PickCube oracle", "external_crop_200", Path("outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508/pickcube_oracle_object_pose"), "PickCube-v1", "pick", "oracle_object_pose", "none", "pick_success", "privileged upper bound"),
    RunSpec("StackCube pick box center", "external_crop_200", Path("outputs/h200_60071_exp_a_200_seed_freeze_v2_20260508/stackcube_pick_box_center_depth"), "StackCube-v1", "pick-only", "box_center_depth", "none", "pick_success", "single-pixel center-depth baseline"),
    RunSpec("StackCube pick crop median", "external_crop_200", Path("outputs/h200_60071_exp_a_200_seed_freeze_v2_20260508/stackcube_pick_crop_median"), "StackCube-v1", "pick-only", "crop_median", "none", "pick_success", "crop depth aggregation"),
    RunSpec("StackCube pick crop top-surface", "external_crop_200", Path("outputs/h200_60071_exp_a_200_seed_freeze_v2_20260508/stackcube_pick_crop_top_surface"), "StackCube-v1", "pick-only", "crop_top_surface", "none", "pick_success", "workspace-aware crop heuristic"),
    RunSpec("StackCube pick oracle", "external_crop_200", Path("outputs/h200_60071_exp_a_200_seed_freeze_v2_20260508/stackcube_pick_oracle_object_pose"), "StackCube-v1", "pick-only", "oracle_object_pose", "none", "pick_success", "privileged upper bound"),
    RunSpec("StackCube pick-place box center", "external_crop_200", Path("outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508/stackcube_pickplace_box_center_depth"), "StackCube-v1", "pick-place", "box_center_depth", "predicted_place_object", "task_success", "non-oracle pick and explicit reference place"),
    RunSpec("StackCube pick-place crop median", "external_crop_200", Path("outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508/stackcube_pickplace_crop_median"), "StackCube-v1", "pick-place", "crop_median", "predicted_place_object", "task_success", "non-oracle pick and explicit reference place"),
    RunSpec("StackCube pick-place crop top-surface", "external_crop_200", Path("outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508/stackcube_pickplace_crop_top_surface"), "StackCube-v1", "pick-place", "crop_top_surface", "predicted_place_object", "task_success", "non-oracle pick and explicit reference place"),
    RunSpec("PickSingleYCB oracle", "noncube_feasibility", Path("outputs/h200_60071_noncube_feasibility_gpu4_7_20260508/ycb_oracle_object_pose"), "PickSingleYCB-v1", "oracle gate", "oracle_object_pose", "none", "pick_success", "runtime compatibility gate"),
    RunSpec("PickSingleYCB crop top-surface", "noncube_feasibility", Path("outputs/h200_60071_noncube_feasibility_gpu4_7_20260508/ycb_crop_top_surface"), "PickSingleYCB-v1", "crop gate", "crop_top_surface", "none", "pick_success", "runtime compatibility gate"),
    RunSpec("PickClutterYCB oracle", "noncube_feasibility", Path("outputs/h200_60071_noncube_feasibility_gpu4_7_20260508/clutter_ycb_oracle_object_pose"), "PickClutterYCB-v1", "oracle gate", "oracle_object_pose", "none", "pick_success", "runtime compatibility gate"),
    RunSpec("PickClutterYCB crop top-surface", "noncube_feasibility", Path("outputs/h200_60071_noncube_feasibility_gpu4_7_20260508/clutter_ycb_crop_top_surface"), "PickClutterYCB-v1", "crop gate", "crop_top_surface", "none", "pick_success", "runtime compatibility gate"),
    RunSpec("PickSingleEGAD oracle", "noncube_feasibility", Path("outputs/h200_60071_noncube_feasibility_gpu4_7_20260508/egad_oracle_object_pose"), "PickSingleEGAD-v1", "oracle gate", "oracle_object_pose", "none", "pick_success", "runtime compatibility gate"),
    RunSpec("PickSingleEGAD crop top-surface", "noncube_feasibility", Path("outputs/h200_60071_noncube_feasibility_gpu4_7_20260508/egad_crop_top_surface"), "PickSingleEGAD-v1", "crop gate", "crop_top_surface", "none", "pick_success", "runtime compatibility gate"),
    RunSpec("LiftPeg oracle", "noncube_feasibility", Path("outputs/h200_60071_noncube_feasibility_gpu4_7_20260508/liftpeg_oracle_object_pose"), "LiftPegUpright-v1", "oracle gate", "oracle_object_pose", "none", "pick_success", "executor mismatch diagnostic"),
    RunSpec("LiftPeg crop top-surface", "noncube_feasibility", Path("outputs/h200_60071_noncube_feasibility_gpu4_7_20260508/liftpeg_crop_top_surface"), "LiftPegUpright-v1", "crop gate", "crop_top_surface", "none", "pick_success", "executor mismatch diagnostic"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=Path("paper") / "figures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [summarize_run(spec) for spec in RUNS]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    write_json(rows, args.output_dir / "external_crop_200seed_summary.json")
    write_csv(rows, args.output_dir / "external_crop_200seed_summary.csv")
    (args.output_dir / "external_crop_200seed_summary.md").write_text(render_markdown(rows), encoding="utf-8")
    draw_figure(rows, args.figure_dir / "external_crop_200seed_results")
    print(f"Wrote external crop/non-cube summary to {args.output_dir}")
    return 0


def summarize_run(spec: RunSpec) -> dict[str, Any]:
    summary_path = spec.path / "benchmark_summary.json"
    rows_path = spec.path / "benchmark_rows.csv"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("aggregate_metrics", {})
    total = int(metrics.get("total_runs", summary.get("total_runs", 0)) or 0)
    failed = int(metrics.get("failed_runs", 0) or 0)
    pick_success = float(metrics.get("pick_success_rate", metrics.get("is_grasped_rate", 0.0)) or 0.0)
    place_success = float(metrics.get("place_success_rate", 0.0) or 0.0)
    task_success = float(metrics.get("task_success_rate", 0.0) or 0.0)
    target_available = float(metrics.get("fraction_with_3d_target", 0.0) or 0.0)
    grasp_attempted = float(metrics.get("grasp_attempted_rate", 0.0) or 0.0)
    place_attempted = float(metrics.get("place_attempted_rate", 0.0) or 0.0)
    primary_rate = {"pick_success": pick_success, "task_success": task_success, "place_success": place_success}[spec.primary_metric]
    primary_count = round(primary_rate * total)
    ci_low, ci_high = wilson_interval(primary_count, total)
    return {
        "label": spec.label,
        "group": spec.group,
        "env": spec.env,
        "mode": spec.mode,
        "target_source": spec.target_source,
        "place_source": spec.place_source,
        "primary_metric": spec.primary_metric,
        "primary_success_count": primary_count,
        "primary_success_rate": primary_rate,
        "primary_ci_low": ci_low,
        "primary_ci_high": ci_high,
        "total_runs": total,
        "failed_runs": failed,
        "target_available_rate": target_available,
        "grasp_attempted_rate": grasp_attempted,
        "pick_success_rate": pick_success,
        "place_attempted_rate": place_attempted,
        "place_success_rate": place_success,
        "task_success_rate": task_success,
        "main_failure_stage": main_failure_stage(metrics),
        "interpretation": spec.interpretation,
        "summary_path": str(summary_path),
        "rows_path": str(rows_path),
        "rows_present": rows_path.exists(),
        "target_error_fields_available": target_error_fields_available(rows_path),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def main_failure_stage(metrics: dict[str, Any]) -> str:
    stages: dict[str, int] = {}
    for key in ("pick_stage_counts", "place_stage_counts"):
        for stage, count in (metrics.get(key) or {}).items():
            if stage == "success":
                continue
            stages[stage] = stages.get(stage, 0) + int(count)
    if not stages:
        return "none"
    return max(stages.items(), key=lambda item: item[1])[0]


def target_error_fields_available(rows_path: Path) -> bool:
    if not rows_path.exists():
        return False
    with rows_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration:
            return False
    fields = set(header)
    target_fields = {"target_world_xyz", "target_xyz", "pick_target_xyz"}
    oracle_fields = {"oracle_object_pose_xyz", "oracle_pick_xyz", "oracle_target_xyz"}
    return bool(fields & target_fields) and bool(fields & oracle_fields)


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# External Crop 200-Seed and Non-Cube Feasibility Summary",
        "",
        "Generated from lightweight H200 benchmark summaries. Pick-only task success is treated as a raw environment flag, not the primary metric.",
        "",
        "## External RGB-D Crop Baseline",
        "",
        "| Env | Mode | Target source | N | Failed | Target avail. | Attempt | Pick | Place | Task/RawEnv | Primary 95% CI | Main failure |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        if row["group"] != "external_crop_200":
            continue
        lines.append(
            "| {env} | {mode} | `{target}` | {n} | {failed} | {target_avail:.3f} | {attempt:.3f} | {pick:.3f} | {place:.3f} | {task:.3f} | [{lo:.3f}, {hi:.3f}] | {failure} |".format(
                env=row["env"],
                mode=row["mode"],
                target=row["target_source"] if row["place_source"] == "none" else f"{row['target_source']} + {row['place_source']}",
                n=row["total_runs"],
                failed=row["failed_runs"],
                target_avail=row["target_available_rate"],
                attempt=row["grasp_attempted_rate"],
                pick=row["pick_success_rate"],
                place=row["place_success_rate"],
                task=row["task_success_rate"],
                lo=row["primary_ci_low"],
                hi=row["primary_ci_high"],
                failure=row["main_failure_stage"],
            )
        )
    lines.extend([
        "",
        "## Non-Cube Feasibility Gate",
        "",
        "| Env | Gate | Target source | N | Failed | Target avail. | Attempt | Pick | RawEnv | Main failure | Interpretation |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for row in rows:
        if row["group"] != "noncube_feasibility":
            continue
        lines.append(
            "| {env} | {mode} | `{target}` | {n} | {failed} | {target_avail:.3f} | {attempt:.3f} | {pick:.3f} | {task:.3f} | {failure} | {interp} |".format(
                env=row["env"],
                mode=row["mode"],
                target=row["target_source"],
                n=row["total_runs"],
                failed=row["failed_runs"],
                target_avail=row["target_available_rate"],
                attempt=row["grasp_attempted_rate"],
                pick=row["pick_success_rate"],
                task=row["task_success_rate"],
                failure=row["main_failure_stage"],
                interp=row["interpretation"],
            )
        )
    if not any(row["target_error_fields_available"] for row in rows):
        lines.extend([
            "",
            "## Target-Error Extraction",
            "",
            "The pulled `benchmark_rows.csv` files do not contain both predicted target XYZ and oracle target XYZ fields.",
            "Therefore this freeze does not fabricate a target-error-vs-success curve; that analysis remains a future derived-output task after target/oracle XYZ logging is standardized.",
        ])
    lines.append("")
    return "\n".join(lines)


def draw_figure(rows: list[dict[str, Any]], output_stem: Path) -> None:
    import matplotlib.pyplot as plt

    ordered = [
        ("PickCube pick", "PickCube-v1", "pick", "pick_success_rate"),
        ("StackCube pick", "StackCube-v1", "pick-only", "pick_success_rate"),
        ("StackCube task", "StackCube-v1", "pick-place", "task_success_rate"),
    ]
    sources = ["box_center_depth", "crop_median", "crop_top_surface", "oracle_object_pose"]
    colors = {
        "box_center_depth": "#ef4444",
        "crop_median": "#0ea5e9",
        "crop_top_surface": "#22c55e",
        "oracle_object_pose": "#7c3aed",
    }
    labels = {
        "box_center_depth": "Box center",
        "crop_median": "Crop median",
        "crop_top_surface": "Crop top-surface",
        "oracle_object_pose": "Oracle",
    }
    fig, ax = plt.subplots(figsize=(7.1, 3.0))
    width = 0.18
    x_positions = list(range(len(ordered)))
    for offset, source in enumerate(sources):
        vals = []
        for _, env, mode, metric in ordered:
            row = next((r for r in rows if r["env"] == env and r["mode"] == mode and r["target_source"] == source), None)
            vals.append(float(row[metric]) if row else 0.0)
        xs = [x + (offset - 1.5) * width for x in x_positions]
        ax.bar(xs, vals, width=width, label=labels[source], color=colors[source])
        for x, value in zip(xs, vals):
            if value > 0:
                ax.text(x, value + 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=6.4)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([item[0] for item in ordered], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Success rate", fontsize=8.5)
    ax.set_title("External RGB-D crop baselines over 200 seeds", fontsize=10)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=7, frameon=False)
    fig.text(
        0.5,
        0.005,
        "Pick-only rows use pick success; StackCube pick-place uses task success with predicted green-cube place target.",
        ha="center",
        fontsize=7,
        color="#475569",
    )
    fig.tight_layout(rect=[0.02, 0.13, 0.98, 0.96])
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
