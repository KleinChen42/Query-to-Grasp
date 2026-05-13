"""Generate RAS tables and figures from the completed Sunday H200 runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs") / "ras_sunday_results_summary_latest"
DEFAULT_TABLE_DIR = Path("paper_ras") / "tables"
DEFAULT_FIGURE_DIR = Path("paper_ras") / "figures"


@dataclass(frozen=True)
class SummarySpec:
    label: str
    env: str
    mode: str
    target_source: str
    place_source: str
    primary_metric: str
    paths: tuple[Path, ...]
    claim_boundary: str


EXTERNAL_SPECS: tuple[SummarySpec, ...] = (
    SummarySpec("PickCube box center", "PickCube-v1", "pick", "box_center_depth", "none", "pick", (
        Path("outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508/pickcube_box_center_depth"),
        Path("outputs/h200_60071_ras_external_crop_seed200_399_20260509/pickcube_box_center_depth"),
    ), "crop target-source diagnostic"),
    SummarySpec("PickCube crop median", "PickCube-v1", "pick", "crop_median", "none", "pick", (
        Path("outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508/pickcube_crop_median"),
        Path("outputs/h200_60071_ras_external_crop_seed200_399_20260509/pickcube_crop_median"),
    ), "crop target-source diagnostic"),
    SummarySpec("PickCube crop top-surface", "PickCube-v1", "pick", "crop_top_surface", "none", "pick", (
        Path("outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508/pickcube_crop_top_surface"),
        Path("outputs/h200_60071_ras_external_crop_seed200_399_20260509/pickcube_crop_top_surface"),
    ), "crop target-source diagnostic"),
    SummarySpec("PickCube oracle", "PickCube-v1", "pick", "oracle_object_pose", "none", "pick", (
        Path("outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508/pickcube_oracle_object_pose"),
        Path("outputs/h200_60071_ras_external_crop_seed200_399_20260509/pickcube_oracle_object_pose"),
    ), "privileged diagnostic upper bound"),
    SummarySpec("StackCube pick box center", "StackCube-v1", "pick-only", "box_center_depth", "none", "pick", (
        Path("outputs/h200_60071_exp_a_200_seed_freeze_v2_20260508/stackcube_pick_box_center_depth"),
        Path("outputs/h200_60071_ras_external_crop_seed200_399_20260509/stackcube_pickonly_box_center_depth"),
    ), "crop target-source diagnostic"),
    SummarySpec("StackCube pick crop median", "StackCube-v1", "pick-only", "crop_median", "none", "pick", (
        Path("outputs/h200_60071_exp_a_200_seed_freeze_v2_20260508/stackcube_pick_crop_median"),
        Path("outputs/h200_60071_ras_external_crop_seed200_399_20260509/stackcube_pickonly_crop_median"),
    ), "crop target-source diagnostic"),
    SummarySpec("StackCube pick crop top-surface", "StackCube-v1", "pick-only", "crop_top_surface", "none", "pick", (
        Path("outputs/h200_60071_exp_a_200_seed_freeze_v2_20260508/stackcube_pick_crop_top_surface"),
        Path("outputs/h200_60071_ras_external_crop_seed200_399_20260509/stackcube_pickonly_crop_top_surface"),
    ), "crop target-source diagnostic"),
    SummarySpec("StackCube pick oracle", "StackCube-v1", "pick-only", "oracle_object_pose", "none", "pick", (
        Path("outputs/h200_60071_exp_a_200_seed_freeze_v2_20260508/stackcube_pick_oracle_object_pose"),
    ), "privileged diagnostic upper bound"),
    SummarySpec("StackCube pick-place box center", "StackCube-v1", "pick-place", "box_center_depth", "predicted_place_object", "task", (
        Path("outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508/stackcube_pickplace_box_center_depth"),
    ), "non-oracle reference-place diagnostic"),
    SummarySpec("StackCube pick-place crop median", "StackCube-v1", "pick-place", "crop_median", "predicted_place_object", "task", (
        Path("outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508/stackcube_pickplace_crop_median"),
    ), "non-oracle reference-place diagnostic"),
    SummarySpec("StackCube pick-place crop top-surface", "StackCube-v1", "pick-place", "crop_top_surface", "predicted_place_object", "task", (
        Path("outputs/h200_60071_exp_a_200_seed_freeze_gpu4_7_resume_20260508/stackcube_pickplace_crop_top_surface"),
        Path("outputs/h200_60071_ras_external_crop_seed200_399_20260509/stackcube_pickplace_crop_top_surface_pred_green"),
    ), "non-oracle reference-place diagnostic"),
)


NOISY_SPECS: tuple[SummarySpec, ...] = (
    SummarySpec("PickCube pick noise 1cm", "PickCube-v1", "noisy pick", "oracle_object_pose + 1cm pick noise", "none", "pick", (
        Path("outputs/h200_60071_noisy_oracle_pick_pickcube_1cm_seed0_49"),
        Path("outputs/h200_60071_ras_sunday_queue_20260509/noisy_pickcube_pick_1cm"),
    ), "privileged sensitivity probe"),
    SummarySpec("PickCube pick noise 2cm", "PickCube-v1", "noisy pick", "oracle_object_pose + 2cm pick noise", "none", "pick", (
        Path("outputs/h200_60071_noisy_oracle_pick_pickcube_2cm_seed0_49"),
        Path("outputs/h200_60071_ras_sunday_queue_20260509/noisy_pickcube_pick_2cm"),
    ), "privileged sensitivity probe"),
    SummarySpec("PickCube pick noise 5cm", "PickCube-v1", "noisy pick", "oracle_object_pose + 5cm pick noise", "none", "pick", (
        Path("outputs/h200_60071_noisy_oracle_pick_pickcube_5cm_seed0_49"),
        Path("outputs/h200_60071_ras_sunday_queue_20260509/noisy_pickcube_pick_5cm"),
    ), "privileged sensitivity probe"),
    SummarySpec("StackCube pick noise 1cm", "StackCube-v1", "noisy pick", "oracle_object_pose + 1cm pick noise", "none", "pick", (
        Path("outputs/h200_60071_noisy_oracle_pick_stackcube_1cm_seed0_49"),
        Path("outputs/h200_60071_ras_sunday_queue_20260509/noisy_stackcube_pick_1cm"),
    ), "privileged sensitivity probe"),
    SummarySpec("StackCube pick noise 2cm", "StackCube-v1", "noisy pick", "oracle_object_pose + 2cm pick noise", "none", "pick", (
        Path("outputs/h200_60071_noisy_oracle_pick_stackcube_2cm_seed0_49"),
        Path("outputs/h200_60071_ras_sunday_queue_20260509/noisy_stackcube_pick_2cm"),
    ), "privileged sensitivity probe"),
    SummarySpec("StackCube pick noise 5cm", "StackCube-v1", "noisy pick", "oracle_object_pose + 5cm pick noise", "none", "pick", (
        Path("outputs/h200_60071_noisy_oracle_pick_stackcube_5cm_seed0_49"),
        Path("outputs/h200_60071_ras_sunday_queue_20260509/noisy_stackcube_pick_5cm"),
    ), "privileged sensitivity probe"),
    SummarySpec("StackCube place noise 1cm", "StackCube-v1", "noisy place", "oracle_object_pose", "oracle_cubeB_pose + 1cm place noise", "task", (
        Path("outputs/h200_60071_noisy_oracle_place_stackcube_1cm_seed0_49"),
    ), "privileged sensitivity probe"),
    SummarySpec("StackCube place noise 2cm", "StackCube-v1", "noisy place", "oracle_object_pose", "oracle_cubeB_pose + 2cm place noise", "task", (
        Path("outputs/h200_60071_noisy_oracle_place_stackcube_2cm_seed0_49"),
        Path("outputs/h200_60071_ras_sunday_queue_20260509/noisy_stackcube_place_2cm"),
    ), "privileged sensitivity probe"),
    SummarySpec("StackCube place noise 5cm", "StackCube-v1", "noisy place", "oracle_object_pose", "oracle_cubeB_pose + 5cm place noise", "task", (
        Path("outputs/h200_60071_noisy_oracle_place_stackcube_5cm_seed0_49"),
        Path("outputs/h200_60071_ras_sunday_queue_20260509/noisy_stackcube_place_5cm"),
    ), "privileged sensitivity probe"),
)


YCB_SPECS: tuple[SummarySpec, ...] = (
    SummarySpec("PickSingleYCB box center", "PickSingleYCB-v1", "pick", "box_center_depth", "none", "pick", (
        Path("outputs/h200_60071_ras_sunday_queue_20260509/ycb_box_center_depth"),
    ), "non-cube target-source diagnostic"),
    SummarySpec("PickSingleYCB crop median", "PickSingleYCB-v1", "pick", "crop_median", "none", "pick", (
        Path("outputs/h200_60071_ras_sunday_queue_20260509/ycb_crop_median"),
    ), "non-cube target-source diagnostic"),
    SummarySpec("PickSingleYCB crop top-surface", "PickSingleYCB-v1", "pick", "crop_top_surface", "none", "pick", (
        Path("outputs/h200_60071_ras_sunday_queue_20260509/ycb_crop_top_surface"),
    ), "non-cube target-source diagnostic"),
    SummarySpec("PickSingleYCB oracle", "PickSingleYCB-v1", "pick", "oracle_object_pose", "none", "pick", (
        Path("outputs/h200_60071_ras_sunday_queue_20260509/ycb_oracle_object_pose"),
    ), "privileged non-cube executor-feasibility probe"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    external = [summarize(spec, "external_crop") for spec in EXTERNAL_SPECS]
    noisy = [summarize(spec, "noisy_oracle") for spec in NOISY_SPECS]
    ycb = [summarize(spec, "ycb_noncube") for spec in YCB_SPECS]
    all_rows = external + noisy + ycb

    write_csv(external, args.output_dir / "ras_external_crop_freeze_summary.csv")
    write_csv(noisy, args.output_dir / "ras_noisy_oracle_freeze_summary.csv")
    write_csv(ycb, args.output_dir / "ras_ycb_noncube_freeze_summary.csv")
    write_json({"rows": all_rows}, args.output_dir / "ras_sunday_results_summary.json")
    (args.output_dir / "ras_sunday_results_summary.md").write_text(render_markdown(external, noisy, ycb), encoding="utf-8")

    write_table_bundle(external, args.table_dir / "table_external_crop_200_with_ci")
    write_table_bundle(noisy, args.table_dir / "table_noisy_oracle_with_ci")
    write_table_bundle(ycb, args.table_dir / "table_noncube_gate_with_ci")
    draw_external_crop(external, args.figure_dir / "ras_crop_baseline_ci")
    draw_noisy(noisy, args.figure_dir / "ras_noisy_oracle_sensitivity")
    draw_ycb(ycb, args.figure_dir / "ras_ycb_noncube_ladder")
    print(f"Wrote RAS Sunday summaries to {args.output_dir}")
    return 0


def summarize(spec: SummarySpec, group: str) -> dict[str, Any]:
    accum = {
        "total_runs": 0,
        "failed_runs": 0,
        "pick_success_count": 0,
        "place_success_count": 0,
        "task_success_count": 0,
        "target_available_count": 0,
        "grasp_attempted_count": 0,
        "place_attempted_count": 0,
    }
    stage_counts: dict[str, int] = {}
    present_paths: list[str] = []
    for path in spec.paths:
        summary_path = path / "benchmark_summary.json"
        if not summary_path.exists():
            continue
        present_paths.append(str(summary_path))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = summary["aggregate_metrics"]
        total = int(summary.get("total_runs", metrics.get("total_runs", 0)) or 0)
        accum["total_runs"] += total
        accum["failed_runs"] += int(metrics.get("failed_runs", 0) or 0)
        accum["pick_success_count"] += round(float(metrics.get("pick_success_rate", 0.0) or 0.0) * total)
        accum["place_success_count"] += round(float(metrics.get("place_success_rate", 0.0) or 0.0) * total)
        accum["task_success_count"] += round(float(metrics.get("task_success_rate", 0.0) or 0.0) * total)
        accum["target_available_count"] += round(float(metrics.get("fraction_with_3d_target", 0.0) or 0.0) * total)
        accum["grasp_attempted_count"] += round(float(metrics.get("grasp_attempted_rate", 0.0) or 0.0) * total)
        accum["place_attempted_count"] += round(float(metrics.get("place_attempted_rate", 0.0) or 0.0) * total)
        for key in ("pick_stage_counts", "place_stage_counts"):
            for stage, count in (metrics.get(key) or {}).items():
                if stage != "success":
                    stage_counts[stage] = stage_counts.get(stage, 0) + int(count)
    if not present_paths:
        raise FileNotFoundError(f"No benchmark_summary.json found for {spec.label}")
    n = accum["total_runs"]
    count_key = {"pick": "pick_success_count", "place": "place_success_count", "task": "task_success_count"}[spec.primary_metric]
    primary_count = accum[count_key]
    ci_low, ci_high = wilson_interval(primary_count, n)
    row = {
        "group": group,
        "label": spec.label,
        "env": spec.env,
        "mode": spec.mode,
        "target_source": spec.target_source,
        "place_source": spec.place_source,
        "primary_metric": spec.primary_metric,
        "success_count": primary_count,
        "n": n,
        "success_rate": rate(primary_count, n),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "failed_runs": accum["failed_runs"],
        "target_available": rate(accum["target_available_count"], n),
        "grasp_attempted": rate(accum["grasp_attempted_count"], n),
        "pick_success_rate": rate(accum["pick_success_count"], n),
        "place_attempted": rate(accum["place_attempted_count"], n),
        "place_success_rate": rate(accum["place_success_count"], n),
        "task_success_rate": rate(accum["task_success_count"], n),
        "main_failure": max(stage_counts.items(), key=lambda item: item[1])[0] if stage_counts else "none",
        "claim_boundary": spec.claim_boundary,
        "artifact_sources": ";".join(present_paths),
    }
    return row


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_table_bundle(rows: list[dict[str, Any]], stem: Path) -> None:
    write_csv(rows, stem.with_suffix(".csv"))
    stem.with_suffix(".tex").write_text(render_latex(rows, stem.name), encoding="utf-8")


def render_latex(rows: list[dict[str, Any]], table_name: str) -> str:
    if table_name == "table_external_crop_200_with_ci":
        visible = ["env", "mode", "target_source", "n", "primary_metric", "success_ci", "main_failure"]
    elif table_name == "table_noisy_oracle_with_ci":
        visible = ["env", "mode", "target_source", "place_source", "n", "primary_metric", "success_ci"]
    elif table_name == "table_noncube_gate_with_ci":
        visible = ["env", "target_source", "n", "target_available", "grasp_attempted", "success_ci", "claim_boundary"]
    else:
        visible = list(rows[0].keys())
    enriched = []
    for row in rows:
        copied = dict(row)
        copied["success_ci"] = f"{row['success_rate']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        enriched.append(copied)
    lines = [
        "\\begin{tabular}{" + "l" * len(visible) + "}",
        "\\toprule",
        " & ".join(latex_escape(display_header(field)) for field in visible) + r" \\",
        "\\midrule",
    ]
    for row in enriched:
        lines.append(" & ".join(format_latex(row[field]) for field in visible) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def format_latex(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return latex_escape(str(value))


def display_header(field: str) -> str:
    return {
        "env": "Environment",
        "mode": "Mode",
        "target_source": "Target source",
        "place_source": "Place source",
        "n": "N",
        "primary_metric": "Metric",
        "success_ci": "Success (95% CI)",
        "main_failure": "Main failure",
        "target_available": "Target avail.",
        "grasp_attempted": "Attempt",
        "claim_boundary": "Claim boundary",
    }.get(field, field.replace("_", " ").title())


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def render_markdown(external: list[dict[str, Any]], noisy: list[dict[str, Any]], ycb: list[dict[str, Any]]) -> str:
    lines = [
        "# RAS Sunday Results Summary",
        "",
        "This summary is generated from lightweight H200 artifacts only.",
        "",
        "## External Crop Baseline",
        "",
        "| Env | Mode | Target source | N | Success | 95% CI | Main failure |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in external:
        lines.append(f"| {row['env']} | {row['mode']} | `{row['target_source']}` | {row['n']} | {row['success_rate']:.3f} | [{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {row['main_failure']} |")
    lines.extend(["", "## Noisy Oracle", "", "| Env | Mode | Target source | Place source | N | Success | 95% CI |", "| --- | --- | --- | --- | ---: | ---: | --- |"])
    for row in noisy:
        lines.append(f"| {row['env']} | {row['mode']} | `{row['target_source']}` | `{row['place_source']}` | {row['n']} | {row['success_rate']:.3f} | [{row['ci_low']:.3f}, {row['ci_high']:.3f}] |")
    lines.extend(["", "## PickSingleYCB Non-Cube Diagnostic", "", "| Target source | N | Pick success | 95% CI | Main failure |", "| --- | ---: | ---: | --- | --- |"])
    for row in ycb:
        lines.append(f"| `{row['target_source']}` | {row['n']} | {row['success_rate']:.3f} | [{row['ci_low']:.3f}, {row['ci_high']:.3f}] | {row['main_failure']} |")
    lines.append("")
    return "\n".join(lines)


def draw_external_crop(rows: list[dict[str, Any]], stem: Path) -> None:
    groups = [
        ("PickCube pick", lambda r: r["env"] == "PickCube-v1"),
        ("StackCube pick", lambda r: r["env"] == "StackCube-v1" and r["mode"] == "pick-only"),
        ("StackCube task", lambda r: r["env"] == "StackCube-v1" and r["mode"] == "pick-place"),
    ]
    sources = ["box_center_depth", "crop_median", "crop_top_surface", "oracle_object_pose"]
    draw_grouped_bars(rows, groups, sources, stem, "External crop target-source baseline")


def draw_ycb(rows: list[dict[str, Any]], stem: Path) -> None:
    groups = [("PickSingleYCB pick", lambda r: True)]
    sources = ["box_center_depth", "crop_median", "crop_top_surface", "oracle_object_pose"]
    draw_grouped_bars(rows, groups, sources, stem, "PickSingleYCB non-cube target-source diagnostic")


def draw_grouped_bars(rows: list[dict[str, Any]], groups: list[tuple[str, Any]], sources: list[str], stem: Path, title: str) -> None:
    import matplotlib.pyplot as plt

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
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    width = 0.18
    xs = range(len(groups))
    for idx, source in enumerate(sources):
        values = []
        lows = []
        highs = []
        for _, pred in groups:
            matches = [row for row in rows if row["target_source"] == source and pred(row)]
            if not matches:
                values.append(float("nan"))
                lows.append(0.0)
                highs.append(0.0)
                continue
            row = matches[0]
            values.append(row["success_rate"])
            lows.append(max(0.0, row["success_rate"] - row["ci_low"]))
            highs.append(max(0.0, row["ci_high"] - row["success_rate"]))
        positions = [x + (idx - 1.5) * width for x in xs]
        ax.bar(positions, values, width=width, label=labels[source], color=colors[source], edgecolor="#111827", linewidth=0.4)
        ax.errorbar(positions, values, yerr=[lows, highs], fmt="none", ecolor="#111827", elinewidth=0.8, capsize=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([label for label, _ in groups])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def draw_noisy(rows: list[dict[str, Any]], stem: Path) -> None:
    import matplotlib.pyplot as plt

    series = [
        ("PickCube pick", "PickCube-v1", "noisy pick", "#2563eb"),
        ("StackCube pick", "StackCube-v1", "noisy pick", "#dc2626"),
        ("StackCube place", "StackCube-v1", "noisy place", "#16a34a"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for label, env, mode, color in series:
        points = [row for row in rows if row["env"] == env and row["mode"] == mode]
        points = sorted(points, key=lambda row: parse_noise(row["target_source"] + " " + row["place_source"]))
        xs = [parse_noise(row["target_source"] + " " + row["place_source"]) for row in points]
        ys = [row["success_rate"] for row in points]
        yerr = [[row["success_rate"] - row["ci_low"] for row in points], [row["ci_high"] - row["success_rate"] for row in points]]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=1.8, capsize=3, label=label, color=color)
    ax.set_xlabel("Target perturbation (cm)")
    ax.set_ylabel("Primary success rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=220)
    plt.close(fig)


def parse_noise(text: str) -> int:
    for value in (1, 2, 5):
        if f"{value}cm" in text:
            return value
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
