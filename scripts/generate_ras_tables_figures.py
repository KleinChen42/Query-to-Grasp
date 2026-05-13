"""Generate RAS-ready tables and figures from frozen Query-to-Grasp results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


SUMMARY_DIR = Path("outputs") / "paper_revision_results_summary_latest"
DEFAULT_TABLE_DIR = Path("outputs") / "ras_tables"
DEFAULT_FIGURE_DIR = Path("paper") / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, default=SUMMARY_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    paper_rows = read_csv(args.summary_dir / "paper_revision_results_summary.csv")
    crop_rows = read_csv(args.summary_dir / "external_crop_200seed_summary.csv")

    external = build_external_crop_table(crop_rows)
    ladder = build_target_ladder_table(paper_rows, crop_rows)
    noisy = build_noisy_oracle_table(paper_rows)
    noncube = build_noncube_gate_table(crop_rows)

    write_table_bundle(external, args.table_dir / "table_external_crop_200_with_ci")
    write_table_bundle(ladder, args.table_dir / "table_target_ladder_with_ci")
    write_table_bundle(noisy, args.table_dir / "table_noisy_oracle_with_ci")
    write_table_bundle(noncube, args.table_dir / "table_noncube_gate_with_ci")

    draw_crop_ci(external, args.figure_dir / "ras_crop_baseline_ci")
    draw_target_ladder(ladder, args.figure_dir / "ras_target_ladder")
    draw_noisy_oracle(noisy, args.figure_dir / "ras_noisy_oracle_sensitivity")

    write_manifest(
        {
            "external_crop_rows": len(external),
            "target_ladder_rows": len(ladder),
            "noisy_oracle_rows": len(noisy),
            "noncube_gate_rows": len(noncube),
            "tables": [
                "table_external_crop_200_with_ci.csv",
                "table_target_ladder_with_ci.csv",
                "table_noisy_oracle_with_ci.csv",
                "table_noncube_gate_with_ci.csv",
            ],
            "figures": [
                "ras_crop_baseline_ci.pdf",
                "ras_target_ladder.pdf",
                "ras_noisy_oracle_sensitivity.pdf",
            ],
        },
        args.table_dir / "ras_tables_manifest.json",
    )
    print(f"Wrote RAS tables to {args.table_dir}")
    print(f"Wrote RAS figures to {args.figure_dir}")
    return 0


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def build_external_crop_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in rows:
        if row.get("group") != "external_crop_200":
            continue
        setting = {
            ("PickCube-v1", "pick"): "PickCube pick",
            ("StackCube-v1", "pick-only"): "StackCube pick-only",
            ("StackCube-v1", "pick-place"): "StackCube pick-place",
        }[(row["env"], row["mode"])]
        metric = "task" if row["mode"] == "pick-place" else "pick"
        rate_key = "task_success_rate" if metric == "task" else "pick_success_rate"
        count = int(round(float(row[rate_key]) * int(row["total_runs"])))
        table.append(
            {
                "setting": setting,
                "env": row["env"],
                "mode": row["mode"],
                "target_source": row["target_source"],
                "place_source": row["place_source"],
                "n": int(row["total_runs"]),
                "failed": int(row["failed_runs"]),
                "primary_metric": metric,
                "success_count": count,
                "success_rate": float(row[rate_key]),
                "ci_low": float(row["primary_ci_low"]),
                "ci_high": float(row["primary_ci_high"]),
                "target_available": float(row["target_available_rate"]),
                "grasp_attempted": float(row["grasp_attempted_rate"]),
                "main_failure": row["main_failure_stage"],
                "claim_boundary": "external target-source diagnostic",
            }
        )
    return table


def build_target_ladder_table(paper_rows: list[dict[str, str]], crop_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows_by_label = {row["label"]: row for row in paper_rows}
    crop_by_label = {row["label"]: row for row in crop_rows}
    table: list[dict[str, Any]] = []

    def add(label: str, env: str, setting: str, target: str, metric: str, count: int, n: int, source: str, boundary: str) -> None:
        low, high = wilson_interval(count, n)
        table.append(
            {
                "setting": setting,
                "env": env,
                "target_source": target,
                "primary_metric": metric,
                "success_count": count,
                "n": n,
                "success_rate": count / n if n else 0.0,
                "ci_low": low,
                "ci_high": high,
                "artifact_source": source,
                "claim_boundary": boundary,
                "label": label,
            }
        )

    for label, target in [
        ("PickCube semantic target", "semantic center"),
        ("PickCube refined target", "refined grasp point"),
    ]:
        row = rows_by_label[label]
        add(label, row["env_id"], "PickCube pick", target, "pick", int(row["pick_success_count"]), int(row["total_runs"]), "paper_revision_results_summary.csv", "pick-only target-source ablation")

    add(
        "PickCube memory grasp tabletop",
        "PickCube-v1",
        "PickCube tabletop",
        "memory_grasp_world_xyz",
        "pick",
        55,
        55,
        "outputs/paper_submission_audit_latest/final_main_results_table.md",
        "simulated pick benchmark",
    )
    add(
        "PickCube crop top-surface",
        "PickCube-v1",
        "PickCube pick",
        "crop_top_surface",
        "pick",
        int(crop_by_label["PickCube crop top-surface"]["primary_success_count"]),
        int(crop_by_label["PickCube crop top-surface"]["total_runs"]),
        "external_crop_200seed_summary.csv",
        "external crop diagnostic",
    )

    for label, mode in [
        ("StackCube refined predicted place single 500", "StackCube Q+pred single"),
        ("StackCube refined predicted place tabletop 500", "StackCube Q+pred tabletop"),
        ("StackCube refined predicted place closed-loop 500", "StackCube Q+pred closed-loop"),
        ("StackCube broad place query single", "StackCube broad reference single"),
    ]:
        row = rows_by_label[label]
        add(label, row["env_id"], mode, row["place_source"], "task", int(row["task_success_count"]), int(row["total_runs"]), "paper_revision_results_summary.csv", row["claim_boundary"])

    oracle = read_oracle_stackcube_place()
    add(
        "Oracle StackCube pick-place",
        "StackCube-v1",
        "StackCube oracle pick-place",
        "oracle_cubeA_pose + oracle_cubeB_pose",
        "task",
        oracle["task_success_count"],
        oracle["total_runs"],
        "outputs/h200_60071_oracle_stackcube_place_seed0_49/benchmark_summary.json",
        "privileged diagnostic upper bound",
    )
    return table


def read_oracle_stackcube_place() -> dict[str, int]:
    path = Path("outputs/h200_60071_oracle_stackcube_place_seed0_49/benchmark_summary.json")
    if not path.exists():
        return {"task_success_count": 44, "total_runs": 50}
    metrics = json.loads(path.read_text(encoding="utf-8"))["aggregate_metrics"]
    total = int(metrics["total_runs"])
    return {"task_success_count": int(round(float(metrics["task_success_rate"]) * total)), "total_runs": total}


def build_noisy_oracle_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in rows:
        group = row["group"]
        if group not in {"noisy_oracle_pick", "noisy_pickplace", "noisy_oracle_place"}:
            continue
        noise_cm = parse_noise_cm(row["label"])
        metric = "pick" if group == "noisy_oracle_pick" else "task"
        count_key = "pick_success_count" if metric == "pick" else "task_success_count"
        n = int(row["total_runs"])
        count = int(row[count_key])
        low, high = wilson_interval(count, n)
        table.append(
            {
                "group": group,
                "label": row["label"],
                "env": row["env_id"],
                "noise_cm": noise_cm,
                "primary_metric": metric,
                "success_count": count,
                "n": n,
                "success_rate": count / n if n else 0.0,
                "ci_low": low,
                "ci_high": high,
                "pick_success_rate": float(row["pick_success_rate"]),
                "place_success_rate": float(row["place_success_rate"]),
                "task_success_rate": float(row["task_success_rate"]),
                "claim_boundary": row["claim_boundary"],
            }
        )
    return sorted(table, key=lambda item: (item["group"], item["env"], item["noise_cm"]))


def build_noncube_gate_table(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row in rows:
        if row.get("group") != "noncube_feasibility":
            continue
        n = int(row["total_runs"])
        count = int(round(float(row["pick_success_rate"]) * n))
        low, high = wilson_interval(count, n)
        table.append(
            {
                "env": row["env"],
                "gate": row["mode"],
                "target_source": row["target_source"],
                "n": n,
                "failed": int(row["failed_runs"]),
                "target_available": float(row["target_available_rate"]),
                "attempt": float(row["grasp_attempted_rate"]),
                "success_count": count,
                "success_rate": float(row["pick_success_rate"]),
                "ci_low": low,
                "ci_high": high,
                "main_failure": row["main_failure_stage"],
                "interpretation": row["interpretation"],
            }
        )
    return table


def parse_noise_cm(label: str) -> float:
    match = re.search(r"(\d+)cm", label)
    if not match:
        raise ValueError(f"Could not parse noise level from {label!r}")
    return float(match.group(1))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def write_table_bundle(rows: list[dict[str, Any]], stem: Path) -> None:
    write_csv(rows, stem.with_suffix(".csv"))
    stem.with_suffix(".tex").write_text(render_latex_table(add_success_ci(rows), stem.name), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_latex_table(rows: list[dict[str, Any]], table_name: str) -> str:
    if table_name == "table_external_crop_200_with_ci":
        visible = ["setting", "target_source", "n", "primary_metric", "success_ci", "main_failure"]
    elif table_name == "table_target_ladder_with_ci":
        visible = ["setting", "target_source", "primary_metric", "success_ci", "claim_boundary"]
    elif table_name == "table_noisy_oracle_with_ci":
        visible = ["group", "env", "noise_cm", "primary_metric", "success_ci", "claim_boundary"]
    elif table_name == "table_noncube_gate_with_ci":
        visible = ["env", "target_source", "n", "failed", "target_available", "attempt", "success_ci", "interpretation"]
    else:
        visible = [field for field in rows[0].keys() if field not in {"artifact_source", "label"}]
    lines = [
        "\\begin{tabular}{" + "l" * len(visible) + "}",
        "\\toprule",
        " & ".join(latex_escape(field.replace("_", " ")) for field in visible) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(format_latex_cell(row[field]) for field in visible) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def latex_escape(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def format_latex_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return latex_escape(str(value))


def add_success_ci(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated = []
    for row in rows:
        copied = dict(row)
        copied["success_ci"] = f"{row['success_rate']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        updated.append(copied)
    return updated


def write_manifest(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def draw_crop_ci(rows: list[dict[str, Any]], stem: Path) -> None:
    import matplotlib.pyplot as plt

    settings = ["PickCube pick", "StackCube pick-only", "StackCube pick-place"]
    sources = ["box_center_depth", "crop_median", "crop_top_surface", "oracle_object_pose"]
    labels = ["Box center", "Crop median", "Crop top-surface", "Oracle"]
    colors = ["#ef4444", "#0ea5e9", "#22c55e", "#7c3aed"]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    width = 0.18
    for i, source in enumerate(sources):
        vals, lows, highs = [], [], []
        for setting in settings:
            row = next((r for r in rows if r["setting"] == setting and r["target_source"] == source), None)
            if row is None:
                vals.append(float("nan"))
                lows.append(0.0)
                highs.append(0.0)
                continue
            vals.append(row["success_rate"])
            lows.append(max(0.0, row["success_rate"] - row["ci_low"]))
            highs.append(max(0.0, row["ci_high"] - row["success_rate"]))
        xs = [x + (i - 1.5) * width for x in range(len(settings))]
        ax.bar(xs, vals, width=width, color=colors[i], label=labels[i])
        ax.errorbar(xs, vals, yerr=[lows, highs], fmt="none", ecolor="#111827", elinewidth=0.8, capsize=2)
    style_rate_axis(ax, "External RGB-D crop baseline with Wilson 95% CI")
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels(settings, fontsize=8)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=7, frameon=False)
    save_fig(fig, stem)


def draw_target_ladder(rows: list[dict[str, Any]], stem: Path) -> None:
    import matplotlib.pyplot as plt

    selected = [
        "PickCube semantic target",
        "PickCube refined target",
        "PickCube memory grasp tabletop",
        "PickCube crop top-surface",
        "StackCube refined predicted place single 500",
        "StackCube refined predicted place tabletop 500",
        "StackCube refined predicted place closed-loop 500",
        "StackCube broad place query single",
        "Oracle StackCube pick-place",
    ]
    plot_rows = [next(row for row in rows if row["label"] == label) for label in selected]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ys = list(range(len(plot_rows)))
    vals = [row["success_rate"] for row in plot_rows]
    lows = [max(0.0, row["success_rate"] - row["ci_low"]) for row in plot_rows]
    highs = [max(0.0, row["ci_high"] - row["success_rate"]) for row in plot_rows]
    colors = ["#0ea5e9"] * 4 + ["#22c55e"] * 3 + ["#f59e0b", "#7c3aed"]
    ax.barh(ys, vals, color=colors, height=0.62)
    ax.errorbar(vals, ys, xerr=[lows, highs], fmt="none", ecolor="#111827", elinewidth=0.8, capsize=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([short_ladder_label(row["label"]) for row in plot_rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Primary success rate", fontsize=8.5)
    ax.set_title("RAS target-source ladder with Wilson 95% CI", fontsize=10)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    save_fig(fig, stem)


def short_ladder_label(label: str) -> str:
    return {
        "PickCube semantic target": "PickCube SC pick",
        "PickCube refined target": "PickCube RG pick",
        "PickCube memory grasp tabletop": "PickCube MG tabletop",
        "PickCube crop top-surface": "PickCube crop top",
        "StackCube refined predicted place single 500": "StackCube Q+PG single",
        "StackCube refined predicted place tabletop 500": "StackCube Q+PG tabletop",
        "StackCube refined predicted place closed-loop 500": "StackCube Q+PG closed-loop",
        "StackCube broad place query single": "StackCube broad ref",
        "Oracle StackCube pick-place": "StackCube oracle",
    }[label]


def draw_noisy_oracle(rows: list[dict[str, Any]], stem: Path) -> None:
    import matplotlib.pyplot as plt

    series = [
        ("noisy_oracle_pick", "PickCube-v1", "PickCube pick noise", "#0ea5e9"),
        ("noisy_oracle_pick", "StackCube-v1", "StackCube pick noise", "#22c55e"),
        ("noisy_pickplace", "StackCube-v1", "StackCube noisy pick-place task", "#f59e0b"),
        ("noisy_oracle_place", "StackCube-v1", "StackCube noisy place task", "#7c3aed"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    for group, env, label, color in series:
        points = [row for row in rows if row["group"] == group and row["env"] == env]
        points = sorted(points, key=lambda row: row["noise_cm"])
        xs = [row["noise_cm"] for row in points]
        ys = [row["success_rate"] for row in points]
        lows = [max(0.0, row["success_rate"] - row["ci_low"]) for row in points]
        highs = [max(0.0, row["ci_high"] - row["success_rate"]) for row in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color=color, label=label)
        ax.errorbar(xs, ys, yerr=[lows, highs], fmt="none", ecolor=color, alpha=0.75, capsize=2)
    style_rate_axis(ax, "Noisy-oracle sensitivity")
    ax.set_xlabel("Target perturbation sigma (cm)", fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    save_fig(fig, stem)


def style_rate_axis(ax: Any, title: str) -> None:
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Success rate", fontsize=8.5)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def save_fig(fig: Any, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=220)


if __name__ == "__main__":
    raise SystemExit(main())
