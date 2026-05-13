"""Generate RAS tables and figures for synthetic RGB-D sensor stress.

The script consumes benchmark directories produced by
``outputs/ras_sensor_stress_v2_20260512`` and writes a compact RAS-facing
summary. It intentionally treats the experiment as a synthetic robustness
proxy rather than real-world validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_INPUT_ROOT = Path("outputs") / "ras_sensor_stress_v2_20260512"
DEFAULT_OUTPUT_ROOT = Path("outputs") / "ras_revision_aggressive_20260511" / "sensor_stress"
DEFAULT_TABLE_DIR = Path("paper_ras") / "tables"
DEFAULT_FIGURE_DIR = Path("paper_ras") / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    long_root = args.input_root / "long"
    if not long_root.exists():
        raise FileNotFoundError(f"Missing long-run directory: {long_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    rows = [summarize_run(path.parent) for path in sorted(long_root.rglob("benchmark_summary.json"))]
    if len(rows) != 25:
        raise ValueError(f"Expected 25 long-run summaries, found {len(rows)} in {long_root}")
    rows.sort(key=lambda row: (row["task_sort"], row["source_sort"], row["condition_sort"]))

    write_csv(rows, args.output_root / "sensor_stress_summary.csv")
    write_csv(rows, args.table_dir / "table_sensor_stress_with_ci.csv")
    write_json({"rows": rows, "num_rows": len(rows), "input_root": str(args.input_root)}, args.output_root / "summary_sensor_stress.json")
    write_markdown(rows, args.output_root / "summary_sensor_stress.md")
    write_latex_table(rows, args.table_dir / "table_sensor_stress_with_ci.tex")
    make_figures(rows, args.figure_dir)
    print(f"Wrote {len(rows)} sensor-stress rows.")


def summarize_run(run_dir: Path) -> dict[str, Any]:
    summary = read_json(run_dir / "benchmark_summary.json")
    rows = read_csv(run_dir / "benchmark_rows.csv")
    metrics = summary["aggregate_metrics"]
    total = int(summary["total_runs"])
    failed = int(metrics.get("failed_runs", 0))
    pick_success = int(round(float(metrics.get("pick_success_rate", 0.0)) * total))
    place_success = int(round(float(metrics.get("place_success_rate", 0.0)) * total))
    task_success = int(round(float(metrics.get("task_success_rate", 0.0)) * total))
    target_available = int(round(float(metrics.get("fraction_with_3d_target", 0.0)) * total))
    attempted = int(round(float(metrics.get("grasp_attempted_rate", 0.0)) * total))

    env_id = str(summary.get("env_id"))
    executor = str(summary.get("pick_executor"))
    source = str(summary.get("grasp_target_mode"))
    place_source = str(summary.get("place_target_source"))
    depth_noise = float(summary.get("depth_noise_std_m", 0.0))
    dropout = float(summary.get("depth_dropout_prob", 0.0))
    task = classify_task(env_id, executor, place_source)
    condition = classify_condition(depth_noise, dropout)
    stage_counts = metrics.get("pick_stage_counts", {})
    main_stage = max(stage_counts.items(), key=lambda item: int(item[1]))[0] if stage_counts else "unknown"

    valid_before = mean_float(row.get("valid_depth_pixels_before") for row in rows)
    valid_after = mean_float(row.get("valid_depth_pixels_after") for row in rows)
    dropped = mean_float(row.get("dropped_depth_pixels") for row in rows)

    pick_lo, pick_hi = wilson_interval(pick_success, total)
    task_lo, task_hi = wilson_interval(task_success, total)
    target_lo, target_hi = wilson_interval(target_available, total)

    return {
        "task": task,
        "env_id": env_id,
        "pick_executor": executor,
        "target_source": source,
        "place_target_source": place_source,
        "condition": condition,
        "depth_noise_std_m": depth_noise,
        "depth_dropout_prob": dropout,
        "n": total,
        "failed_runs": failed,
        "target_available": target_available,
        "target_available_rate": target_available / total if total else 0.0,
        "target_available_ci_low": target_lo,
        "target_available_ci_high": target_hi,
        "grasp_attempted": attempted,
        "grasp_attempted_rate": attempted / total if total else 0.0,
        "pick_success": pick_success,
        "pick_success_rate": pick_success / total if total else 0.0,
        "pick_ci_low": pick_lo,
        "pick_ci_high": pick_hi,
        "place_success": place_success,
        "place_success_rate": place_success / total if total else 0.0,
        "task_success": task_success,
        "task_success_rate": task_success / total if total else 0.0,
        "task_ci_low": task_lo,
        "task_ci_high": task_hi,
        "mean_valid_depth_pixels_before": valid_before,
        "mean_valid_depth_pixels_after": valid_after,
        "mean_dropped_depth_pixels": dropped,
        "mean_num_3d_points": float(metrics.get("mean_num_3d_points", 0.0)),
        "main_pick_stage": main_stage,
        "run_dir": str(run_dir),
        "task_sort": task_sort(task),
        "source_sort": source_sort(source),
        "condition_sort": condition_sort(condition),
    }


def classify_task(env_id: str, executor: str, place_source: str) -> str:
    if env_id == "PickCube-v1":
        return "PickCube pick"
    if env_id == "StackCube-v1" and executor == "sim_pick_place":
        return "StackCube pick-place"
    if env_id == "StackCube-v1":
        return "StackCube pick"
    return f"{env_id} {executor}"


def classify_condition(depth_noise: float, dropout: float) -> str:
    if depth_noise > 0.0:
        return f"depth noise {depth_noise * 1000:.0f} mm"
    if dropout > 0.0:
        return f"depth dropout {dropout * 100:.0f}%"
    return "clean"


def task_sort(task: str) -> int:
    return {"PickCube pick": 0, "StackCube pick": 1, "StackCube pick-place": 2}.get(task, 99)


def source_sort(source: str) -> int:
    return {"box_center_depth": 0, "crop_top_surface": 1}.get(source, 99)


def condition_sort(condition: str) -> int:
    order = {
        "clean": 0,
        "depth noise 5 mm": 1,
        "depth noise 10 mm": 2,
        "depth dropout 10%": 3,
        "depth dropout 20%": 4,
    }
    return order.get(condition, 99)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def mean_float(values: Any) -> float:
    parsed: list[float] = []
    for value in values:
        try:
            if value in (None, ""):
                continue
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    return sum(parsed) / len(parsed) if parsed else 0.0


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    public_keys = [
        "task",
        "target_source",
        "condition",
        "n",
        "failed_runs",
        "target_available_rate",
        "grasp_attempted_rate",
        "pick_success_rate",
        "pick_ci_low",
        "pick_ci_high",
        "task_success_rate",
        "task_ci_low",
        "task_ci_high",
        "mean_valid_depth_pixels_after",
        "mean_num_3d_points",
        "main_pick_stage",
        "run_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=public_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in public_keys})


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# RAS Sensor-Stress Summary",
        "",
        "Synthetic RGB-D degradation proxy. This is not real-world validation.",
        "",
        "| Task | Source | Condition | N | Target avail. | Pick | Task | Main stage |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {task} | {source} | {cond} | {n} | {ta:.3f} | {pick:.3f} | {task_sr:.3f} | {stage} |".format(
                task=row["task"],
                source=row["target_source"],
                cond=row["condition"],
                n=row["n"],
                ta=row["target_available_rate"],
                pick=row["pick_success_rate"],
                task_sr=row["task_success_rate"],
                stage=row["main_pick_stage"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Synthetic RGB-D sensor-stress diagnostic. Noise and dropout are applied before 3D target extraction. Values are success rates with Wilson 95\\% confidence intervals. This is a simulation proxy, not real-world validation.}",
        "\\label{tab:sensor-stress}",
        "\\small",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Task / source & Condition & $N$ & Target avail. & Pick & Task/raw \\\\",
        "\\midrule",
    ]
    current_group = None
    for row in rows:
        group = f"{row['task']} / {format_source(row['target_source'])}"
        if current_group is not None and group != current_group:
            lines.append("\\midrule")
        current_group = group
        lines.append(
            f"{escape_latex(group)} & {escape_latex(row['condition'])} & {row['n']} & "
            f"{rate_ci(row['target_available_rate'], row['target_available_ci_low'], row['target_available_ci_high'])} & "
            f"{rate_ci(row['pick_success_rate'], row['pick_ci_low'], row['pick_ci_high'])} & "
            f"{rate_ci(row['task_success_rate'], row['task_ci_low'], row['task_ci_high'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figures(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    condition_order = ["clean", "depth noise 5 mm", "depth noise 10 mm", "depth dropout 10%", "depth dropout 20%"]
    x = list(range(len(condition_order)))
    colors = {"box_center_depth": "#dc2626", "crop_top_surface": "#2563eb"}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, task in zip(axes, ["PickCube pick", "StackCube pick"], strict=True):
        for source in ["box_center_depth", "crop_top_surface"]:
            series = [find_row(rows, task, source, cond)["pick_success_rate"] for cond in condition_order]
            lower = [
                max(0.0, find_row(rows, task, source, cond)["pick_success_rate"] - find_row(rows, task, source, cond)["pick_ci_low"])
                for cond in condition_order
            ]
            upper = [
                max(0.0, find_row(rows, task, source, cond)["pick_ci_high"] - find_row(rows, task, source, cond)["pick_success_rate"])
                for cond in condition_order
            ]
            ax.errorbar(x, series, yerr=[lower, upper], marker="o", linewidth=1.8, capsize=3, color=colors[source], label=format_source(source))
        ax.set_title(task)
        ax.set_xticks(x, ["clean", "5mm", "10mm", "drop10", "drop20"], rotation=20)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Pick success")
    axes[1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_sensor_stress_pick_success")

    pp_rows = [row for row in rows if row["task"] == "StackCube pick-place"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    task_series = [find_row(pp_rows, "StackCube pick-place", "crop_top_surface", cond)["task_success_rate"] for cond in condition_order]
    pick_series = [find_row(pp_rows, "StackCube pick-place", "crop_top_surface", cond)["pick_success_rate"] for cond in condition_order]
    ax.plot(x, pick_series, marker="o", linewidth=1.8, color="#2563eb", label="pick")
    ax.plot(x, task_series, marker="s", linewidth=1.8, color="#16a34a", label="task/place")
    ax.set_xticks(x, ["clean", "5mm", "10mm", "drop10", "drop20"], rotation=20)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_title("StackCube pick-place sensor stress")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_sensor_stress_task_success")

    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for task, linestyle in [("PickCube pick", "-"), ("StackCube pick", "--")]:
        for source in ["box_center_depth", "crop_top_surface"]:
            series = [find_row(rows, task, source, cond)["mean_num_3d_points"] for cond in condition_order]
            ax.plot(x, series, marker="o", linestyle=linestyle, linewidth=1.8, color=colors[source], label=f"{task}, {format_source(source)}")
    ax.set_xticks(x, ["clean", "5mm", "10mm", "drop10", "drop20"], rotation=20)
    ax.set_ylabel("Mean crop 3D points")
    ax.set_title("Depth support under synthetic degradation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, figure_dir / "figure_sensor_stress_depth_support")


def find_row(rows: list[dict[str, Any]], task: str, source: str, condition: str) -> dict[str, Any]:
    for row in rows:
        if row["task"] == task and row["target_source"] == source and row["condition"] == condition:
            return row
    raise KeyError((task, source, condition))


def save_figure(fig: Any, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")


def format_source(source: str) -> str:
    return {
        "box_center_depth": "box center",
        "crop_top_surface": "crop top-surface",
    }.get(source, source.replace("_", " "))


def rate_ci(rate: float, low: float, high: float) -> str:
    return f"{rate:.3f} [{low:.3f},{high:.3f}]"


def escape_latex(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%")


if __name__ == "__main__":
    main()
