"""Generate RAS target-error-to-success analysis from per-run artifacts.

The script works from complete run directories containing ``pick_result.json``
and ``summary.json``. It does not require rerunning simulation. For ordinary
target-source rows, it matches each seed to the same-seed oracle_object_pose row
and computes the Euclidean distance between executed targets. For noisy-oracle
rows, it uses the recorded applied noise as the target perturbation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


ERROR_BINS_CM = [
    (0.0, 0.5, "[0,0.5)"),
    (0.5, 1.0, "[0.5,1)"),
    (1.0, 2.0, "[1,2)"),
    (2.0, 5.0, "[2,5)"),
    (5.0, 10.0, "[5,10)"),
    (10.0, math.inf, ">=10"),
]


@dataclass
class RunRecord:
    dataset: str
    env_id: str
    task: str
    experiment: str
    seed: int
    query: str
    target_source: str
    target_xyz: list[float] | None
    pick_success: bool
    place_success: bool | None
    task_success: bool | None
    stage: str | None
    target_available: bool
    num_3d_points: int | None
    path: str


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text())


def parse_seed(path: Path, summary: dict[str, Any]) -> int | None:
    for part in path.parts:
        m = re.search(r"seed_(\d+)", part)
        if m:
            return int(m.group(1))
    seed = summary.get("seed")
    return int(seed) if seed is not None else None


def vec(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def dist_cm(a: list[float], b: list[float]) -> float:
    return 100.0 * math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def infer_dataset_and_env(path: Path) -> tuple[str, str, str]:
    s = str(path).replace("\\", "/").lower()
    if ("ras_revision_aggressive_20260511" in s or "h200_60071_ras_sunday_queue_20260509" in s) and "ycb_" in s:
        return "ycb_1600", "PickSingleYCB-v1", "pick"
    if "h200_60071_exp_a_200_seed_freeze" in s:
        if "pickcube" in s:
            return "external_crop_200", "PickCube-v1", "pick"
        if "stackcube_pickplace" in s:
            return "external_crop_200", "StackCube-v1", "pick_place"
        if "stackcube_pickonly" in s or "stackcube_pick_" in s:
            return "external_crop_200", "StackCube-v1", "pick"
    if "h200_60071_noisy_oracle" in s:
        if "pickcube" in s:
            return "noisy_oracle", "PickCube-v1", "pick"
        if "pickplace" in s:
            return "noisy_oracle", "StackCube-v1", "pick_place"
        return "noisy_oracle", "StackCube-v1", "pick"
    if "liftpeg" in s:
        return "noncube_gate", "LiftPegUpright-v1", "pick"
    return "unknown", "unknown", "unknown"


def experiment_dir(path: Path) -> str:
    for parent in path.parents:
        if (parent / "benchmark_summary.json").exists() or (parent / "benchmark_rows.csv").exists():
            return parent.name
    # Fallback to the directory under runs.
    parts = list(path.parts)
    if "runs" in parts:
        idx = parts.index("runs")
        if idx > 0:
            return parts[idx - 1]
    return path.parent.name


def target_source_from(summary: dict[str, Any], pick: dict[str, Any], experiment: str) -> str:
    meta = pick.get("metadata") if isinstance(pick.get("metadata"), dict) else {}
    source = summary.get("grasp_target_mode") or meta.get("grasp_target_mode")
    if source:
        return str(source)
    for candidate in (
        "box_center_depth",
        "crop_median",
        "crop_top_surface",
        "oracle_object_pose",
        "semantic",
        "refined",
    ):
        if candidate in experiment:
            return candidate
    return str(meta.get("target_used_for_pick") or "unknown")


def read_runs(roots: list[Path]) -> list[RunRecord]:
    records: list[RunRecord] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for pick_path in sorted(root.rglob("pick_result.json")):
            pick_path = pick_path.resolve()
            if pick_path in seen:
                continue
            seen.add(pick_path)
            summary_path = pick_path.with_name("summary.json")
            if not summary_path.exists():
                continue
            pick = load_json(pick_path)
            summary = load_json(summary_path)
            seed = parse_seed(pick_path, summary)
            if seed is None:
                continue
            dataset, env_id, task = infer_dataset_and_env(pick_path)
            exp = experiment_dir(pick_path)
            target_xyz = vec(pick.get("target_xyz")) or vec(summary.get("target_xyz")) or vec(summary.get("world_xyz"))
            source = target_source_from(summary, pick, exp)
            records.append(
                RunRecord(
                    dataset=dataset,
                    env_id=env_id,
                    task=task,
                    experiment=exp,
                    seed=seed,
                    query=str(summary.get("query") or ""),
                    target_source=source,
                    target_xyz=target_xyz,
                    pick_success=bool(pick.get("pick_success", pick.get("success", summary.get("pick_success", False)))),
                    place_success=pick.get("place_success", summary.get("place_success")),
                    task_success=pick.get("task_success", summary.get("task_success")),
                    stage=pick.get("stage", summary.get("pick_stage")),
                    target_available=target_xyz is not None,
                    num_3d_points=summary.get("num_3d_points"),
                    path=str(pick_path),
                )
            )
    return records


def oracle_group_key(record: RunRecord) -> tuple[str, str, int]:
    if record.dataset == "ycb_1600":
        return (record.dataset, record.env_id, record.seed)
    if record.dataset == "external_crop_200":
        # Pick-place rows share the same pick oracle as StackCube pick-only.
        return (record.dataset, record.env_id, record.seed)
    return (record.dataset, record.env_id, record.seed)


def noisy_error_cm(path: str, pick_path: Path | None = None) -> float | None:
    if pick_path is None:
        pick_path = Path(path)
    pick = load_json(pick_path)
    meta = pick.get("metadata") if isinstance(pick.get("metadata"), dict) else {}
    noise = meta.get("applied_noise")
    if isinstance(noise, list) and len(noise) == 3:
        try:
            return 100.0 * math.sqrt(sum(float(x) ** 2 for x in noise))
        except (TypeError, ValueError):
            return None
    before = vec(meta.get("target_xyz_before_noise"))
    target = vec(pick.get("target_xyz"))
    if before is not None and target is not None:
        return dist_cm(target, before)
    return None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def error_bin(error_cm: float) -> str:
    for lo, hi, label in ERROR_BINS_CM:
        if lo <= error_cm < hi:
            return label
    return "unknown"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt_rate(value: float, lo: float, hi: float) -> str:
    return f"{value:.3f} [{lo:.3f},{hi:.3f}]"


def write_latex_table(path: Path, rows: list[dict[str, Any]], caption: str, label: str) -> None:
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\hline",
        "Error bin (cm) & N & Pick success & Median error \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['error_bin_cm']} & {row['n']} & {row['pick_rate_ci']} & {float(row['median_error_cm']):.2f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    generated: list[str] = []
    if not rows:
        return generated

    # Bin success plot.
    labels = [r["error_bin_cm"] for r in rows]
    x = list(range(len(labels)))
    pick = [float(r["pick_success_rate"]) for r in rows]
    lo = [float(r["pick_success_rate"]) - float(r["pick_ci_low"]) for r in rows]
    hi = [float(r["pick_ci_high"]) - float(r["pick_success_rate"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.errorbar(x, pick, yerr=[lo, hi], marker="o", linewidth=1.8, capsize=3, color="#2F6B9A")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Target error bin (cm)")
    ax.set_ylabel("Pick success")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = output_dir / f"figure_error_bins_success.{ext}"
        fig.savefig(out, dpi=220)
        generated.append(str(out))
    plt.close(fig)

    return generated


def build_analysis(records: list[RunRecord]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    oracle: dict[tuple[str, str, int], RunRecord] = {}
    for record in records:
        if record.target_source == "oracle_object_pose" and record.target_xyz is not None:
            oracle[oracle_group_key(record)] = record

    rows: list[dict[str, Any]] = []
    skipped = Counter()
    for record in records:
        error_cm: float | None = None
        oracle_path = ""
        oracle_target = None
        if record.dataset == "noisy_oracle":
            error_cm = noisy_error_cm(record.path)
        elif record.target_source == "oracle_object_pose" and record.target_xyz is not None:
            error_cm = 0.0
            oracle_target = record.target_xyz
            oracle_path = record.path
        else:
            ref = oracle.get(oracle_group_key(record))
            if ref is None or ref.target_xyz is None:
                skipped["missing_oracle_match"] += 1
                continue
            if record.target_xyz is None:
                skipped["missing_target_xyz"] += 1
                continue
            error_cm = dist_cm(record.target_xyz, ref.target_xyz)
            oracle_target = ref.target_xyz
            oracle_path = ref.path

        if error_cm is None:
            skipped["missing_error"] += 1
            continue
        rows.append(
            {
                "dataset": record.dataset,
                "env_id": record.env_id,
                "task": record.task,
                "experiment": record.experiment,
                "seed": record.seed,
                "query": record.query,
                "target_source": record.target_source,
                "target_error_cm": f"{error_cm:.6f}",
                "error_bin_cm": error_bin(error_cm),
                "pick_success": int(bool(record.pick_success)),
                "place_success": "" if record.place_success is None else int(bool(record.place_success)),
                "task_success": "" if record.task_success is None else int(bool(record.task_success)),
                "target_available": int(bool(record.target_available)),
                "num_3d_points": "" if record.num_3d_points is None else record.num_3d_points,
                "stage": record.stage or "",
                "target_xyz": json.dumps(record.target_xyz),
                "oracle_target_xyz": json.dumps(oracle_target),
                "artifact_path": record.path,
                "oracle_artifact_path": oracle_path,
            }
        )

    bin_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["error_bin_cm"]].append(row)

    order = [label for _, _, label in ERROR_BINS_CM]
    for label in order:
        group = grouped.get(label, [])
        if not group:
            continue
        n = len(group)
        pick_k = sum(int(r["pick_success"]) for r in group)
        pick_lo, pick_hi = wilson(pick_k, n)
        place_vals = [int(r["place_success"]) for r in group if r["place_success"] != ""]
        task_vals = [int(r["task_success"]) for r in group if r["task_success"] != ""]
        place_rate = sum(place_vals) / len(place_vals) if place_vals else 0.0
        task_rate = sum(task_vals) / len(task_vals) if task_vals else 0.0
        place_lo, place_hi = wilson(sum(place_vals), len(place_vals)) if place_vals else (0.0, 0.0)
        task_lo, task_hi = wilson(sum(task_vals), len(task_vals)) if task_vals else (0.0, 0.0)
        bin_rows.append(
            {
                "error_bin_cm": label,
                "n": n,
                "pick_success_rate": f"{pick_k / n:.6f}",
                "pick_ci_low": f"{pick_lo:.6f}",
                "pick_ci_high": f"{pick_hi:.6f}",
                "pick_rate_ci": fmt_rate(pick_k / n, pick_lo, pick_hi),
                "place_n": len(place_vals),
                "place_success_rate": f"{place_rate:.6f}",
                "place_ci_low": f"{place_lo:.6f}",
                "place_ci_high": f"{place_hi:.6f}",
                "place_rate_ci": "-" if not place_vals else fmt_rate(place_rate, place_lo, place_hi),
                "task_n": len(task_vals),
                "task_success_rate": f"{task_rate:.6f}",
                "task_ci_low": f"{task_lo:.6f}",
                "task_ci_high": f"{task_hi:.6f}",
                "task_rate_ci": "-" if not task_vals else fmt_rate(task_rate, task_lo, task_hi),
                "median_error_cm": f"{median(float(r['target_error_cm']) for r in group):.6f}",
            }
        )

    summary = {
        "records_scanned": len(records),
        "oracle_matches": len(oracle),
        "rows_with_target_error": len(rows),
        "skipped": dict(skipped),
        "datasets": dict(Counter(r.dataset for r in records)),
        "target_error_datasets": dict(Counter(r["dataset"] for r in rows)),
        "target_sources": dict(Counter(r.target_source for r in records)),
    }
    return rows, bin_rows, summary


def copy_if_requested(src: Path, dst_dir: Path | None) -> None:
    if dst_dir is None:
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    dst.write_bytes(src.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, help="Root artifact directory to scan. Repeatable.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper-tables-dir", default=None)
    parser.add_argument("--paper-figures-dir", default=None)
    args = parser.parse_args()

    roots = [Path(r).resolve() for r in args.root]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_runs(roots)
    rows, bin_rows, summary = build_analysis(records)

    row_fields = [
        "dataset",
        "env_id",
        "task",
        "experiment",
        "seed",
        "query",
        "target_source",
        "target_error_cm",
        "error_bin_cm",
        "pick_success",
        "place_success",
        "task_success",
        "target_available",
        "num_3d_points",
        "stage",
        "target_xyz",
        "oracle_target_xyz",
        "artifact_path",
        "oracle_artifact_path",
    ]
    bin_fields = [
        "error_bin_cm",
        "n",
        "pick_success_rate",
        "pick_ci_low",
        "pick_ci_high",
        "pick_rate_ci",
        "place_n",
        "place_success_rate",
        "place_ci_low",
        "place_ci_high",
        "place_rate_ci",
        "task_n",
        "task_success_rate",
        "task_ci_low",
        "task_ci_high",
        "task_rate_ci",
        "median_error_cm",
    ]

    rows_csv = output_dir / "rows_target_error_all.csv"
    bins_csv = output_dir / "table_error_bins_with_ci.csv"
    bins_tex = output_dir / "table_error_bins_with_ci.tex"
    summary_json = output_dir / "summary_error_correlation.json"
    write_csv(rows_csv, rows, row_fields)
    write_csv(bins_csv, bin_rows, bin_fields)
    write_latex_table(
        bins_tex,
        bin_rows,
        "Target-error bins and execution outcomes. Error is computed from matched oracle-object targets for real target-source rows and from the recorded perturbation for noisy-oracle rows.",
        "tab:target-error-bins",
    )
    figures = make_plots(output_dir, bin_rows)

    summary["outputs"] = {
        "rows": str(rows_csv),
        "bins_csv": str(bins_csv),
        "bins_tex": str(bins_tex),
        "figures": figures,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tables_dir = Path(args.paper_tables_dir).resolve() if args.paper_tables_dir else None
    figures_dir = Path(args.paper_figures_dir).resolve() if args.paper_figures_dir else None
    copy_if_requested(bins_csv, tables_dir)
    copy_if_requested(bins_tex, tables_dir)
    for fig in figures:
        copy_if_requested(Path(fig), figures_dir)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
