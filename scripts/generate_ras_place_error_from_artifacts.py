"""Generate StackCube place-target error diagnostics from per-run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ERROR_BINS_CM = [
    (0.0, 1.0, "[0,1)"),
    (1.0, 2.0, "[1,2)"),
    (2.0, 5.0, "[2,5)"),
    (5.0, 10.0, "[5,10)"),
    (10.0, math.inf, ">=10"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def vec(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def dist_cm(a: list[float], b: list[float]) -> float:
    return 100.0 * math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def parse_seed(path: Path, summary: dict[str, Any]) -> int | None:
    for part in path.parts:
        if "seed_" in part:
            tail = part.split("seed_", 1)[1]
            digits = "".join(ch for ch in tail if ch.isdigit())
            if digits:
                return int(digits)
    seed = summary.get("seed")
    return int(seed) if seed is not None else None


def bin_label(error_cm: float) -> str:
    for lo, hi, label in ERROR_BINS_CM:
        if lo <= error_cm < hi:
            return label
    return "unknown"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def experiment_name(path: Path) -> str:
    for parent in path.parents:
        if (parent / "benchmark_rows.csv").exists() or (parent / "benchmark_summary.json").exists():
            return parent.name
    return path.parent.name


def collect_oracle_place(roots: list[Path]) -> dict[int, list[float]]:
    oracle: dict[int, list[float]] = {}
    for root in roots:
        if not root.exists():
            continue
        for pick_path in root.rglob("pick_result.json"):
            summary_path = pick_path.with_name("summary.json")
            if not summary_path.exists():
                continue
            pick = load_json(pick_path)
            summary = load_json(summary_path)
            seed = parse_seed(pick_path, summary)
            if seed is None:
                continue
            meta = pick.get("metadata") if isinstance(pick.get("metadata"), dict) else {}
            place_meta = meta.get("place_target_metadata") if isinstance(meta.get("place_target_metadata"), dict) else {}
            before = vec(place_meta.get("place_xyz_before_noise"))
            target = vec(meta.get("place_target_xyz")) or vec(pick.get("place_xyz"))
            if before is not None:
                oracle.setdefault(seed, before)
            elif target is not None and meta.get("place_target_source") == "oracle_cubeB_pose":
                oracle.setdefault(seed, target)
    return oracle


def collect_rows(roots: list[Path], oracle: dict[int, list[float]]) -> tuple[list[dict[str, Any]], Counter]:
    rows: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for root in roots:
        if not root.exists():
            continue
        for pick_path in root.rglob("pick_result.json"):
            summary_path = pick_path.with_name("summary.json")
            if not summary_path.exists():
                continue
            pick = load_json(pick_path)
            summary = load_json(summary_path)
            seed = parse_seed(pick_path, summary)
            if seed is None:
                skipped["missing_seed"] += 1
                continue
            meta = pick.get("metadata") if isinstance(pick.get("metadata"), dict) else {}
            place_source = meta.get("place_target_source") or summary.get("place_target_source")
            place_xyz = vec(meta.get("place_target_xyz")) or vec(pick.get("place_xyz")) or vec(summary.get("place_target_xyz"))
            place_meta = meta.get("place_target_metadata") if isinstance(meta.get("place_target_metadata"), dict) else {}
            oracle_xyz = oracle.get(seed)
            error_cm = None
            if place_source == "oracle_cubeB_pose":
                before = vec(place_meta.get("place_xyz_before_noise"))
                applied = place_meta.get("applied_place_noise")
                if isinstance(applied, list) and len(applied) == 3:
                    error_cm = 100.0 * math.sqrt(sum(float(x) ** 2 for x in applied))
                elif before is not None and place_xyz is not None:
                    error_cm = dist_cm(place_xyz, before)
                else:
                    error_cm = 0.0
                oracle_xyz = before or place_xyz
            elif place_xyz is not None and oracle_xyz is not None:
                error_cm = dist_cm(place_xyz, oracle_xyz)
            else:
                skipped["missing_place_or_oracle"] += 1
                continue

            rows.append(
                {
                    "seed": seed,
                    "experiment": experiment_name(pick_path),
                    "place_target_source": place_source or "unknown",
                    "place_error_cm": f"{error_cm:.6f}",
                    "error_bin_cm": bin_label(float(error_cm)),
                    "pick_success": int(bool(pick.get("pick_success", summary.get("pick_success", False)))),
                    "place_success": "" if pick.get("place_success", summary.get("place_success")) is None else int(bool(pick.get("place_success", summary.get("place_success")))),
                    "task_success": "" if pick.get("task_success", summary.get("task_success")) is None else int(bool(pick.get("task_success", summary.get("task_success")))),
                    "place_target_xyz": json.dumps(place_xyz),
                    "oracle_place_xyz": json.dumps(oracle_xyz),
                    "artifact_path": str(pick_path),
                }
            )
    return rows, skipped


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["error_bin_cm"]].append(row)
    out: list[dict[str, Any]] = []
    for _, _, label in ERROR_BINS_CM:
        group = grouped.get(label, [])
        if not group:
            continue
        n = len(group)
        task_vals = [int(r["task_success"]) for r in group if r["task_success"] != ""]
        place_vals = [int(r["place_success"]) for r in group if r["place_success"] != ""]
        task_rate = sum(task_vals) / len(task_vals) if task_vals else 0.0
        place_rate = sum(place_vals) / len(place_vals) if place_vals else 0.0
        task_lo, task_hi = wilson(sum(task_vals), len(task_vals)) if task_vals else (0.0, 0.0)
        place_lo, place_hi = wilson(sum(place_vals), len(place_vals)) if place_vals else (0.0, 0.0)
        errors = sorted(float(r["place_error_cm"]) for r in group)
        out.append(
            {
                "error_bin_cm": label,
                "n": n,
                "place_n": len(place_vals),
                "place_success_rate": f"{place_rate:.6f}",
                "place_ci_low": f"{place_lo:.6f}",
                "place_ci_high": f"{place_hi:.6f}",
                "task_n": len(task_vals),
                "task_success_rate": f"{task_rate:.6f}",
                "task_ci_low": f"{task_lo:.6f}",
                "task_ci_high": f"{task_hi:.6f}",
                "median_place_error_cm": f"{errors[len(errors)//2]:.6f}",
            }
        )
    return out


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Place error (cm) & N & Place success & Task success \\\\",
        "\\midrule",
    ]
    for row in rows:
        place = f"{float(row['place_success_rate']):.3f}"
        task = f"{float(row['task_success_rate']):.3f}"
        lines.append(f"{row['error_bin_cm']} & {row['n']} & {place} & {task} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def plot(output_dir: Path, rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    labels = [r["error_bin_cm"] for r in rows]
    x = list(range(len(labels)))
    task = [float(r["task_success_rate"]) for r in rows]
    place = [float(r["place_success_rate"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.plot(x, place, marker="o", label="place success", color="#2F6B9A")
    ax.plot(x, task, marker="s", label="task success", color="#B4473A")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Place-target error bin (cm)")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    outs = []
    for ext in ("pdf", "png"):
        p = output_dir / f"figure_place_error_success.{ext}"
        fig.savefig(p, dpi=220)
        outs.append(str(p))
    return outs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", action="append", default=[])
    parser.add_argument("--oracle-root", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper-tables-dir", default=None)
    parser.add_argument("--paper-figures-dir", default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    oracle = collect_oracle_place([Path(p) for p in args.oracle_root])
    rows, skipped = collect_rows([Path(p) for p in args.pred_root], oracle)
    summary_rows = summarize(rows)

    fields = [
        "seed",
        "experiment",
        "place_target_source",
        "place_error_cm",
        "error_bin_cm",
        "pick_success",
        "place_success",
        "task_success",
        "place_target_xyz",
        "oracle_place_xyz",
        "artifact_path",
    ]
    write_csv(out / "rows_place_error_all.csv", rows, fields)
    summary_fields = [
        "error_bin_cm",
        "n",
        "place_n",
        "place_success_rate",
        "place_ci_low",
        "place_ci_high",
        "task_n",
        "task_success_rate",
        "task_ci_low",
        "task_ci_high",
        "median_place_error_cm",
    ]
    write_csv(out / "table_place_error_bins_with_ci.csv", summary_rows, summary_fields)
    write_latex(out / "table_place_error_bins_with_ci.tex", summary_rows)
    figures = plot(out, summary_rows)
    summary = {
        "oracle_seed_count": len(oracle),
        "rows_with_place_error": len(rows),
        "skipped": dict(skipped),
        "outputs": {
            "rows": str(out / "rows_place_error_all.csv"),
            "summary": str(out / "table_place_error_bins_with_ci.csv"),
            "figures": figures,
        },
    }
    (out / "summary_place_error.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.paper_tables_dir:
        dst = Path(args.paper_tables_dir)
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("table_place_error_bins_with_ci.csv", "table_place_error_bins_with_ci.tex"):
            (dst / name).write_bytes((out / name).read_bytes())
    if args.paper_figures_dir:
        dst = Path(args.paper_figures_dir)
        dst.mkdir(parents=True, exist_ok=True)
        for fig in figures:
            p = Path(fig)
            (dst / p.name).write_bytes(p.read_bytes())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
