"""Generate RAS YCB expansion summaries, tables, and figures.

This script merges the original 200-seed PickSingleYCB gate with the
2026-05-11/12 H200 continuation chunks. It intentionally reads only lightweight
benchmark summaries/rows and produces paper-facing aggregate diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs") / "ras_revision_aggressive_20260511" / "ycb_expansion_summary"
DEFAULT_TABLE_DIR = Path("paper_ras") / "tables"
DEFAULT_FIGURE_DIR = Path("paper_ras") / "figures"


@dataclass(frozen=True)
class Chunk:
    source: str
    label: str
    path: Path


SOURCES = ("box_center_depth", "crop_median", "crop_top_surface", "oracle_object_pose")


def chunks() -> list[Chunk]:
    items: list[Chunk] = []
    for source in SOURCES:
        items.append(
            Chunk(
                source,
                "000-199",
                Path("outputs") / "h200_60071_ras_sunday_queue_20260509" / f"ycb_{source}",
            )
        )
        for start in (200, 400, 600, 800):
            items.append(
                Chunk(
                    source,
                    f"{start}-{start + 199}",
                    Path("outputs")
                    / "ras_revision_aggressive_20260511"
                    / "ycb_aggregate_expansion"
                    / f"ycb_{source}_seed{start}_{start + 199}",
                )
            )
        items.append(
            Chunk(
                source,
                "1000-1199",
                Path("outputs")
                / "ras_revision_aggressive_20260511"
                / "ycb_gpu4_extra_1000_1199"
                / f"ycb_{source}_seed1000_1199",
            )
        )
        for start in (1200, 1400):
            items.append(
                Chunk(
                    source,
                    f"{start}-{start + 199}",
                    Path("outputs")
                    / "ras_revision_aggressive_20260511"
                    / "ycb_post_main_1200_1599"
                    / f"ycb_{source}_seed{start}_{start + 199}",
                )
            )
    return items


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

    chunk_rows = [summarize_chunk(chunk) for chunk in chunks()]
    missing = [row for row in chunk_rows if row["missing"]]
    if missing:
        missing_paths = "\n".join(str(row["path"]) for row in missing)
        raise FileNotFoundError(f"Missing expected YCB chunk summaries:\n{missing_paths}")

    aggregate = aggregate_by_source(chunk_rows)
    failure_rows = build_failure_taxonomy(chunk_rows)

    write_csv(chunk_rows, args.output_dir / "ycb_chunk_consistency.csv")
    write_csv(aggregate, args.output_dir / "ycb_expansion_summary.csv")
    write_csv(failure_rows, args.output_dir / "ycb_failure_taxonomy.csv")
    write_json(
        {
            "aggregate": aggregate,
            "chunks": chunk_rows,
            "failure_taxonomy": failure_rows,
        },
        args.output_dir / "ycb_expansion_summary.json",
    )
    (args.output_dir / "ycb_expansion_summary.md").write_text(
        render_markdown(aggregate, failure_rows),
        encoding="utf-8",
    )

    write_table_bundle(aggregate, args.table_dir / "table_ycb_expansion_with_ci")
    write_table_bundle(chunk_rows, args.table_dir / "table_ycb_chunk_consistency")
    write_table_bundle(failure_rows, args.table_dir / "table_ycb_failure_taxonomy")
    # Keep the existing manuscript include path up to date.
    write_table_bundle(aggregate, args.table_dir / "table_noncube_gate_with_ci")

    draw_ladder(aggregate, args.figure_dir / "ras_ycb_noncube_ladder")
    draw_ladder(aggregate, args.figure_dir / "ras_ycb_expansion_ladder")
    draw_chunk_consistency(chunk_rows, args.figure_dir / "ras_ycb_chunk_consistency")

    update_manifest(args.table_dir / "ras_tables_manifest.json")
    print(f"Wrote YCB expansion summary to {args.output_dir}")
    return 0


def summarize_chunk(chunk: Chunk) -> dict[str, Any]:
    summary_path = chunk.path / "benchmark_summary.json"
    rows_path = chunk.path / "benchmark_rows.csv"
    if not summary_path.exists() or not rows_path.exists():
        return {
            "source": chunk.source,
            "chunk": chunk.label,
            "path": str(chunk.path),
            "missing": True,
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["aggregate_metrics"]
    total = int(summary.get("total_runs", metrics.get("total_runs", 0)) or 0)
    pick_success_count = round(float(metrics.get("pick_success_rate", 0.0) or 0.0) * total)
    target_available_count = round(float(metrics.get("fraction_with_3d_target", 0.0) or 0.0) * total)
    attempted_count = round(float(metrics.get("grasp_attempted_rate", 0.0) or 0.0) * total)
    stage_counts = {str(k): int(v) for k, v in (metrics.get("pick_stage_counts") or {}).items()}
    ci_low, ci_high = wilson_interval(pick_success_count, total)
    return {
        "source": chunk.source,
        "chunk": chunk.label,
        "n": total,
        "success_count": pick_success_count,
        "pick_success_rate": rate(pick_success_count, total),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "target_available": rate(target_available_count, total),
        "grasp_attempted": rate(attempted_count, total),
        "failed_runs": int(metrics.get("failed_runs", 0) or 0),
        "mean_num_detections": float(metrics.get("mean_num_detections", 0.0) or 0.0),
        "mean_num_3d_points": float(metrics.get("mean_num_3d_points", 0.0) or 0.0),
        "main_failure": main_failure(stage_counts),
        "stage_counts_json": json.dumps(stage_counts, sort_keys=True),
        "path": str(chunk.path),
        "missing": False,
    }


def aggregate_by_source(chunk_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in SOURCES:
        selected = [row for row in chunk_rows if row["source"] == source]
        n = sum(int(row["n"]) for row in selected)
        success_count = sum(int(row["success_count"]) for row in selected)
        target_available = sum(float(row["target_available"]) * int(row["n"]) for row in selected) / n
        attempted = sum(float(row["grasp_attempted"]) * int(row["n"]) for row in selected) / n
        failed = sum(int(row["failed_runs"]) for row in selected)
        mean_dets = sum(float(row["mean_num_detections"]) * int(row["n"]) for row in selected) / n
        mean_points = sum(float(row["mean_num_3d_points"]) * int(row["n"]) for row in selected) / n
        stages: Counter[str] = Counter()
        for row in selected:
            stages.update(json.loads(row["stage_counts_json"]))
        ci_low, ci_high = wilson_interval(success_count, n)
        out.append(
            {
                "env": "PickSingleYCB-v1",
                "query": "object",
                "target_source": source,
                "n": n,
                "success_count": success_count,
                "pick_success_rate": rate(success_count, n),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "target_available": target_available,
                "grasp_attempted": attempted,
                "failed_runs": failed,
                "mean_num_detections": mean_dets,
                "mean_num_3d_points": mean_points,
                "main_failure": main_failure(dict(stages)),
                "claim_boundary": (
                    "privileged executor-feasibility probe"
                    if source == "oracle_object_pose"
                    else "non-cube crop target-source diagnostic"
                ),
            }
        )
    return out


def build_failure_taxonomy(chunk_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in SOURCES:
        stages: Counter[str] = Counter()
        n = 0
        for row in chunk_rows:
            if row["source"] != source:
                continue
            n += int(row["n"])
            stages.update(json.loads(row["stage_counts_json"]))
        for stage, count in sorted(stages.items(), key=lambda item: (-item[1], item[0])):
            out.append(
                {
                    "target_source": source,
                    "stage": stage,
                    "count": count,
                    "rate": rate(count, n),
                    "n": n,
                    "interpretation": stage_interpretation(source, stage),
                }
            )
    return out


def stage_interpretation(source: str, stage: str) -> str:
    if stage == "success":
        return "successful top-down pick"
    if source == "oracle_object_pose":
        return "controller/object-geometry limit under privileged target"
    return "crop-derived target is available but not reliably executable"


def main_failure(stage_counts: dict[str, int]) -> str:
    failures = {k: v for k, v in stage_counts.items() if k != "success"}
    if not failures:
        return "none"
    return max(failures.items(), key=lambda item: item[1])[0]


def rate(count: float, total: int) -> float:
    return float(count) / total if total else 0.0


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_table_bundle(rows: list[dict[str, Any]], stem: Path) -> None:
    write_csv(rows, stem.with_suffix(".csv"))
    stem.with_suffix(".tex").write_text(render_latex(rows), encoding="utf-8")


def render_latex(rows: list[dict[str, Any]]) -> str:
    if "chunk" in rows[0]:
        header = "\\begin{tabular}{llrrrr}\n\\toprule\nSource & Chunk & N & Pick & 95\\% CI & Main failure \\\\\n\\midrule\n"
        body = []
        for row in rows:
            body.append(
                f"{pretty_source(row['source'])} & {row['chunk']} & {row['n']} & "
                f"{fmt(row['pick_success_rate'])} & "
                f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] & "
                f"{row['main_failure'].replace('_', ' ')} \\\\"
            )
        return header + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
    if "stage" in rows[0]:
        header = "\\begin{tabular}{llrrp{0.32\\linewidth}}\n\\toprule\nSource & Stage & Count & Rate & Interpretation \\\\\n\\midrule\n"
        body = []
        for row in rows:
            body.append(
                f"{pretty_source(row['target_source'])} & {row['stage'].replace('_', ' ')} & "
                f"{row['count']} & {fmt(row['rate'])} & {row['interpretation']} \\\\"
            )
        return header + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"
    header = "\\begin{tabular}{lrrrrrl}\n\\toprule\nTarget source & N & Target & Attempt & Pick & 95\\% CI & Boundary \\\\\n\\midrule\n"
    body = []
    for row in rows:
        body.append(
            f"{pretty_source(row['target_source'])} & {row['n']} & "
            f"{fmt(row['target_available'])} & {fmt(row['grasp_attempted'])} & "
            f"{fmt(row['pick_success_rate'])} & "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] & "
            f"{row['claim_boundary']} \\\\"
        )
    return header + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"


def render_markdown(aggregate: list[dict[str, Any]], failure_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# RAS YCB Expansion Summary",
        "",
        "Merged PickSingleYCB-v1 `object` query diagnostics over seeds 0-1599.",
        "",
        "## Aggregate Ladder",
        "",
        "| Target source | N | Pick | 95% CI | Target avail. | Main failure |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in aggregate:
        lines.append(
            f"| `{row['target_source']}` | {row['n']} | {fmt(row['pick_success_rate'])} | "
            f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}] | {fmt(row['target_available'])} | {row['main_failure']} |"
        )
    lines += ["", "## Failure Taxonomy", "", "| Target source | Stage | Count | Rate | Interpretation |", "|---|---|---:|---:|---|"]
    for row in failure_rows:
        lines.append(
            f"| `{row['target_source']}` | {row['stage']} | {row['count']} | {fmt(row['rate'])} | {row['interpretation']} |"
        )
    lines += [
        "",
        "Interpretation: all target sources maintain 3D target availability, but crop-derived targets remain weak on PickSingleYCB while oracle targets are partially executable. This supports a diagnostic claim, not broad YCB manipulation.",
        "",
    ]
    return "\n".join(lines)


def draw_ladder(rows: list[dict[str, Any]], stem: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [pretty_source(row["target_source"]) for row in rows]
    values = [row["pick_success_rate"] for row in rows]
    lowers = [row["pick_success_rate"] - row["ci_low"] for row in rows]
    uppers = [row["ci_high"] - row["pick_success_rate"] for row in rows]
    colors = ["#8fb9d9", "#9ac79b", "#e5bd6a", "#b8a0d9"]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = range(len(rows))
    ax.bar(x, values, yerr=[lowers, uppers], capsize=4, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_xticks(list(x), labels, rotation=12, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Pick success")
    ax.set_title("PickSingleYCB target-source diagnostic")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def draw_chunk_consistency(rows: list[dict[str, Any]], stem: Path) -> None:
    import matplotlib.pyplot as plt

    chunks_order = ["000-199", "200-399", "400-599", "600-799", "800-999", "1000-1199", "1200-1399", "1400-1599"]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    for source in SOURCES:
        by_chunk = {row["chunk"]: row for row in rows if row["source"] == source}
        values = [by_chunk[ch]["pick_success_rate"] for ch in chunks_order]
        ax.plot(chunks_order, values, marker="o", linewidth=1.8, label=pretty_source(source))
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Pick success")
    ax.set_xlabel("Seed chunk")
    ax.set_title("PickSingleYCB chunk consistency")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)


def update_manifest(path: Path) -> None:
    manifest: dict[str, Any] = {}
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    tables = set(manifest.get("tables", []))
    figures = set(manifest.get("figures", []))
    tables.update(
        {
            "table_ycb_expansion_with_ci.csv",
            "table_ycb_expansion_with_ci.tex",
            "table_ycb_chunk_consistency.csv",
            "table_ycb_chunk_consistency.tex",
            "table_ycb_failure_taxonomy.csv",
            "table_ycb_failure_taxonomy.tex",
        }
    )
    figures.update(
        {
            "ras_ycb_expansion_ladder.pdf",
            "ras_ycb_expansion_ladder.png",
            "ras_ycb_chunk_consistency.pdf",
            "ras_ycb_chunk_consistency.png",
        }
    )
    manifest["tables"] = sorted(tables)
    manifest["figures"] = sorted(figures)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def pretty_source(source: str) -> str:
    return {
        "box_center_depth": "Box center",
        "crop_median": "Crop median",
        "crop_top_surface": "Crop top-surface",
        "oracle_object_pose": "Oracle object pose",
    }.get(source, source.replace("_", " "))


def fmt(value: float) -> str:
    return f"{value:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
