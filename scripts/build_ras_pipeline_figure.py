"""Build the reproducible vector pipeline figure for the RAS manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DEFAULT_OUTPUT = Path("paper_ras") / "figures" / "pipeline_overview_vector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.0, 5.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    colors = {
        "input": "#dbeafe",
        "ground": "#dcfce7",
        "lift": "#fef3c7",
        "memory": "#ede9fe",
        "ladder": "#ffedd5",
        "exec": "#dcfce7",
        "metric": "#f3f4f6",
        "diag": "#f8fafc",
        "ref": "#ffe4e6",
    }

    headers = [
        ("Inputs", 0.5, 2.2),
        ("Retrieval to 3D Action Targets", 3.0, 8.4),
        ("Execution and Diagnosis", 10.0, 3.4),
    ]
    for text, x, width in headers:
        ax.text(x + width / 2, 5.05, text, ha="center", va="center", fontsize=15, fontweight="bold")
        ax.plot([x, x + width], [4.83, 4.83], color="#111827", linewidth=1.0)
        ax.plot([x, x], [4.83, 4.70], color="#111827", linewidth=1.0)
        ax.plot([x + width, x + width], [4.83, 4.70], color="#111827", linewidth=1.0)

    boxes = {
        "query": box(ax, 0.35, 3.65, 1.75, 0.58, "Language query $q$", colors["input"]),
        "rgbd": box(ax, 0.35, 2.65, 1.75, 0.72, "RGB-D views\n$(I_v,D_v,T_v)$", colors["input"]),
        "ground": box(ax, 2.55, 2.75, 1.95, 1.20, "Open-vocabulary\n2D grounding\n\nboxes $b_i$, labels $c_i$,\nscores $s_i$", colors["ground"]),
        "lift": box(ax, 5.00, 2.75, 1.95, 1.20, "RGB-D lifting +\ncamera-frame alignment\n\nworld points $X_i$,\ngrip samples $g_i$", colors["lift"]),
        "memory": box(ax, 7.40, 2.75, 1.95, 1.20, "Multi-view\nobject memory\n\n$m_k=\\{X,g,$ views,\nscores$\\}$", colors["memory"]),
        "ladder": box(ax, 9.85, 2.45, 2.20, 1.65, "Target-source ladder\n\nSC: semantic center\nRG/MG: grasp target\nTG: task guard\nOP: oracle pose\nPG: predicted ref.", colors["ladder"], edge="#ea580c"),
        "exec": box(ax, 12.40, 2.72, 1.45, 1.20, "Scripted executor\n\nsim_topdown\nsim_pick_place", colors["exec"]),
        "metric": box(ax, 12.30, 0.90, 1.65, 1.05, "Metrics + failure\nattribution\n\npick, place, RawEnv\nfailure type", colors["metric"]),
        "ref_query": box(ax, 4.70, 0.68, 1.85, 0.58, "Reference query $q_p$:\ngreen cube", colors["ref"]),
        "pred_ref": box(ax, 7.00, 0.68, 1.85, 0.58, "Predicted place\nreference $p_B$", colors["ref"]),
        "oracle": box(ax, 8.90, 0.62, 1.55, 0.70, "or privileged\noracle cubeB pose", "#ffffff", dashed=True),
    }

    arrow(ax, boxes["query"], boxes["ground"])
    arrow(ax, boxes["rgbd"], boxes["ground"])
    arrow(ax, boxes["ground"], boxes["lift"])
    arrow(ax, boxes["lift"], boxes["memory"])
    arrow(ax, boxes["memory"], boxes["ladder"])
    arrow(ax, boxes["ladder"], boxes["exec"])
    arrow(ax, boxes["exec"], boxes["metric"], start_side="bottom", end_side="top")
    arrow(ax, boxes["ref_query"], boxes["pred_ref"])
    arrow(ax, boxes["pred_ref"], boxes["ladder"], start_side="right", end_side="bottom")
    arrow(ax, boxes["oracle"], boxes["ladder"], start_side="right", end_side="bottom", dashed=True)

    diag_box(ax, 2.62, 1.95, "CLIP top-1\nchange rate")
    diag_box(ax, 4.72, 1.60, "cross-view\nspread")
    diag_box(ax, 5.88, 1.60, "geometry\nconfidence")
    diag_box(ax, 7.42, 1.60, "memory\nfragmentation")
    diag_box(ax, 9.58, 1.45, "target-source\nablation")
    diag_box(ax, 10.86, 1.45, "noisy-oracle\nsensitivity")
    diag_box(ax, 12.62, 0.20, "failure\ntaxonomy")

    down_arrow(ax, boxes["ground"], (3.35, 2.02))
    down_arrow(ax, boxes["lift"], (5.62, 1.68))
    down_arrow(ax, boxes["memory"], (8.10, 1.68))
    down_arrow(ax, boxes["ladder"], (10.36, 1.55))
    down_arrow(ax, boxes["metric"], (13.10, 0.55))

    ax.text(
        7.0,
        0.15,
        "Oracle and noisy-oracle rows are privileged diagnostic probes, not deployable perception claims.",
        ha="center",
        va="center",
        fontsize=11,
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=220)
    plt.close(fig)
    return 0


def box(ax, x: float, y: float, w: float, h: float, text: str, face: str, edge: str = "#1f2937", dashed: bool = False):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.1,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10.5)
    return (x, y, w, h)


def diag_box(ax, x: float, y: float, text: str) -> None:
    box(ax, x, y, 1.15, 0.55, text, "#f8fafc", edge="#6b7280")


def point(rect, side: str) -> tuple[float, float]:
    x, y, w, h = rect
    if side == "right":
        return x + w, y + h / 2
    if side == "left":
        return x, y + h / 2
    if side == "top":
        return x + w / 2, y + h
    if side == "bottom":
        return x + w / 2, y
    raise ValueError(side)


def arrow(ax, start, end, start_side: str = "right", end_side: str = "left", dashed: bool = False) -> None:
    p0 = point(start, start_side)
    p1 = point(end, end_side)
    patch = FancyArrowPatch(
        p0,
        p1,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.2,
        color="#374151",
        linestyle="--" if dashed else "-",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(patch)


def down_arrow(ax, start, end_xy: tuple[float, float]) -> None:
    p0 = point(start, "bottom")
    patch = FancyArrowPatch(
        p0,
        end_xy,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=0.9,
        color="#6b7280",
        linestyle="--",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(patch)


if __name__ == "__main__":
    raise SystemExit(main())
