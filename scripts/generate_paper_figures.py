"""Generate camera-ready static figures for the Query-to-Grasp paper draft."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figure PNG/PDF assets.")
    parser.add_argument("--output-dir", type=Path, default=Path("paper") / "figures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_paper_figures(output_dir=args.output_dir)
    print(f"Wrote paper figures to {args.output_dir}")
    return 0


def generate_paper_figures(output_dir: Path) -> None:
    """Generate the current paper's static overview and geometry figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _import_matplotlib()
    draw_pipeline_overview(plt, output_dir=output_dir)
    draw_geometry_memory_ablation(plt, output_dir=output_dir)


def draw_pipeline_overview(plt, output_dir: Path) -> None:
    """Draw a compact left-to-right Query-to-Grasp system diagram."""

    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    nodes = [
        ("Language\nquery", 0.06, 0.62, "#d9e8ff"),
        ("Open-vocab\n2D grounding", 0.22, 0.62, "#e4f5df"),
        ("RGB-D lifting\n+ frame alignment", 0.40, 0.62, "#fff2cc"),
        ("Multi-view\nobject memory", 0.59, 0.62, "#efe1ff"),
        ("Target-source\nselection", 0.76, 0.62, "#ffe3dc"),
        ("Simulated\npick/place", 0.91, 0.62, "#d9ead3"),
    ]
    for label, x, y, color in nodes:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.42", facecolor=color, edgecolor="#334155", linewidth=1.0),
        )
    for (_, x0, y0, _), (_, x1, y1, _) in zip(nodes[:-1], nodes[1:]):
        ax.annotate(
            "",
            xy=(x1 - 0.075, y1),
            xytext=(x0 + 0.075, y0),
            arrowprops=dict(arrowstyle="->", linewidth=1.2, color="#334155"),
        )

    ax.text(
        0.50,
        0.25,
        "Diagnostics: CLIP top-1 changes, cross-view spread, re-observation state,\n"
        "target-source ablations, pick/place/task success, and failure taxonomy",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=0.9),
    )
    ax.text(
        0.50,
        0.06,
        "Claim boundary: high-fidelity simulation only; no real-robot or full non-oracle StackCube stacking claim.",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#475569",
    )
    save_figure(fig, output_dir / "pipeline_overview")


def draw_geometry_memory_ablation(plt, output_dir: Path) -> None:
    """Draw the accepted geometry-alignment result used in the paper narrative."""

    labels = ["Cross-view spread (m)", "Memory objects / run"]
    before = [1.0693, 3.3333]
    after = [0.0518, 1.3333]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.7))
    colors = ["#ef4444", "#22c55e"]
    for ax, label, b_val, a_val in zip(axes, labels, before, after):
        bars = ax.bar(["Before\nalignment", "After\nalignment"], [b_val, a_val], color=colors, width=0.62)
        ax.set_title(label, fontsize=9.5)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        upper = max(b_val, a_val) * 1.22
        ax.set_ylim(0, upper)
        for bar, value in zip(bars, [b_val, a_val]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + upper * 0.03,
                f"{value:.4g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Camera-frame alignment enables meaningful multi-view object memory", fontsize=10.5, y=0.98)
    fig.text(
        0.5,
        0.02,
        "H200 tabletop_3 diagnostic: corrected RGB-D lifting reduces projection spread and memory fragmentation.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.93])
    save_figure(fig, output_dir / "geometry_memory_ablation")


def save_figure(fig, stem: Path) -> None:
    """Save one figure as PNG and PDF."""

    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate paper figures.") from exc


if __name__ == "__main__":
    raise SystemExit(main())
