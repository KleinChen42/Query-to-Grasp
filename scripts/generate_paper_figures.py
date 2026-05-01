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
    """Generate the current paper's static overview, result, and diagnostic figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _import_matplotlib()
    draw_pipeline_overview(plt, output_dir=output_dir)
    draw_geometry_memory_ablation(plt, output_dir=output_dir)
    draw_target_source_results(plt, output_dir=output_dir)
    draw_stackcube_failure_taxonomy(plt, output_dir=output_dir)


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


def draw_target_source_results(plt, output_dir: Path) -> None:
    """Draw the frozen target-source result summary for the main paper."""

    rows = [
        ("PickCube\nfull MV/CL", 1.00, None, 0.1455, "#2563eb"),
        ("StackCube\npick-only tabletop", 0.62, None, 0.00, "#0f766e"),
        ("StackCube\npick-only CL", 0.52, None, 0.00, "#0f766e"),
        ("Oracle\npick-place", 0.94, 0.88, 0.88, "#7c3aed"),
        ("Query pick +\noracle place\nsingle-view", 0.88, 0.72, 0.72, "#ea580c"),
        ("Query pick +\noracle place\ntabletop", 0.62, 0.52, 0.52, "#ea580c"),
        ("Query pick +\noracle place\nclosed-loop", 0.52, 0.48, 0.48, "#ea580c"),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    labels = [row[0] for row in rows]
    y_positions = list(range(len(rows)))
    bar_height = 0.22

    pick_values = [row[1] for row in rows]
    place_values = [0.0 if row[2] is None else row[2] for row in rows]
    task_values = [row[3] for row in rows]

    ax.barh([y + bar_height for y in y_positions], pick_values, height=bar_height, color="#2563eb", label="Pick")
    ax.barh(y_positions, place_values, height=bar_height, color="#16a34a", label="Place")
    ax.barh([y - bar_height for y in y_positions], task_values, height=bar_height, color="#f97316", label="Task")

    for y, pick, place, task in zip(y_positions, pick_values, [row[2] for row in rows], task_values):
        ax.text(pick + 0.018, y + bar_height, f"{pick:.2f}", va="center", fontsize=7.5, color="#1e293b")
        if place is not None:
            ax.text(place + 0.018, y, f"{place:.2f}", va="center", fontsize=7.5, color="#1e293b")
        else:
            ax.text(0.018, y, "n/a", va="center", fontsize=7.2, color="#64748b")
        ax.text(task + 0.018, y - bar_height, f"{task:.2f}", va="center", fontsize=7.5, color="#1e293b")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Success rate", fontsize=8.5)
    ax.set_title("Target-source quality determines executable manipulation success", fontsize=10.5)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    ax.invert_yaxis()
    fig.text(
        0.5,
        0.02,
        "Frozen paper rows: PickCube uses memory grasp targets; StackCube bridge uses query pick with oracle cubeB placement.",
        ha="center",
        fontsize=7.6,
        color="#475569",
    )
    fig.tight_layout(rect=[0.02, 0.07, 0.98, 0.96])
    save_figure(fig, output_dir / "target_source_results")


def draw_stackcube_failure_taxonomy(plt, output_dir: Path) -> None:
    """Draw the expanded StackCube failure taxonomy used in the limitation section."""

    benchmarks = [
        ("Tabletop\nno CLIP", {"Wrong fused\ngrip obs.": 14, "Memory /\nlow support": 5, "Third-object\nabsorption": 0, "Controller /\ncontact": 0}),
        ("Tabletop\nwith CLIP", {"Wrong fused\ngrip obs.": 14, "Memory /\nlow support": 5, "Third-object\nabsorption": 0, "Controller /\ncontact": 0}),
        ("Closed-loop\nno CLIP", {"Wrong fused\ngrip obs.": 8, "Memory /\nlow support": 0, "Third-object\nabsorption": 11, "Controller /\ncontact": 5}),
        ("Closed-loop\nwith CLIP", {"Wrong fused\ngrip obs.": 8, "Memory /\nlow support": 1, "Third-object\nabsorption": 10, "Controller /\ncontact": 5}),
    ]
    categories = ["Wrong fused\ngrip obs.", "Memory /\nlow support", "Third-object\nabsorption", "Controller /\ncontact"]
    colors = {
        "Wrong fused\ngrip obs.": "#dc2626",
        "Memory /\nlow support": "#f59e0b",
        "Third-object\nabsorption": "#7c3aed",
        "Controller /\ncontact": "#64748b",
    }

    fig, ax = plt.subplots(figsize=(7.1, 3.25))
    y_positions = list(range(len(benchmarks)))
    left = [0] * len(benchmarks)

    for category in categories:
        values = [counts[category] for _, counts in benchmarks]
        bars = ax.barh(y_positions, values, left=left, color=colors[category], label=category, height=0.58)
        for bar, value, base in zip(bars, values, left):
            if value > 0:
                ax.text(base + value / 2, bar.get_y() + bar.get_height() / 2, str(value), ha="center", va="center", fontsize=7.5, color="white")
        left = [base + value for base, value in zip(left, values)]

    for y, total in zip(y_positions, left):
        ax.text(total + 0.45, y, f"{total} failures", va="center", fontsize=7.6, color="#334155")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for label, _ in benchmarks], fontsize=8)
    ax.set_xlabel("Failure count over 50 seeds", fontsize=8.5)
    ax.set_title("StackCube exposes target-source and association limitations", fontsize=10.5)
    ax.set_xlim(0, max(left) + 5)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=7.2, frameon=False)
    ax.invert_yaxis()
    fig.tight_layout(rect=[0.02, 0.12, 0.98, 0.97])
    save_figure(fig, output_dir / "stackcube_failure_taxonomy")


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
