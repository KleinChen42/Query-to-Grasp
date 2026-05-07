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
    """Draw a variable-flow overview for the Query-to-Grasp diagnostic system."""

    fig, ax = plt.subplots(figsize=(7.35, 3.15))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(label: str, x: float, y: float, w: float, h: float, color: str, fontsize: float = 6.8) -> None:
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            linespacing=1.15,
            bbox=dict(
                boxstyle="round,pad=0.23",
                facecolor=color,
                edgecolor="#334155",
                linewidth=0.9,
            ),
        )

    def arrow(x0: float, y0: float, x1: float, y1: float, label: str = "") -> None:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", linewidth=1.05, color="#334155", shrinkA=2, shrinkB=2),
        )
        if label:
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2 + 0.035,
                label,
                ha="center",
                va="center",
                fontsize=5.7,
                color="#334155",
            )

    ax.text(0.02, 0.94, "Inputs", fontsize=8.0, fontweight="bold", color="#0f172a")
    ax.text(0.25, 0.94, "Retrieval to 3D action targets", fontsize=8.0, fontweight="bold", color="#0f172a")
    ax.text(0.75, 0.94, "Execution and diagnosis", fontsize=8.0, fontweight="bold", color="#0f172a")

    # Main variable flow.
    box("Query\n$q$", 0.055, 0.73, 0.08, 0.10, "#dbeafe")
    box("RGB-D views\n$(I_v,D_v,T_v)$", 0.055, 0.51, 0.11, 0.12, "#e0f2fe")
    box("2D grounding\n$b_i,c_i,s_i$", 0.175, 0.62, 0.11, 0.13, "#dcfce7")
    box("Lift + align\n$X_i, g_i$", 0.315, 0.62, 0.11, 0.13, "#fef3c7")
    box("Object memory\n$m_k=\\{X,g,v,s\\}$", 0.455, 0.62, 0.12, 0.13, "#ede9fe")
    box("Target source\nSC/RG/MG/TG\nQ/OP/PG", 0.610, 0.62, 0.12, 0.15, "#ffedd5")
    box("Executor\npick/place", 0.765, 0.62, 0.10, 0.12, "#dcfce7")
    box("Metrics\npick/place/task\nfailure type", 0.910, 0.62, 0.10, 0.13, "#f1f5f9")

    arrow(0.085, 0.72, 0.130, 0.65)
    arrow(0.092, 0.52, 0.130, 0.59)
    arrow(0.225, 0.62, 0.265, 0.62)
    arrow(0.365, 0.62, 0.405, 0.62)
    arrow(0.510, 0.62, 0.555, 0.62)
    arrow(0.665, 0.62, 0.715, 0.62)
    arrow(0.815, 0.62, 0.860, 0.62)

    # StackCube reference branch.
    box("Reference query\n$q_p$: green cube", 0.315, 0.36, 0.12, 0.10, "#e0f2fe", fontsize=6.5)
    box("Place reference\n$p_B$ or OB", 0.610, 0.36, 0.12, 0.10, "#ffe4e6", fontsize=6.5)
    arrow(0.370, 0.36, 0.550, 0.36, "StackCube branch")
    arrow(0.610, 0.41, 0.610, 0.535)

    # Diagnostic taps are shown below the variable flow, not as another module chain.
    diagnostics = [
        ("CLIP top-1\nchange", 0.175),
        ("spread +\nfragmentation", 0.315),
        ("views +\ngeom. conf.", 0.455),
        ("target-source\nablation", 0.610),
        ("stage counts\n+ RawEnv", 0.765),
    ]
    for label, x in diagnostics:
        ax.text(
            x,
            0.18,
            label,
            ha="center",
            va="center",
            fontsize=6.2,
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=0.75),
        )
        arrow(x, 0.28, x, 0.245)

    ax.text(
        0.50,
        0.06,
        "Diagnostic boundary: modules are separable; oracle/noisy-oracle rows are privileged probes, not deployable perception claims.",
        ha="center",
        va="center",
        fontsize=7.0,
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
    """Draw a compact primary-metric ladder without overloaded grouped bars."""

    rows = [
        ("PickCube refined grasp\nRG -> sim_topdown", 1.0000, "pick", "#2563eb"),
        ("PickCube memory grasp\nMG -> sim_topdown", 1.0000, "pick", "#2563eb"),
        ("StackCube pick-only\nTG tabletop", 0.6200, "pick", "#0f766e"),
        ("StackCube pick-only\nTG closed-loop", 0.5200, "pick", "#0f766e"),
        ("Q-pick + predicted place\nsingle", 0.5520, "task", "#ea580c"),
        ("Q-pick + predicted place\ntabletop", 0.4720, "task", "#ea580c"),
        ("Q-pick + predicted place\nclosed-loop", 0.4460, "task", "#ea580c"),
        ("Q-pick + oracle place\nsingle", 0.7200, "task", "#7c3aed"),
        ("Oracle pick-place\nOA -> OB", 0.8800, "task", "#7c3aed"),
    ]

    fig, ax = plt.subplots(figsize=(7.25, 3.55))
    y_positions = list(range(len(rows)))
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = [row[3] for row in rows]

    for y, value, metric, color in zip(y_positions, values, [row[2] for row in rows], colors):
        ax.hlines(y, 0, value, color=color, linewidth=2.4, alpha=0.8)
        ax.scatter(value, y, s=46, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(value + 0.025, y, f"{value:.3f} {metric}", va="center", fontsize=7.2, color="#1e293b")

    ax.axvline(1.0, color="#94a3b8", linestyle=":", linewidth=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=7.1)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Primary success rate", fontsize=8.5)
    ax.set_title("Target-source ladder: from query targets to diagnostic upper bounds", fontsize=10.0)
    ax.grid(axis="x", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="#2563eb", label="PickCube pick", markersize=5, linewidth=2),
        plt.Line2D([0], [0], marker="o", color="#0f766e", label="StackCube pick-only", markersize=5, linewidth=2),
        plt.Line2D([0], [0], marker="o", color="#ea580c", label="Non-oracle place bridge", markersize=5, linewidth=2),
        plt.Line2D([0], [0], marker="o", color="#7c3aed", label="Privileged diagnostic", markersize=5, linewidth=2),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.56, -0.18),
        ncol=4,
        fontsize=6.8,
        frameon=False,
        columnspacing=1.1,
        handlelength=1.8,
    )
    fig.text(
        0.5,
        0.005,
        "Pick-only rows use pick success as the primary metric; pick-place rows use task success. Oracle rows are diagnostic probes.",
        ha="center",
        fontsize=7.2,
        color="#475569",
    )
    fig.tight_layout(rect=[0.02, 0.16, 0.98, 0.96])
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
