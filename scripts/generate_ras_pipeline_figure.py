"""Generate the RAS Query-to-Grasp pipeline as deterministic vector artwork."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("paper_ras") / "figures"


def add_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    face: str = "#eef4fb",
    edge: str = "#4f5b66",
    fontsize: int = 8,
    lw: float = 0.9,
    radius: float = 0.07,
    weight: str = "regular",
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
        mutation_aspect=1,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1f2933",
        weight=weight,
        linespacing=1.15,
    )
    return patch


def arrow(ax, start, end, *, color="#3f4750", lw=1.1, style="-|>", rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def header(ax, x0, x1, title):
    y = 6.55
    ax.plot([x0, x1], [y, y], color="#1f2933", lw=0.8)
    ax.plot([x0, x0], [y, y - 0.10], color="#1f2933", lw=0.8)
    ax.plot([x1, x1], [y, y - 0.10], color="#1f2933", lw=0.8)
    ax.text((x0 + x1) / 2, y + 0.13, title, ha="center", va="bottom", fontsize=11, weight="bold")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.0, 6.2))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 7.0)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    header(ax, 0.35, 2.25, "Inputs")
    header(ax, 2.55, 9.35, "Retrieval to 3D Action Targets")
    header(ax, 9.65, 12.65, "Execution and Diagnosis")

    # Main flow boxes.
    query = add_box(ax, 0.35, 5.25, 1.75, 0.55, "Language query $q$", face="#e7f1fb")
    rgbd = add_box(ax, 0.35, 4.45, 1.75, 0.65, "RGB-D views\n$(I_v,D_v,T_v)$", face="#e7f1fb")

    grounding = add_box(
        ax,
        2.55,
        4.70,
        1.75,
        1.05,
        "Open-vocabulary\n2D grounding\n\n$\\{b_i, \\ell_i, s_i\\}$",
        face="#eef8e8",
        fontsize=8,
        weight="bold",
    )
    lifting = add_box(
        ax,
        4.75,
        4.70,
        1.75,
        1.05,
        "RGB-D lifting +\ncamera alignment\n\n$X_i$, grasp $g_i$",
        face="#fff7df",
        fontsize=8,
        weight="bold",
    )
    memory = add_box(
        ax,
        6.95,
        4.70,
        1.65,
        1.05,
        "Multi-view\nobject memory\n\n$m_k=\\{X,g,V,s\\}$",
        face="#f0ecfb",
        fontsize=8,
        weight="bold",
    )
    ladder = add_box(
        ax,
        8.95,
        4.55,
        1.75,
        1.35,
        "Target-source ladder\n\nSC semantic center\nRG/MG grasp target\nTG task guard\nOP oracle pose\nPG predicted ref.",
        face="#fff0dc",
        edge="#c67821",
        fontsize=7.3,
        weight="bold",
    )
    executor = add_box(
        ax,
        11.0,
        4.75,
        1.35,
        1.0,
        "Scripted executor\n\nsim_topdown\nsim_pick_place",
        face="#edf8ea",
        fontsize=8,
        weight="bold",
    )
    metrics = add_box(
        ax,
        11.0,
        3.25,
        1.35,
        0.85,
        "Metrics + failure\nattribution\n\npick, place,\nRawEnv, type",
        face="#f5f5f5",
        fontsize=7.5,
        weight="bold",
    )

    # Main arrows.
    arrow(ax, (2.10, 5.52), (2.55, 5.25))
    arrow(ax, (2.10, 4.75), (2.55, 5.15))
    arrow(ax, (4.30, 5.22), (4.75, 5.22))
    arrow(ax, (6.50, 5.22), (6.95, 5.22))
    arrow(ax, (8.60, 5.22), (8.95, 5.22))
    arrow(ax, (10.70, 5.22), (11.00, 5.22))
    arrow(ax, (11.68, 4.75), (11.68, 4.10))

    # Diagnostic probes band.
    diag = FancyBboxPatch(
        (2.55, 1.10),
        9.80,
        0.82,
        boxstyle="round,pad=0.018,rounding_size=0.06",
        linewidth=0.8,
        edgecolor="#9aa5b1",
        facecolor="#f6f8fa",
    )
    ax.add_patch(diag)
    ax.text(2.75, 1.67, "Diagnostic probes", ha="left", va="center", fontsize=8.5, weight="bold", color="#3f4750")
    probes = [
        "CLIP top-1\nchange",
        "cross-view\nspread",
        "geometry\nconfidence",
        "memory\nfragmentation",
        "target-source\nablation",
        "noisy-oracle\nsensitivity",
        "failure\ntaxonomy",
    ]
    x = 3.75
    for p in probes:
        add_box(ax, x, 1.25, 0.95, 0.45, p, face="#ffffff", edge="#b7c0c9", fontsize=6.7, radius=0.04, lw=0.7)
        x += 1.18

    # Light diagnostic connectors.
    for sx in [3.42, 5.62, 7.78, 9.82, 11.68]:
        arrow(ax, (sx, 4.65), (sx, 1.93), color="#8b949e", lw=0.7, style="-", rad=0.0)

    # StackCube reference branch.
    ref_q = add_box(ax, 5.05, 2.70, 1.45, 0.50, "Reference query $q_p$\n\"green cube\"", face="#fdeaf1", fontsize=7.2)
    pred_ref = add_box(ax, 6.80, 2.70, 1.45, 0.50, "Predicted place\nreference $p_B$", face="#fdeaf1", fontsize=7.2)
    oracle_ref = add_box(ax, 8.45, 2.70, 1.35, 0.50, "or privileged\noracle cubeB pose", face="#ffffff", edge="#8b949e", fontsize=6.9, lw=0.8)
    arrow(ax, (6.50, 2.95), (6.80, 2.95))
    arrow(ax, (8.25, 2.95), (8.45, 2.95), style="-")
    arrow(ax, (9.13, 3.20), (9.82, 4.55), rad=-0.12)
    ax.text(
        6.9,
        0.38,
        "Oracle and noisy-oracle rows are privileged diagnostics, not deployable perception claims.",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#4b5563",
    )

    plt.tight_layout(pad=0.3)
    fig.savefig(OUT_DIR / "pipeline_overview_vector.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "pipeline_overview_vector.svg", bbox_inches="tight")
    fig.savefig(OUT_DIR / "pipeline_overview_vector.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
