#!/usr/bin/env python3
"""Figure 1: the design, reader, environment, and evaluator information flow."""
from __future__ import annotations

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENT, AGENT_L = "#E69F00", "#FBF0DA"
WORLD, WORLD_L = "#0072B2", "#DCEBF5"
SCORE, SCORE_L = "#009E73", "#DAF0E9"
INK, MUT = "#1A1A1A", "#666666"


def box(ax, x, y, w, h, edge, fill, lw=1.2, radius=.055):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.015,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))


def arrow(ax, start, end, color, *, dashed=False, rad=0, lw=1.15):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=9, linewidth=lw,
        color=color, linestyle=(0, (3, 2)) if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=4))


def main(out="paper/figures/fig0_overview.pdf"):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.15); ax.axis("off")

    ax.text(.25, 4.08, "AGENT", color=AGENT, fontsize=8, fontweight="bold")
    box(ax, .25, 2.62, 2.65, .92, AGENT, AGENT_L)
    ax.text(1.575, 3.30, "DESIGN POLICY", ha="center", fontsize=9, fontweight="bold")
    ax.text(1.575, 3.02, "choose treatments, replication,", ha="center", fontsize=7.3)
    ax.text(1.575, 2.82, "and physical allocation  $D_r$", ha="center", fontsize=7.3)
    box(ax, 3.30, 2.62, 2.65, .92, AGENT, AGENT_L)
    ax.text(4.625, 3.30, "HISTORY READER", ha="center", fontsize=9, fontweight="bold")
    ax.text(4.625, 3.02, "interpret visible history  $H_t$", ha="center", fontsize=7.3)
    ax.text(4.625, 2.82, "and recommend  $\\hat{x}_t$", ha="center", fontsize=7.3)

    box(ax, .55, 3.64, 1.95, .34, AGENT, "white", lw=.9, radius=.035)
    ax.text(1.525, 3.81, "design aid (optional)", ha="center", va="center", fontsize=7)
    arrow(ax, (1.525, 3.64), (1.525, 3.54), AGENT, dashed=True, lw=.9)
    box(ax, 3.65, 3.64, 1.95, .34, AGENT, "white", lw=.9, radius=.035)
    ax.text(4.625, 3.81, "inference aid (optional)", ha="center", va="center", fontsize=7)
    arrow(ax, (4.625, 3.64), (4.625, 3.54), AGENT, dashed=True, lw=.9)

    ax.text(.25, 2.27, "EXPERIMENT", color=WORLD, fontsize=8, fontweight="bold")
    box(ax, .25, .62, 5.70, 1.42, WORLD, WORLD_L)
    ax.text(.48, 1.76, "FACILITY + LATENT SITE $\\theta$", fontsize=9, fontweight="bold")
    ax.text(.48, 1.45, "commit named units for a 210-day crop cycle", fontsize=7.4)
    ax.text(.48, 1.20, "treatments remain immutable; occupied capacity cannot be released", fontsize=7.4)
    ax.text(.48, .91, "timestamped interim measurements  $\\rightarrow$  terminal outcomes",
            fontsize=7.4, color=WORLD)

    arrow(ax, (1.575, 2.62), (1.575, 2.04), WORLD)
    ax.text(1.70, 2.30, "$D_r$", color=WORLD, fontsize=8)
    arrow(ax, (3.78, 2.04), (3.78, 2.62), WORLD)
    ax.text(3.92, 2.29, "$H_t$", color=WORLD, fontsize=8)
    arrow(ax, (3.55, 2.04), (2.30, 2.62), WORLD, rad=-.18)
    ax.text(2.69, 2.27, "next round", color=WORLD, fontsize=6.8)

    ax.text(6.40, 4.08, "BENCHMARK ONLY", color=SCORE, fontsize=8, fontweight="bold")
    box(ax, 6.40, .62, 3.35, 2.92, SCORE, SCORE_L)
    ax.text(8.075, 3.28, "EVALUATOR", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(8.075, 2.94, "no information returns to the agent", ha="center",
            fontsize=7.1, color=MUT, style="italic")
    ax.text(6.68, 2.55, "Exact outcomes", fontsize=8, fontweight="bold", color=SCORE)
    ax.text(6.68, 2.30, "$\\bar{\\mathcal{R}}(r)$  recommendation regret", fontsize=7.3)
    ax.text(6.68, 2.07, "$\\mathcal{R}_{\\mathrm{cum}}$  experimental opportunity cost", fontsize=7.3)
    ax.text(6.68, 1.67, "Common-history diagnostics", fontsize=8,
            fontweight="bold", color=SCORE)
    ax.text(6.68, 1.42, "$b(H_t)$  risk left by the visible history", fontsize=7.3)
    ax.text(6.68, 1.19, "$g(H_t,\\hat{x}_t)$  excess risk of the recommendation", fontsize=7.3)
    ax.text(6.68, .82, "access: recommendation + hidden truth/model", fontsize=7, color=MUT)

    arrow(ax, (5.95, 3.05), (6.40, 3.05), SCORE)
    ax.text(6.03, 3.18, "$\\hat{x}_t$", fontsize=7.5, color=SCORE)
    ax.text(.25, .27,
            "A closed-loop agent may implement design and reading in one model; fixed-history replay separates the roles experimentally.",
            fontsize=7.2, color=MUT)
    fig.savefig(ROOT / out, bbox_inches="tight", pad_inches=.03)
    print("wrote", out)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm"})
    main()
