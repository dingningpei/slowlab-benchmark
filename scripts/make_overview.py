#!/usr/bin/env python3
"""Figure 1: three horizontal bands for who is acting, left to right for time.

This version fixes two problems with the old figure:
  * the agent side was missing (prompt / JSON / validation / retry / tools), so the
    Design box appeared from nowhere;
  * the Measures box read Slow / Costly / Irreversible, which came from an earlier
    draft that intended to score each of the four conditions -- and directly
    contradicts the paper's statement that we do not score the conditions.

The palette is Okabe-Ito and colour-blind safe; the middle band is deliberately
achromatic, because nothing happens there.

Note: the figure shipped in paper/figures/fig0_overview.pdf is a hand-drawn version
of this same diagram. Running this script overwrites it with the matplotlib version.
"""
from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORLD, WORLD_L = "#0072B2", "#DCEBF5"
AGENT, AGENT_L = "#E69F00", "#FBF0DA"
WAIT,  WAIT_L  = "#6E6E6E", "#F2F2F2"
SCORE, SCORE_L = "#009E73", "#DAF0E9"
INK, MUT = "#1A1A1A", "#6E6E6E"
T, S = 8.6, 7.2


def box(ax, x, y, w, h, edge, fill, lw=1.15, r=0.035):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2))


def arw(ax, p, q, color, lw=1.1, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8.5, lw=lw,
                                 color=color, zorder=4, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=1.5, shrinkB=1.5))


def main(out="paper/figures/fig0_overview.pdf"):
    fig, ax = plt.subplots(figsize=(7.4, 3.35))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.45); ax.axis("off")
    ax.text(0.42, 3.30, "SlowLab", fontsize=11.5, weight="bold", color=INK, va="center")

    # ── Band 1: the agent and the harness ─────────────────
    yA, hA = 2.24, .62
    ax.text(0.12, yA + hA / 2, "AGENT", fontsize=S - .6, color=AGENT, weight="bold",
            va="center", rotation=90, ha="center")
    steps = [(0.44, 1.16, "prompt", "factors,\nbudget"),
             (1.86, 1.10, "LLM", "one call\nper round"),
             (3.22, 1.42, "JSON design", "treatments,\nallocation"),
             (4.90, 1.20, "validate", "mechanics\nonly")]
    for x, w, n, sub in steps:
        box(ax, x, yA, w, hA, AGENT, AGENT_L)
        ax.text(x + w / 2, yA + .41, n, fontsize=T, weight="bold", color=INK, ha="center")
        ax.text(x + w / 2, yA + .17, sub, fontsize=S - .8, color=MUT, ha="center",
                linespacing=1.3)
    for i in range(3):
        arw(ax, (steps[i][0] + steps[i][1], yA + hA / 2),
            (steps[i + 1][0], yA + hA / 2), AGENT)
    # Rejection -> free retry (drawn above, with enough headroom)
    arw(ax, (5.50, yA + hA + .02), (3.93, yA + hA + .02), AGENT, rad=.34,
        ls=(0, (2.4, 1.6)))
    ax.text(4.72, yA + hA + .34, "infeasible $\\rightarrow$ free retry, counted",
            fontsize=S - .8, color=AGENT, ha="center")
    # Tools (the ablation): below left of the LLM, joined by a dashed line
    box(ax, 1.86, yA - .46, 1.10, .34, AGENT, "white", lw=.9, r=.03)
    ax.text(2.41, yA - .29, "tools (ablation)", fontsize=S - .9, color=INK, ha="center")
    arw(ax, (2.41, yA - .12), (2.41, yA), AGENT, lw=.9, ls=(0, (2, 1.5)))

    # ── Band 2: commit and wait ───────────────────────────
    # *Not a processing step.* Commitment is a property of the edge that hands the
    # design to the world: wait one cycle, unobservable throughout, irrevocable. So
    # it is drawn as a shaded band the arrow passes through, not a box, and not two
    # arrow segments -- in the code, a design that passes validate goes straight
    # into truth().
    yB, hB = 1.30, .46
    ax.text(0.12, yB + hB / 2, "COMMIT", fontsize=S - .6, color=WAIT, weight="bold",
            va="center", rotation=90, ha="center")
    ax.add_patch(Rectangle((0.44, yB), 5.66, hB, facecolor=WAIT_L,
                           edgecolor="none", zorder=1))
    ax.text(0.62, yB + .29, "one growing cycle $\\;\\cdot\\;$ 210 days $\\;\\cdot\\;$ "
            "nothing is observable", fontsize=T - .6, color=INK, ha="left")
    ax.text(0.62, yB + .12, "units stay occupied; the design cannot be recalled",
            fontsize=S - .9, color=MUT, ha="left")

    # ── Band 3: the world and the observations ────────────
    yC, hC = 0.30, .62
    ax.text(0.12, yC + hC / 2, "WORLD", fontsize=S - .6, color=WORLD, weight="bold",
            va="center", rotation=90, ha="center")
    box(ax, 0.44, yC, 1.30, hC, WORLD, WORLD_L)
    ax.text(1.09, yC + .41, "site $\\theta$", fontsize=T, weight="bold", color=INK,
            ha="center")
    ax.text(1.09, yC + .17, "from a published\nspread", fontsize=S - .8, color=MUT,
            ha="center", linespacing=1.3)
    # Crop model and observation are one box: the design enters the model, not an
    # "observation" step
    box(ax, 2.02, yC, 4.08, hC, WORLD, WORLD_L)
    ax.text(3.60, yC + .41, "TOMGRO $+$ prices", fontsize=T, weight="bold",
            color=INK, ha="center")
    ax.text(3.60, yC + .17, "one value per unit, with plant,\nloop and chamber noise",
            fontsize=S - .8, color=MUT, ha="center", linespacing=1.3)
    arw(ax, (1.74, yC + hC / 2), (2.02, yC + hC / 2), WORLD)

    # The design drops from the agent straight into the crop model, through the
    # commit band; the dashed segment inside the band marks the unobservable period
    arw(ax, (5.50, yA), (5.50, yB + hB + .02), WAIT)
    ax.add_patch(FancyArrowPatch((5.50, yB + hB + .02), (5.50, yB - .02),
                                 arrowstyle="-", lw=1.1, color=WAIT, zorder=4,
                                 linestyle=(0, (3, 2))))
    arw(ax, (5.50, yB - .02), (5.50, yC + hC), WAIT)
    ax.text(5.62, yA - .17, "design", fontsize=S - .8, color=WAIT, ha="left")

    # Observations return to the agent
    arw(ax, (6.10, yC + hC / 2), (6.40, yC + hC / 2), WORLD)
    ax.add_patch(FancyArrowPatch((6.40, yC + hC / 2), (6.40, yA + hA / 2),
                                 arrowstyle="-", lw=1.1, color=WORLD, zorder=4))
    arw(ax, (6.40, yA + hA / 2), (6.10, yA + hA / 2), WORLD)
    ax.text(6.44, 2.02, "round", fontsize=S - .7, color=WORLD, ha="left", va="center")
    ax.text(6.44, 1.88, "$r{+}1$", fontsize=S - .7, color=WORLD, ha="left", va="center")

    # ── Scoring ───────────────────────────────────────────
    box(ax, 7.26, 0.30, 2.56, 2.56, SCORE, SCORE_L)
    ax.text(8.54, 2.66, "What is scored", fontsize=T + .4, weight="bold",
            color=INK, ha="center")
    rows = [("$\\bar{\\mathcal{R}}(r)$", "realised regret,", "read every round"),
            ("$\\mathcal{R}_{\\mathrm{cum}}$", "the profit those", "experiments forwent"),
            ("$c(D_r)$", "design efficiency:", "what the design allowed"),
            ("$\\eta_r$", "decision efficiency:", "how much was realised")]
    for i, (sym, a, b) in enumerate(rows):
        y = 2.28 - i * .52
        ax.text(7.44, y, sym, fontsize=T + .6, color=SCORE, va="center")
        ax.text(8.02, y + .09, a, fontsize=S - .5, color=INK, va="center")
        ax.text(8.02, y - .09, b, fontsize=S - 1.1, color=MUT, va="center")
    ax.text(8.54, 0.44, "the four conditions are not scored", fontsize=S - 1.4,
            color=MUT, ha="center", style="italic")
    arw(ax, (6.86, 1.53), (7.26, 1.53), SCORE)

    fig.savefig(ROOT / out, bbox_inches="tight", pad_inches=0.02)
    print("wrote", out)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm"})
    main()
