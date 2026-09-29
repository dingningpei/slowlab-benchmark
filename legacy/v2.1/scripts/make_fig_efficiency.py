"""Figure 2 — design efficiency c vs decision efficiency eta, one point per agent/task.

Values are taken directly from Table 3 (paper/sections/_results_table.tex) rather
than from results/*.json, because after 2026-09-09 the world those JSON files
describe no longer exists. Once they are re-run, switch back to reading the result
files.

    python scripts/make_fig_efficiency.py
"""
from __future__ import annotations
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parents[1] / "paper" / "figures"

# (task, agent, c%, eta%, is_reference)
DATA = [
    ("Sanity",   "DeepSeek-V4-flash", 39, 9,  False),
    ("Sanity",   "MiMo-v2.5",         38, 10, False),
    ("Sanity",   "GLM-5.3-flash",     39, 20, False),
    ("Sanity",   "Bayesian opt.",     49, 49, True),
    ("Optimise", "DeepSeek-V4-flash", 35, 10, False),
    ("Optimise", "MiMo-v2.5",         34, 12, False),
    ("Optimise", "GLM-5.3-flash",     45, 26, False),
    ("Optimise", "Bayesian opt.",     37, 45, True),
    ("Screen",   "DeepSeek-V4-flash", 22, 1,  False),
    ("Screen",   "MiMo-v2.5",         33, 1,  False),
    ("Screen",   "GLM-5.3-flash",     39, 1,  False),
    ("Screen",   "Bayesian opt.",     47, 1,  True),
    ("Transfer", "DeepSeek-V4-flash", 26, 1,  False),
    ("Transfer", "MiMo-v2.5",         27, 1,  False),
    ("Transfer", "GLM-5.3-flash",     32, 1,  False),
    ("Transfer", "Bayesian opt.",     34, 1,  True),
]

LEFT, RIGHT = ("Sanity", "Optimise"), ("Screen", "Transfer")
MARK = {"Sanity": "o", "Optimise": "s", "Screen": "^", "Transfer": "D"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for ax, tasks, title in ((axes[0], LEFT, "ceiling reachable"),
                             (axes[1], RIGHT, "ceiling out of reach")):
        for task, agent, c, eta, ref in DATA:
            if task not in tasks:
                continue
            ax.scatter(c, eta, marker=MARK[task], s=64,
                       facecolors="none" if ref else "black",
                       edgecolors="black", linewidths=1.2, zorder=3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("design efficiency $c$ (%)")
        ax.set_xlim(15, 60); ax.set_ylim(-3, 55)
        ax.grid(alpha=0.25, linewidth=0.5)
        for t in tasks:
            ax.scatter([], [], marker=MARK[t], s=64, facecolors="none",
                       edgecolors="black", label=t)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel(r"decision efficiency $\eta$ (%)")
    axes[0].annotate("references", (49, 49), (41, 44), fontsize=8,
                     arrowprops=dict(arrowstyle="-", lw=0.6))
    axes[0].annotate("models", (39, 20), (30, 24), fontsize=8,
                     arrowprops=dict(arrowstyle="-", lw=0.6))
    axes[1].annotate(r"every agent at $\eta \approx 1\%$", (34, 1), (33, 10),
                     fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.6))
    fig.text(0.5, -0.02, "filled = LLM agent    hollow = reference strategy",
             ha="center", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_efficiency.{ext}", bbox_inches="tight", dpi=200)
    print("wrote", OUT / "fig2_efficiency.pdf")


if __name__ == "__main__":
    main()
