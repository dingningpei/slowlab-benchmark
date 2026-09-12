#!/usr/bin/env python3
"""The figure for E6: one "experimenting begins to pay" curve per configuration."""
import sys, pathlib, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COL = {"random_spread": "#7F7F7F", "best_fixed": "#ED7D31",
       "classical_doe": "#70AD47", "gp_ucb_rep1": "#4472C4"}
NAME = {"random_spread": "Random",
        "classical_doe": "Planned design", "gp_ucb_rep1": "Adaptive design"}
MK = {"random_spread": "o", "classical_doe": "s", "gp_ucb_rep1": "^"}


def main(out="paper/figures/fig3_budget.pdf"):
    d = json.loads((ROOT / "results" / "budget_curve.json").read_text())
    cfgs = [c for c in ("Sanity", "Screen", "Optimise", "Transfer") if c in d]
    fig, axes = plt.subplots(1, len(cfgs), figsize=(2.35 * len(cfgs), 2.55),
                             gridspec_kw=dict(wspace=.34))
    for ax, c in zip(np.atleast_1d(axes), cfgs):
        for a in ("random_spread", "classical_doe", "gp_ucb_rep1"):
            g = d[c][a]; R = np.array(g["R"]); m = np.array(g["regret"]); se = np.array(g["se"])
            ax.plot(R, m, marker=MK[a], ms=3.6, lw=1.5, color=COL[a], label=NAME[a])
            ax.fill_between(R, m - se, m + se, color=COL[a], alpha=.13, lw=0)
        from slowlab import TASKS
        from slowlab.fixed_reference import best_fixed
        _, p = best_fixed(TASKS[c])
        ax.axhline(p, color=COL["best_fixed"], lw=1.4, ls="--")
        # mark the crossing point
        gm = np.array(d[c]["gp_ucb_rep1"]["regret"]); R = np.array(d[c]["gp_ucb_rep1"]["R"])
        below = np.where(gm < p)[0]
        if len(below):
            ax.axvline(R[below[0]], color="#BFBFBF", lw=.9, ls=":", zorder=0)
            ax.annotate(f"$R^\\star={R[below[0]]}$", xy=(R[below[0]], ax.get_ylim()[1]),
                        xytext=(2, -8), textcoords="offset points",
                        fontsize=7, color="#595959", va="top")
        ax.set_yscale("log"); ax.set_title(c, fontsize=9.5, loc="left")
        ax.set_xlabel("Rounds of budget $R$", fontsize=8)
        ax.set_xticks([1, 4, 7, 10])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True); ax.tick_params(labelsize=7.5)
    np.atleast_1d(axes)[0].set_ylabel("Simple regret $\\mathcal{R}(R)$", fontsize=8)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=COL["best_fixed"], ls="--", lw=1.4))
    l.append("Best fixed recommendation (no experiments)")
    fig.legend(h, l, fontsize=7.4, frameon=False, ncol=4,
               loc="lower center", bbox_to_anchor=(.5, -.13))
    o = ROOT / out; o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o, bbox_inches="tight")
    fig.savefig(str(o).replace(".pdf", ".png"), dpi=185, bbox_inches="tight")
    print("wrote", o)


if __name__ == "__main__":
    main()
