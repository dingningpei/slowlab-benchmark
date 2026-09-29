#!/usr/bin/env python3
"""Two figures for the paper: the trajectory (where Slow + Irreversible come from) and
transfer."""
from __future__ import annotations
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAME = {"random_spread": "Random", "classical_doe": "Planned design",
        "gp_ucb_rep1": "Adaptive design"}
ORDER = ["random_spread", "classical_doe", "gp_ucb_rep1"]
COL = {"random_spread": "#9AA79B", "classical_doe": "#4A6FA5", "gp_ucb_rep1": "#C8622A"}
MK = {"random_spread": "s", "classical_doe": "^", "gp_ucb_rep1": "o"}


def trajectory(out="paper/figures/fig1_trajectory.pdf"):
    d = json.loads((ROOT / "results" / "curve_T3.json").read_text())
    R = len(d[ORDER[0]]["mean"]); x = np.arange(1, R + 1)
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw=dict(wspace=.30))
    for a in ORDER:
        m = np.array(d[a]["mean"]); se = np.array(d[a]["se"])
        ax[0].plot(x, m, marker=MK[a], ms=4.2, lw=1.6, color=COL[a], label=NAME[a])
        ax[0].fill_between(x, m - se, m + se, color=COL[a], alpha=.14, lw=0)
        ax[1].plot(x, m / m[0], marker=MK[a], ms=4.2, lw=1.6, color=COL[a])
    # The no-experiment ceiling (a computable oracle constant). Where the curve crosses
    # it is where experimenting begins to pay.
    try:
        import sys as _s; _s.path.insert(0, str(ROOT))
        from slowlab import TASKS
        from slowlab.fixed_reference import best_fixed
        _, lp = best_fixed(TASKS["Optimise"])
        ax[0].axhline(lp, color="#7F7F7F", lw=1.1, ls="--", zorder=1)
        ax[0].annotate("best fixed recommendation (no experiments)", xy=(R, lp),
                       xytext=(-4, 4), textcoords="offset points", ha="right",
                       va="bottom", fontsize=7, color="#7F7F7F")
    except Exception:
        pass
    ax[0].set_yscale("log"); ax[0].set_xticks(x)
    ax[0].set_xlabel("Round (one growing cycle each)")
    ax[0].set_ylabel("Simple regret")
    ax[0].set_title("(a) What each cycle buys", fontsize=9.5, loc="left")
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].axhline(1.0, color="#888", lw=.9, ls=":")
    ax[1].set_xticks(x); ax[1].set_ylim(0, 1.05)
    ax[1].set_xlabel("Round")
    ax[1].set_ylabel("Fraction of round-1 regret\nstill remaining")
    ax[1].set_title("(b) Fraction of the initial regret\nstill remaining",
                    fontsize=9.5, loc="left")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=.25, lw=.6); a.set_axisbelow(True); a.tick_params(labelsize=8)
    o = ROOT / out; o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o, bbox_inches="tight"); fig.savefig(str(o).replace(".pdf", ".png"), dpi=185, bbox_inches="tight")
    print("wrote", o)


def transfer(out="paper/figures/fig2_transfer.pdf"):
    d = json.loads((ROOT / "results" / "transfer_T4.json").read_text())
    A, B = "gp_ucb_profit", "gp_ucb_components"
    ORANGE, GREEN = "#C8622A", "#2C5F2D"
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.95), gridspec_kw=dict(wspace=.32))
    x = np.arange(2); w = .36
    ax[0].bar(x - w / 2, [d[A]["simple"], d[A]["transfer"]], w,
              yerr=[d[A]["simple_se"], d[A]["transfer_se"]], capsize=3,
              color=ORANGE, label="models profit", edgecolor="none")
    ax[0].bar(x + w / 2, [d[B]["simple"], d[B]["transfer"]], w,
              yerr=[d[B]["simple_se"], d[B]["transfer_se"]], capsize=3,
              color=GREEN, label="models yield and cost", edgecolor="none")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(["Training\n$\\lambda=0.5$", "After price change\n$\\lambda=1.0$"], fontsize=8)
    ax[0].set_ylabel("Simple regret"); ax[0].legend(fontsize=7.4, frameon=False)
    ax[0].set_title("(a) Same data, different model", fontsize=9.5, loc="left")
    ax[0].annotate("identical", (0, d[A]["simple"]), xytext=(0, 16),
                   textcoords="offset points", ha="center", fontsize=7.6,
                   color="#444", weight="bold")
    ax[0].annotate("$1.7\\times$", (1, max(d[A]["transfer"], d[B]["transfer"])),
                   xytext=(0, 14), textcoords="offset points", ha="center",
                   fontsize=8, color=ORANGE, weight="bold")
    a_, b_ = np.array(d["per_seed"][A]["transfer"]), np.array(d["per_seed"][B]["transfer"])
    m = max(a_.max(), b_.max()) * 1.06
    ax[1].plot([0, m], [0, m], ls="--", lw=1, color="#888")
    ax[1].scatter(b_, a_, s=26, color=ORANGE, alpha=.75, edgecolor="white", lw=.6)
    ax[1].set_xlim(0, m); ax[1].set_ylim(0, m)
    ax[1].set_xlabel("Transfer regret · models yield and cost")
    ax[1].set_ylabel("Transfer regret · models profit")
    ax[1].set_title("(b) Paired, one point per site", fontsize=9.5, loc="left")
    ax[1].text(.04 * m, .92 * m,
               f"worse for the profit-only\nmodel at {int((a_ > b_).sum())}/{len(a_)} sites\n$t=4.3$",
               fontsize=7.6, color=ORANGE, va="top")
    for q in ax:
        q.spines[["top", "right"]].set_visible(False)
        q.grid(alpha=.25, lw=.6); q.set_axisbelow(True); q.tick_params(labelsize=8)
    o = ROOT / out; o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o, bbox_inches="tight"); fig.savefig(str(o).replace(".pdf", ".png"), dpi=185, bbox_inches="tight")
    print("wrote", o)


if __name__ == "__main__":
    trajectory(); transfer()
