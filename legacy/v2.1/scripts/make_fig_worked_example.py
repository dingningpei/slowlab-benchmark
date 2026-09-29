#!/usr/bin/env python3
"""Figure 2: a worked example of one concrete round.

Two designs at the same site (seed=0) on the same one-round budget of 12 units:
  A  a 4-temperature x 3-density full factorial, n=1 per cell -- mechanically
     feasible, pure-error df = 0
  B  2 temperatures (split-plot, 2 chambers per level) x 3 densities, n=2 per
     treatment -- passes all four checks
Every number in the figure comes from this script actually running SlowLabEnv.
"""
from __future__ import annotations
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.tasks import TASKS
from slowlab.env import SlowLabEnv
from slowlab.design import Design
from slowlab.validity import score_validity

ORANGE, GREEN, GREY = "#C8622A", "#2C5F2D", "#C2CBBE"
TASK = TASKS["T3"]
TEMP, DENS = TASK.factors


# ---------------------------------------------------------------- design construction
def design_A():
    tr, al = {}, {}
    for ci, tv in enumerate([0.0, 0.333, 0.667, 1.0]):
        for li, dv in enumerate([0.0, 0.5, 1.0]):
            k = f"T{ci}D{li}"
            tr[k] = {"day_temp": tv, "density": dv}
            al[k] = [f"c{ci}l{li}"]
    return Design(treatments=tr, allocation=al, randomization_seed=1)


def design_B():
    rng = np.random.default_rng(7)
    Tlv = {"lo": 0.25, "hi": 0.75}
    Dlv = {"d1": 0.2, "d2": 0.5, "d3": 0.8}
    tr, al = {}, {}
    for tn, tv in Tlv.items():
        for dn, dv in Dlv.items():
            tr[f"{tn}_{dn}"] = {"day_temp": tv, "density": dv}
            al[f"{tn}_{dn}"] = []
    for tn, cs in {"lo": [0, 3], "hi": [1, 2]}.items():   # chamber assignment is randomised too
        for c in cs:
            order = list(Dlv); rng.shuffle(order)
            for pos, dn in enumerate(order):
                al[f"{tn}_{dn}"].append(f"c{c}l{pos}")
    return Design(treatments=tr, allocation=al, randomization_seed=7)


def run(design):
    env = SlowLabEnv(TASK, seed=0)
    rep = score_validity(design, env.facility, TASK)
    env.submit_design(design); env.advance()
    obs = {o.unit_id: o.value for o in env.observations()}
    return env, rep, obs


# ---------------------------------------------------------------- drawing components
def grid_panel(ax, design, obs, title, ok, note):
    """Facility map, 4 chambers x 3 loops: background is day temperature, the cell shows density, the corner shows the observation."""
    cmap = plt.get_cmap("RdYlBu_r")
    def shade(t):                      # compressed into the light range so black text stays readable
        return cmap(0.18 + 0.64 * (t - 18.0) / 14.0)
    unit2t = {}
    for tid, units in design.allocation.items():
        for u in units:
            unit2t[u] = design.treatments[tid]
    for c in range(4):
        for l in range(3):
            u = f"c{c}l{l}"
            fv = unit2t[u]
            t_real, d_real = TEMP.denorm(fv["day_temp"]), DENS.denorm(fv["density"])
            col = shade(t_real)
            ax.add_patch(Rectangle((c, 2 - l), 1, 1, facecolor=col,
                                   edgecolor="white", lw=1.6))
            ax.text(c + .5, 2 - l + .62, f"{t_real:.0f}$\\degree$C",
                    ha="center", va="center", fontsize=7.4, color="#111")
            ax.text(c + .5, 2 - l + .38, f"{d_real:.2f}",
                    ha="center", va="center", fontsize=7.4, color="#333")
            ax.text(c + .95, 2 - l + .06, f"{obs[u]:+.3f}", ha="right", va="bottom",
                    fontsize=6.2, color="#444", style="italic")
    ax.set_xlim(0, 4); ax.set_ylim(0, 3)
    ax.set_xticks(np.arange(4) + .5); ax.set_xticklabels([f"chamber {i}" for i in range(4)], fontsize=7)
    ax.set_yticks(np.arange(3) + .5); ax.set_yticklabels([f"loop {i}" for i in (2, 1, 0)], fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title, fontsize=9, loc="left")
    ax.text(0, -.30, note, transform=ax.transAxes, fontsize=7.8,
            color=GREEN if ok else ORANGE, weight="bold")


def response_panel(ax, env, temp_norm, dens_pts, groups, title, truth_label,
                   rec=None, true_opt=None, chamber_col=None):
    """Density profile at one temperature: the true curve plus the observations."""
    gx = np.linspace(0, 1, 60)
    X = np.column_stack([np.full_like(gx, temp_norm), gx])
    ax.plot([DENS.denorm(v) for v in gx], env.truth(X), color="#444", lw=1.4,
            zorder=1, label=truth_label)
    for dx, vals in zip(dens_pts, groups):
        if chamber_col is None:
            ax.plot(dx, vals[0], "o", ms=6.5, color=ORANGE, zorder=3)
        else:
            m, se = np.mean(vals), np.std(vals, ddof=1) / np.sqrt(len(vals))
            ax.errorbar(dx, m, yerr=se, fmt="o", ms=6.5, color=GREEN,
                        capsize=3.5, lw=1.4, zorder=3)
            for v, cc in zip(vals, chamber_col):
                ax.plot(dx, v, "_", ms=11, mew=1.6, color=cc, zorder=2)
    if rec is not None:
        ax.axvline(rec, color=ORANGE, ls=":", lw=1.3, zorder=0)
    if true_opt is not None:
        ax.axvline(true_opt, color="#444", ls="--", lw=1.1, zorder=0)
    ax.set_xlabel("planting density  (plants m$^{-2}$)", fontsize=8)
    ax.set_ylabel("net profit  (EUR m$^{-2}$ d$^{-1}$)", fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25, lw=.6); ax.set_axisbelow(True)
    ax.tick_params(labelsize=7.5)


def main(out="paper/figures/fig2_worked_example.pdf"):
    A, B = design_A(), design_B()
    envA, repA, obsA = run(A)
    envB, repB, obsB = run(B)
    dfA = sum(n - 1 for n in A.replicates().values())
    dfB = sum(n - 1 for n in B.replicates().values())
    assert not repA.passed and repB.passed, (repA.checks, repB.checks)

    fig, ax = plt.subplots(2, 2, figsize=(7.4, 4.55),
                           gridspec_kw=dict(height_ratios=[1, 1.18], hspace=.60, wspace=.28))

    grid_panel(ax[0, 0], A, obsA, "(a) Design A: 4 temperatures $\\times$ 3 densities, $n=1$",
               False, f"rejected: pure-error df $=$ {dfA}")
    grid_panel(ax[0, 1], B, obsB, "(b) Design B: split-plot, 2 temperatures $\\times$ 3 densities, $n=2$",
               True, f"accepted: pure-error df $=$ {dfB}")

    # (c) design A's row at 22.7 degrees
    # Every annotation is computed from the data. true_opt=4.5 and the ylim were
    # once hard-coded, and printed false numbers the moment the environment changed.
    dens_A = [DENS.denorm(v) for v in (0.0, 0.5, 1.0)]
    valsA = [[obsA[f"c1l{i}"]] for i in range(3)]
    gx = np.linspace(0, 1, 400)
    curveA = envA.truth(np.column_stack([np.full_like(gx, 0.333), gx]))
    opt_d = DENS.denorm(float(gx[int(np.argmax(curveA))])); opt_v = float(np.max(curveA))
    rec_d = dens_A[int(np.argmax([v[0] for v in valsA]))]
    response_panel(ax[1, 0], envA, 0.333, dens_A, valsA,
                   "(c) Design A has no way to check itself",
                   "true response at 22.7$\\degree$C", rec=rec_d, true_opt=opt_d)
    ax[1, 0].annotate(f"recommended\n{rec_d:.2f}",
                      (rec_d, max(v[0] for v in valsA)), fontsize=7.4,
                      color=ORANGE, xytext=(-7, -2), textcoords="offset points",
                      ha="right", va="top", weight="bold")
    ax[1, 0].annotate(f"true optimum\n{opt_d:.2f}", (opt_d, opt_v), fontsize=7.4,
                      color="#333", xytext=(-8, -30), textcoords="offset points",
                      ha="right", arrowprops=dict(arrowstyle="->", lw=.9, color="#333"))
    ax[1, 0].text(.02, .035, "no replication $\\Rightarrow$ no error bar is possible",
                  transform=ax[1, 0].transAxes, fontsize=7.2, color=ORANGE, style="italic")

    # (d) design B's row at 21.5 degrees, with points coloured by chamber
    dens_B, valsB, cols = [], [], []
    for dn, dv in [("d1", 0.2), ("d2", 0.5), ("d3", 0.8)]:
        units = B.allocation[f"lo_{dn}"]
        dens_B.append(DENS.denorm(dv))
        valsB.append([obsB[u] for u in units])
        cols.append(["#2E6F9E" if u.startswith("c0") else "#B03A5B" for u in units])
    response_panel(ax[1, 1], envB, 0.25, dens_B, valsB,
                   "(d) Design B can separate chamber from treatment",
                   "true response at 21.5$\\degree$C", chamber_col=cols[0])
    for dx, vv, cc in zip(dens_B, valsB, cols):
        for v, c in zip(vv, cc):
            ax[1, 1].plot(dx, v, "_", ms=12, mew=1.8, color=c, zorder=4)
    ax[1, 1].plot([], [], "_", ms=10, mew=1.8, color="#2E6F9E", label="chamber 0")
    ax[1, 1].plot([], [], "_", ms=10, mew=1.8, color="#B03A5B", label="chamber 3")
    spread = abs(np.mean([v[0] - v[1] for v in valsB]))
    eff = max(np.mean(v) for v in valsB) - min(np.mean(v) for v in valsB)
    rel = "$>$" if spread > eff else "$<$"          # the direction comes from the data too
    ax[1, 1].text(.02, .035,
                  f"chamber gap {spread:.3f} {rel} density effect {eff:.3f}",
                  transform=ax[1, 1].transAxes, fontsize=7.2, color=GREEN, style="italic")

    ax[1, 0].legend(fontsize=6.8, frameon=False, loc="upper left")
    ax[1, 1].legend(fontsize=6.8, frameon=False, loc="upper right", ncol=1)

    outp = ROOT / out
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, bbox_inches="tight")
    fig.savefig(str(outp).replace(".pdf", ".png"), dpi=190, bbox_inches="tight")
    print("wrote", outp)
    print("A df", dfA, "valid", repA.passed, "| B df", dfB, "valid", repB.passed)
    print(f"CAPTION: rec={rec_d:.2f} true_opt={opt_d:.2f} "
          f"shortfall={opt_v - max(v[0] for v in valsA):.4f} "
          f"chamber_gap={spread:.4f} density_effect={eff:.4f}")


if __name__ == "__main__":
    main()
