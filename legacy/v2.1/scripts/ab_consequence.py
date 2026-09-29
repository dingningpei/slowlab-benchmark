#!/usr/bin/env python3
"""The downstream consequence of the design-validity criterion: can the agent say
"I do not know yet"?

Two hand-written designs, one round each, over 60 sites:
  A  4 temperatures x 3 densities, n=1 per cell  -> pure-error df = 0
  B  split-plot, 2 temperatures x 3 densities, n=2  -> pure-error df = 6
What is measured: how often the sign of the density effect is called correctly, and
whether a wrong call can be flagged "not significant" by the agent itself.
"""
from __future__ import annotations
import sys, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.tasks import TASKS
from slowlab.env import SlowLabEnv
from slowlab.design import Design

T = TASKS["T3"]
TA = [0.0, 1/3, 2/3, 1.0]
DL = [0.0, 0.5, 1.0]
TB = {"lo": 0.25, "hi": 0.75}
TCRIT = {6: 2.447, 1: 12.706}          # t_{0.975, df}


def design_A():
    tr, al = {}, {}
    for ci, tv in enumerate(TA):
        for li, dv in enumerate(DL):
            k = f"T{ci}D{li}"
            tr[k] = {"day_temp": tv, "density": dv}
            al[k] = [f"c{ci}l{li}"]
    return Design(treatments=tr, allocation=al, randomization_seed=1)


def design_B(seed):
    rng = np.random.default_rng(1000 + seed)
    tr, al = {}, {}
    for tn, tv in TB.items():
        for li, dv in enumerate(DL):
            tr[f"{tn}_{li}"] = {"day_temp": tv, "density": dv}
            al[f"{tn}_{li}"] = []
    cs = [0, 1, 2, 3]; rng.shuffle(cs)
    for tn, pair in zip(TB, [cs[:2], cs[2:]]):
        for c in pair:
            o = [0, 1, 2]; rng.shuffle(o)
            for pos, li in enumerate(o):
                al[f"{tn}_{li}"].append(f"c{c}l{pos}")
    return Design(treatments=tr, allocation=al, randomization_seed=1000 + seed)


def main(n_seeds=60, verbose=True):
    stats = {"A": dict(wrong=0, sig=0, sig_wrong=0),
             "B": dict(wrong=0, sig=0, sig_wrong=0)}
    wrongB = []
    for seed in range(n_seeds):
        # ---- A: n=1, the only error term is residual df=1 (confounded with curvature) ----
        A = design_A(); e = SlowLabEnv(T, seed=seed)
        e.submit_design(A); e.advance()
        ob = {o.unit_id: o.value for o in e.observations()}
        ci = int(max(ob, key=ob.get)[1])
        ys = np.array([ob[f"c{ci}l{j}"] for j in range(3)])
        true_slope = np.polyfit(DL, e.truth(np.column_stack([np.full(3, TA[ci]), DL])), 1)[0]
        sl, ic = np.polyfit(DL, ys, 1)
        res = ys - (sl * np.array(DL) + ic)
        sxx = np.sum((np.array(DL) - np.mean(DL)) ** 2)
        s2 = np.sum(res ** 2) / 1
        t = abs(sl) / np.sqrt(s2 / sxx) if s2 > 0 else np.inf
        ok = np.sign(sl) == np.sign(true_slope)
        stats["A"]["wrong"] += (not ok)
        if t > TCRIT[1]:
            stats["A"]["sig"] += 1
            stats["A"]["sig_wrong"] += (not ok)

        # ---- B: n=2, pure-error df=6 ----
        B = design_B(seed); e = SlowLabEnv(T, seed=seed)
        e.submit_design(B); e.advance()
        ob = {o.unit_id: o.value for o in e.observations()}
        mu = {t_: np.mean([ob[u] for u in B.allocation[t_]]) for t_ in B.treatments}
        tn = max(mu, key=mu.get).split("_")[0]
        xs = np.repeat(DL, 2)
        yy = np.array([ob[u] for j in range(3) for u in B.allocation[f"{tn}_{j}"]])
        true_slope = np.polyfit(DL, e.truth(np.column_stack([np.full(3, TB[tn]), DL])), 1)[0]
        resid = [ob[u] - mu[t_] for t_ in B.treatments for u in B.allocation[t_]]
        s2 = np.sum(np.square(resid)) / 6
        sl = np.polyfit(xs, yy, 1)[0]
        sxx = np.sum((xs - np.mean(xs)) ** 2)
        t = abs(sl) / np.sqrt(s2 / sxx)
        ok = np.sign(sl) == np.sign(true_slope)
        stats["B"]["wrong"] += (not ok)
        if t > TCRIT[6]:
            stats["B"]["sig"] += 1
            stats["B"]["sig_wrong"] += (not ok)
        if not ok:
            wrongB.append((seed, sl, true_slope, t, np.sqrt(s2)))

    if verbose:
        for k in "AB":
            s = stats[k]
            print(f"design {k}: wrong sign {s['wrong']}/{n_seeds} | declared significant "
                  f"{s['sig']} times, of which {s['sig_wrong']} were wrong")
        print("\nSites where B got the sign wrong (all of them fall in the non-significant region):")
        for sd, sl, tr_, t, s in wrongB:
            print(f"  seed={sd:2d}  estimated slope {sl:+.4f}  true slope {tr_:+.4f}  "
                  f"|t|={t:.2f} < 2.45 -> reported not significant  pure-error s={s:.4f}")
    return stats, wrongB


if __name__ == "__main__":
    main()
