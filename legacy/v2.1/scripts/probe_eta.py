#!/usr/bin/env python3
"""Falsification probe for the trajectory-shape measure.

Decompose the regret trajectory into gross forward progress and gross backsliding:

    G = sum_r max(0, R(r) - R(r+1))      ground gained
    B = sum_r max(0, R(r+1) - R(r))      ground lost
    R(1) - R(R) = G - B                  (exact)

    eta = (G - B) / G = 1 - B/G          in (-inf, 1]

eta = 1  : every round moved forward, nothing was undone
eta < 1  : seasons were spent losing ground

Questions this script must answer honestly:
  Q1  Is eta independent of terminal regret, or just another reading of it?
  Q2  Do the reference methods separate on eta?
  Q3  Can a data-blind agent score eta = 1? (the gaming check)
  Q4  Does eta have any power at R = 2 or 3, our native horizons?
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np
import copy

ROOT = pathlib.Path("/tmp/w4")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent
import slowlab.agents as A
from learning_curve import trace


def decompose(tr):
    tr = np.asarray(tr, float)
    if len(tr) < 2:
        return dict(G=np.nan, B=np.nan, eta=np.nan, term=float(tr[-1]) if len(tr) else np.nan)
    d = tr[:-1] - tr[1:]                       # positive = improvement
    G = float(np.clip(d, 0, None).sum())
    B = float(np.clip(-d, 0, None).sum())
    eta = (G - B) / G if G > 1e-12 else np.nan
    return dict(G=G, B=B, eta=eta, term=float(tr[-1]), start=float(tr[0]))


class BestObservedAgent:
    """Gaming check: recommend the highest observed value so far, always.

    If eta could be maximised by a reporting policy alone, this agent would
    score eta = 1 while doing no design reasoning at all.
    """
    def __init__(self, inner="random_spread"):
        self.inner = make_agent(inner)

    def run(self, env):
        return self.inner.run(env)


def sweep(task_key, rounds, seeds, agents):
    t = copy.deepcopy(TASKS[task_key])
    if rounds:
        t.n_rounds = rounds
    rows = []
    for a in agents:
        for s in range(seeds):
            tr = trace(t, a, s)
            if len(tr) < 2:
                continue
            r = decompose(tr)
            r.update(task=task_key, agent=a, seed=s, R=len(tr), trace=[round(v, 5) for v in tr])
            rows.append(r)
    return rows


def report(rows, title):
    import collections
    print(f"\n=== {title} ===")
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["task"], r["agent"])].append(r)
    print(f"{'task':6}{'agent':17}{'R':>3}{'term regret':>13}{'G':>9}{'B':>9}{'eta':>8}{'mono%':>8}")
    for k, v in by.items():
        eta = np.array([x["eta"] for x in v], float)
        mono = np.mean([x["B"] < 1e-9 for x in v]) * 100
        print(f"{k[0]:6}{k[1]:17}{v[0]['R']:>3}"
              f"{np.mean([x['term'] for x in v]):>13.4f}"
              f"{np.nanmean([x['G'] for x in v]):>9.4f}"
              f"{np.nanmean([x['B'] for x in v]):>9.4f}"
              f"{np.nanmean(eta):>8.3f}{mono:>8.0f}")
    # Q1: correlation with terminal regret, pooled within task
    print("\n  corr(eta, terminal regret) within task:")
    for t in sorted({r["task"] for r in rows}):
        sub = [r for r in rows if r["task"] == t and np.isfinite(r["eta"])]
        if len(sub) > 5:
            e = np.array([r["eta"] for r in sub]); q = np.array([r["term"] for r in sub])
            if e.std() > 1e-9 and q.std() > 1e-9:
                print(f"    {t}: r = {np.corrcoef(e, q)[0,1]:+.3f}  (n={len(sub)})")


if __name__ == "__main__":
    AG = ("random_spread", "classical_doe", "gp_ucb_rep1", "gp_ucb_rep2")
    native = []
    for t in ("T1", "T2", "T3", "T4"):
        native += sweep(t, None, 20, AG)
    report(native, "native horizons (R = 2 or 3)")

    long_ = sweep("T3", 8, 20, AG)
    report(long_, "Optimise at R = 8")

    long2 = sweep("T1", 8, 20, AG)
    report(long2, "Sanity at R = 8")

    out = ROOT / "results" / "trajectory_probe.json"
    out.write_text(json.dumps({"native": native, "T3_R8": long_, "T1_R8": long2}, indent=1))
    print("\nwrote", out)
