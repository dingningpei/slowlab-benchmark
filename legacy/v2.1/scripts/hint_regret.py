#!/usr/bin/env python3
"""Task validity: SPADE's hint-based regret, carried over to SlowLab.

    r_D(task) = r̄_A(task | hint) − r̄_A(task)

The hint is privileged information but not the answer: it tells the agent which
quartile of each factor's range the optimum falls in. It shrinks the search space
to 1/4^d without giving the optimum, and does no designing on the agent's behalf.

SPADE's three readings:
  high regret               -> the task sits at the capability frontier (doable
                               with the hint, not without)
  low regret + high return  -> mastered; the task is too easy
  low regret + low return   -> the task is intractable and the measure is void
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent


def hint_box(env, width=0.25):
    """The quartile box the optimum lies in. Returns (lo, hi) per dimension."""
    xstar, _ = env.truth.oracle()
    lo = np.clip(np.floor(np.asarray(xstar) / width) * width, 0, 1 - width)
    return lo, lo + width


def reward(env, x):
    """Normalised return in [0,1]: improvement over the uninformative recommendation of every factor at its midpoint."""
    _, best = env.truth.oracle()
    mid = float(env.truth(np.full((1, env.task.d), 0.5))[0])
    got = float(env.truth(np.asarray(x, float).reshape(1, -1))[0])
    denom = best - mid
    if denom <= 1e-9:
        return 1.0
    return float(np.clip((got - mid) / denom, 0.0, 1.0))


class Hinted:
    """Remap the agent's candidate points into the hint box -- equivalent to knowing which quartile the optimum is in."""
    def __init__(self, inner, lo, hi):
        self.inner, self.lo, self.hi = inner, lo, hi
        self.name = inner.name + "_hint"

    def run(self, env):
        import slowlab.agents as A
        orig = A._cands
        lo, hi = self.lo, self.hi
        A._cands = lambda d, n, rng: lo + (hi - lo) * rng.random((n, d))
        try:
            x = self.inner.run(env)
        finally:
            A._cands = orig
        return np.clip(np.asarray(x, float), lo, hi)


def main(seeds=20, agents=("classical_doe", "gp_ucb_rep1", "gp_ucb_rep2")):
    out = {}
    for tname in ["T1", "T2", "T3"]:
        rows = {}
        for a in agents:
            rn, rh = [], []
            for s in range(seeds):
                e = SlowLabEnv(TASKS[tname], seed=s)
                x = make_agent(a).run(e); rn.append(reward(e, x))
                e2 = SlowLabEnv(TASKS[tname], seed=s)
                lo, hi = hint_box(e2)
                x2 = Hinted(make_agent(a), lo, hi).run(e2); rh.append(reward(e2, x2))
            rows[a] = (float(np.mean(rn)), float(np.mean(rh)),
                       float(np.mean(rh) - np.mean(rn)))
        out[tname] = rows
    print(f"{'task':5}{'agent':16}{'r_A (no hint)':>16}{'r_A (hint)':>14}{'hint regret':>14}   reading")
    for t, rows in out.items():
        for a, (n, h, g) in rows.items():
            # Faithful to SPADE: intractability is judged by the return staying low
            # *even with the hint*, not by a small gap. A small gap with a high
            # return means mastered.
            if h < 0.45:
                verdict = "intractable (return stays low even with the hint)"
            elif g > 0.15:
                verdict = "frontier (only doable with the hint)"
            elif h > 0.85:
                verdict = "mastered (doable without the hint)"
            else:
                verdict = "near-mastered"
            print(f"{t:5}{a:16}{n:>16.3f}{h:>14.3f}{g:>+14.3f}   {verdict}")
    (ROOT / "results" / "hint_regret.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
