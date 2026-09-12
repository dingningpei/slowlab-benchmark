#!/usr/bin/env python3
"""The third axis of irreversibility: the recovery period after heat damage.

recovery_days is a hyperparameter, in days. A heat-damaged unit enters recovery at
the end of the current round; the length of that recovery is graded by how far the
unit exceeded TCRIT+margin, capped at recovery_days. A cycle is 120 days, so:
  0    off
  <120 blocks only the next round
  >120 blocks two rounds in a row
The lost unit-days *cannot be bought back with money* -- this is what separates this
axis from the Costly one.
"""
from __future__ import annotations
import sys, pathlib, json, copy
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent

AGENTS = ["random_spread", "classical_doe", "gp_ucb_rep1", "gp_ucb_rep2"]


def run(task, a, seeds=20):
    R, L, C, J = [], [], [], []
    for s in range(seeds):
        e = SlowLabEnv(task, seed=s)
        ag = make_agent(a)
        x = ag.run(e)
        r = e.submit_recommendation(x, a)
        R.append(r.simple_regret); L.append(r.lost_unit_days)
        C.append(sum(o.value * task.duration_days for o in e.observations()))
        J.append(len(r.rejections))
    return (np.mean(R), np.std(R, ddof=1) / np.sqrt(seeds),
            np.mean(L), np.mean(C), np.mean(J))


def main(seeds=20, sweep=(0, 30, 60, 120, 240)):
    out = {}
    for rd in sweep:
        t = copy.deepcopy(TASKS["T3"]); t.recovery_days = float(rd)
        print(f"\n=== recovery_days = {rd} days (cycle {t.duration_days} days) ===")
        print(f"{'agent':16}{'regret':>16}{'lost unit-days':>16}{'net cash':>11}{'rejections':>12}")
        out[rd] = {}
        for a in AGENTS:
            m, se, lost, cash, rej = run(t, a, seeds)
            out[rd][a] = {"regret": m, "regret_se": se, "lost_unit_days": lost,
                          "cash": cash, "rejections": rej}
            print(f"{a:16}{m:>10.4f}±{se:.4f}{lost:>13.0f}{cash:>11.1f}{rej:>10.1f}")
    (ROOT / "results" / "recovery_sweep.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
