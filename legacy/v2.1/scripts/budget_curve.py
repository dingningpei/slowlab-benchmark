#!/usr/bin/env python3
"""E6: from what budget onwards does experimenting pay for itself?

For each configuration, sweep the number of rounds R from 1 to 10, run the four
reference points, and compare against the zero-experiment literature prior. The prior
does not vary with R, so it is a horizontal line; the R at which the designed policy's
curve crosses it is the critical budget at which experimenting begins to pay.

This curve answers a question about the regime, not about any one agent: under
experiments this slow and this costly, how many seasons of budget are worth spending.
"""
from __future__ import annotations
import sys, pathlib, json, copy
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab import SlowLabEnv, TASKS
from slowlab.tasks import LABEL
from slowlab.registry import make_agent

AGENTS = ("random_spread", "classical_doe", "gp_ucb_rep1")


def episode(task, agent_name, seed):
    env = SlowLabEnv(task, seed=seed)
    ag = make_agent(agent_name)
    r = env.submit_recommendation(ag.run(env), agent_name)
    return r.simple_regret, r.cumulative_regret


def main(seeds=20, rounds=range(1, 11), out="results/budget_curve.json"):
    res = {}
    for key in ("T1", "T2", "T3", "T4"):
        label = LABEL[key]
        res[label] = {a: {"R": [], "regret": [], "se": [], "cum": []} for a in AGENTS}
        for R in rounds:
            t = copy.deepcopy(TASKS[key]); t.n_rounds = R
            for a in AGENTS:
                v = np.array([episode(t, a, s) for s in range(seeds)], float)
                d = res[label][a]
                d["R"].append(R)
                d["regret"].append(float(v[:, 0].mean()))
                d["se"].append(float(v[:, 0].std(ddof=1) / seeds ** .5))
                d["cum"].append(float(v[:, 1].mean()))
        # Critical budget: where the designed policy first falls below the no-experiment ceiling
        from slowlab.fixed_reference import best_fixed
        _, bf = best_fixed(TASKS[key])
        prior = np.full(len(list(rounds)), bf)
        line = []
        for a in ("classical_doe", "gp_ucb_rep1"):
            m = np.array(res[label][a]["regret"])
            below = np.where(m < prior)[0]
            line.append(f"{a}: R*={list(rounds)[below[0]] if len(below) else '>%d'%max(rounds)}")
        res[label]["crossing"] = line
        print(f"{label:10s} best-fixed {prior.mean():.4f}   " + "   ".join(line))
    (ROOT / out).write_text(json.dumps(res, indent=1))
    print("wrote", out)
    return res


if __name__ == "__main__":
    main()
