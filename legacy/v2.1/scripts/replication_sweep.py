#!/usr/bin/env python3
"""The central negative result: spending a fixed budget on *replication* never pays
under the outcome metric.

Same algorithm, same budget; the only difference is whether a round's width is cut
into q distinct treatments once each or q/2 treatments twice each. Swept along the
noise and budget axes, looking only at regret.
"""
from __future__ import annotations
import sys, pathlib, json, copy
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent


def run(task, agent, seeds=16):
    r = []
    for s in range(seeds):
        e = SlowLabEnv(task, seed=s)
        r.append(e.submit_recommendation(make_agent(agent).run(e), agent).simple_regret)
    return float(np.mean(r)), float(np.std(r, ddof=1) / np.sqrt(seeds))


def main(seeds=16):
    base = TASKS["T3"]
    out = {"noise": [], "budget": []}
    for cv in [0.06, 0.12, 0.24, 0.40]:
        t = copy.deepcopy(base)
        t.plant_cv = cv; t.tau_chamber = cv * 0.83; t.tau_loop = cv * 0.42
        m1, s1 = run(t, "gp_ucb_rep1", seeds); m2, s2 = run(t, "gp_ucb_rep2", seeds)
        out["noise"].append({"cv": cv, "rep1": m1, "rep1_se": s1, "rep2": m2, "rep2_se": s2})
    for nr in [2, 3, 5, 8]:
        t = copy.deepcopy(base)
        t.plant_cv = 0.24; t.tau_chamber = 0.20; t.tau_loop = 0.10; t.n_rounds = nr
        m1, s1 = run(t, "gp_ucb_rep1", seeds); m2, s2 = run(t, "gp_ucb_rep2", seeds)
        out["budget"].append({"units": nr * t.units_per_round, "rep1": m1, "rep1_se": s1,
                              "rep2": m2, "rep2_se": s2})
    (ROOT / "results" / "replication_sweep.json").write_text(json.dumps(out, indent=2))
    for k in out:
        print(k, [(d.get("cv", d.get("units")), round(d["rep1"], 4), round(d["rep2"], 4)) for d in out[k]])
    return out


if __name__ == "__main__":
    main()
