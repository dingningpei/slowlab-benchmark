#!/usr/bin/env python3
"""Improve regret itself by accounting for within-group noise.

Simple regret looks only at the single point the agent finally names, never at
whether its data can separate that point from the others. So pointing correctly
by luck and pointing correctly with evidence score the same.

Verified regret -- the regret an agent can defend:
  1. take the treatments T the agent actually ran, with their observed means
     y-bar_t and replication counts n_t
  2. let t-hat = argmax y-bar_t, the one it will recommend
  3. using the *true* noise sigma -- known to the evaluator, not to the agent --
     find the treatments its own data cannot distinguish from t-hat:
         ȳ_t̂ − ȳ_t  <  z · σ · sqrt(1/n_t̂ + 1/n_t)
  4. verified_regret = f(x*) - min{ f(t) : t indistinguishable from t-hat }

How to read it: **your conclusion only reaches "one of this group", so you are
scored on the worst member of that group.** More replication narrows the
discrimination threshold, shrinks the group, and improves the score. It requires
the agent to say nothing, and rests on no statistical assumption -- sigma is the
environment's true value, not an estimate.
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent
from collections import defaultdict

Z = 1.959964


def episode(task, agent_name, seed):
    env = SlowLabEnv(task, seed=seed)
    ag = make_agent(agent_name)
    x = ag.run(env)
    res = env.submit_recommendation(x, agent_name)
    _, best = env.truth.oracle()

    # The true unit-level noise sd (all three layers combined), known to the evaluator
    rs = env.truth.response_sd
    t = env.task
    sigma = rs * np.sqrt(t.plant_cv ** 2 + t.tau_chamber ** 2 + t.tau_loop ** 2)

    g = defaultdict(list); vec = {}
    for o in env.observations():
        key = (o.design_id, o.treatment)
        g[key].append(o.value)
        d = env._designs[o.design_id]
        vec[key] = [d.treatments[o.treatment][f.name] for f in t.factors]
    if not g:
        return res.simple_regret, res.simple_regret, 0
    keys = list(g)
    mean = np.array([np.mean(g[k]) for k in keys])
    n = np.array([len(g[k]) for k in keys], float)
    X = np.array([vec[k] for k in keys], float)
    truth = env.truth(X)

    hat = int(np.argmax(mean))
    thr = Z * sigma * np.sqrt(1.0 / n[hat] + 1.0 / n)
    tied = (mean[hat] - mean) < thr            # treatments indistinguishable from t-hat
    verified = float(best - truth[tied].min())
    return res.simple_regret, verified, int(tied.sum())


def main(task="T3", seeds=40,
         agents=("random_spread", "classical_doe", "gp_ucb_rep1", "gp_ucb_rep2")):
    t = TASKS[task]
    print(f"{task}, {seeds} sites\n")
    print(f"{'agent':16}{'simple regret':>15}{'verified regret':>17}{'tied-group size':>17}")
    out = {}
    for a in agents:
        S, V, K = [], [], []
        for s in range(seeds):
            sr, vr, k = episode(t, a, s)
            S.append(sr); V.append(vr); K.append(k)
        out[a] = {"simple": float(np.mean(S)), "simple_se": float(np.std(S, ddof=1) / np.sqrt(seeds)),
                  "verified": float(np.mean(V)), "verified_se": float(np.std(V, ddof=1) / np.sqrt(seeds)),
                  "tied": float(np.mean(K))}
        o = out[a]
        print(f"{a:16}{o['simple']:>9.4f}±{o['simple_se']:.4f}"
              f"{o['verified']:>11.4f}±{o['verified_se']:.4f}{o['tied']:>14.1f}")
    (ROOT / "results" / f"verified_{task}.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
