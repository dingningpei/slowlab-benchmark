#!/usr/bin/env python3
"""What is examined here: an agent improving its own design by iteration, driving
regret down round by round.

A single end-of-episode regret cannot separate "got lucky in round 1" from "learned
over the rounds". Here the agent is asked "what would you recommend now" at the end
of *every* round, and that recommendation is scored against the truth, giving a
trajectory:

    regret(1) → regret(2) → ... → regret(R)

Three quantities to read off:
  - the starting regret(1)   -- prior knowledge and luck
  - the final regret(R)      -- the end product
  - the *per-round improvement* -- this one is the ability to improve one's own experiment
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent
import slowlab.agents as A
import copy


def recommend_now(agent, env):
    """Ask the agent what it would recommend given the data so far. Does not change
    its behaviour."""
    X, y = env.as_arrays()
    if not len(y):
        return np.full(env.task.d, 0.5)
    if isinstance(agent, A.GPUCBAgent):
        g = A.GP(noise=max(0.15, env.task.plant_cv)).fit(X, y)
        C = A._cands(env.task.d, 4000, np.random.default_rng(7))
        mu, _ = g.predict(C)
        return C[int(np.argmax(mu))]
    return X[int(np.argmax(y))]


def trace(task, agent_name, seed):
    """Run a whole episode, recording the true regret at the end of each round."""
    env = SlowLabEnv(task, seed=seed)
    ag = make_agent(agent_name)
    _, best = env.truth.oracle()
    out = []
    orig_advance = env.advance

    def hooked():
        obs = orig_advance()
        x = recommend_now(ag, env)
        out.append(best - float(env.truth(np.asarray(x, float).reshape(1, -1))[0]))
        return obs

    env.advance = hooked
    ag.run(env)
    return out


def main(task="T3", rounds=8, seeds=20,
         agents=("random_spread", "classical_doe", "gp_ucb_rep1", "gp_ucb_rep2")):
    t = copy.deepcopy(TASKS[task]); t.n_rounds = rounds
    res = {}
    for a in agents:
        M = []
        for s in range(seeds):
            tr = trace(t, a, s)
            tr = (tr + [tr[-1]] * rounds)[:rounds] if tr else [np.nan] * rounds
            M.append(tr)
        M = np.array(M, float)
        res[a] = {"mean": np.nanmean(M, 0).tolist(),
                  "se": (np.nanstd(M, 0, ddof=1) / np.sqrt(len(M))).tolist()}
    print(f"{task}, {rounds} rounds x {t.units_per_round} units, {seeds} sites\n")
    hdr = "".join(f"{'R'+str(i+1):>9}" for i in range(rounds))
    print(f"{'agent':16}{hdr}{'total drop':>12}{'2nd-half drop':>15}")
    for a in agents:
        m = np.array(res[a]["mean"])
        row = "".join(f"{v:>9.4f}" for v in m)
        half = rounds // 2
        print(f"{a:16}{row}{m[0]-m[-1]:>10.4f}{m[half-1]-m[-1]:>12.4f}")
    (ROOT / "results" / f"curve_{task}.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
