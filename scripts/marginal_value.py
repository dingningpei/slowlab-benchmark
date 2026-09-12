#!/usr/bin/env python3
"""What one more round of experiment is worth to an ideal reasoner.

This tests a premise of section 2: slow feedback binds only if many rounds are needed.
If the first round already captures most of the gain, then slowness does not press on a
good agent at all, only on one that learns slowly.

About 170s per sandbox call with no background survival, so results are written per
task; run repeatedly until ALL DONE.
"""
from __future__ import annotations
import sys, json, pathlib, argparse
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from slowlab import SlowLabEnv, TASKS
from slowlab.achievable import (achievable_regret, posterior_weights, _sigma,
                                candidate_set, response_table)
from eig_of_llm import atoms_for

CFGS = ("Sanity", "Optimise", "Screen", "Transfer")


def main(M=320, n_outer=150, n_rep=4, out="results/marginal.json"):
    fp = ROOT / out
    res = json.loads(fp.read_text()) if fp.exists() else {}
    for cfg in CFGS:
        if cfg in res:
            continue
        t = TASKS[cfg]
        env = SlowLabEnv(t, seed=0); sd = env.truth.response_sd
        taus = (t.tau_chamber * sd, t.tau_loop * sd, t.tau_batch * sd)
        n = min(t.units_per_round, len(env.facility.units)); d = len(t.factors)
        blk = np.array([env.facility.units[i].chamber for i in range(n)])
        a = atoms_for(cfg, M, 12)
        cand = candidate_set(a, rng=np.random.default_rng(0))
        kw = dict(F=response_table(a, cand), cand=cand, n_outer=n_outer)
        z = np.zeros((0, d)); zb = np.zeros(0, int)
        # Repeat several times (different truth atoms and noise draws) and average;
        # a single draw is far too noisy
        runs = []
        for rep in range(n_rep):
            rng = np.random.default_rng(11 + rep)
            truth = int(rng.integers(a.M))
            row = [(0, achievable_regret(a, z, zb, *taus, **kw), float(a.M))]
            w = None
            for r in range(t.n_rounds):
                D = rng.random((n, d))
                mu = a.mean_response(D)
                S = _sigma(a, D, blk, *taus)
                y = mu[truth] + np.linalg.cholesky(S + 1e-14 * np.eye(n)) @ \
                    rng.standard_normal(n)
                w = posterior_weights(a, D, blk, y, *taus, prior_w=w)
                row.append((r + 1,
                            achievable_regret(a, z, zb, *taus, prior_w=w, **kw),
                            float(1 / np.sum(w ** 2))))
            runs.append(row)
        res[cfg] = {"M": M, "n_rep": n_rep,
                    "rounds": [[r,
                                float(np.mean([x[i][1] for x in runs])),
                                float(np.mean([x[i][2] for x in runs]))]
                               for i, (r, _, _) in enumerate(runs[0])]}
        fp.write_text(json.dumps(res, indent=1))
        print(f"{cfg} done", flush=True)
        break
    else:
        print("ALL DONE")
        for c in CFGS:
            rows = res[c]["rounds"]; tot = rows[0][1] - rows[-1][1]
            print(f"\n{c}  (M={res[c]['M']})")
            for i, (r, v, e) in enumerate(rows):
                m = rows[i-1][1] - v if i else None
                s = f"{m:+.5f}  {100*m/tot:>3.0f}% of total" if i else ""
                print(f"   round {r}: R*={v:.5f}  ESS={e:6.1f}   {s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--M", type=int, default=320)
    main(M=ap.parse_args().M)
