#!/usr/bin/env python3
"""A design's information content: what percentage of the uncertainty about where
the optimum lies it removes.

    EIG(D) / H[p(x*_theta)]     0 tells you nothing, 1 pins the optimum exactly

Why not BoxingGym's EIRegret form: it anchors the score on the best of K random
designs, so reading a single number first requires running a batch of reference
designs, and the choice of K changes the score (we measured negative values at
K=12). EIG has two computable endpoints of its own, 0 and H[p(x*)], and needs no
reference strategy. Realised regret is the same, anchored on the true optimum.
Once both measurements are self-contained, reference strategies appear only in
the question of whether a task is well-posed, never in the evaluation itself.

Binning dependence: H is computed under the declared grid binning, so the
percentage is a quantity given that grid.
"""
from __future__ import annotations
import sys, pathlib, json, argparse, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent
from slowlab.eig import build_atoms, eig_of_design
sys.path.insert(0, str(ROOT / "scripts"))
from eig_of_llm import atoms_for   # shares the atom set with the model side; M must match

AGENTS = ("random_spread", "classical_doe", "gp_ucb_rep1")
CFGS = ("Sanity", "Screen", "Optimise", "Transfer")


def designs_of(env):
    names = [f.name for f in env.task.factors]
    out = []
    for d in env._designs:
        pts, blk = [], []
        for tid, uids in d.allocation.items():
            v = [d.treatments[tid][n] for n in names]
            for u in uids:
                if u in env.facility._by_id:
                    pts.append(v); blk.append(env.facility.get(u).chamber)
        if pts:
            out.append((np.asarray(pts, float), np.asarray(blk, int)))
    return out


def main(seeds=8, M=80, n_outer=40, nbins=12, out="results/eig.json", cfgs=None):
    """The defaults for M and n_outer must match scripts/eig_of_llm.py. The paper
    puts reference strategies and models under test in the same table, and
    different M on the two sides turns the comparison into one between rulers.
    Changing one requires changing the other and rerunning both sides."""
    fp = ROOT / out
    res = json.loads(fp.read_text()) if fp.exists() else {}
    res = {k: v for k, v in res.items()
           if v.get("M") == M and v.get("n_outer") == n_outer}   # invalidated if the settings changed
    for cfg in (cfgs or CFGS):
        t = TASKS[cfg]; t0 = time.time()
        env0 = SlowLabEnv(t, seed=0); sd = env0.truth.response_sd
        taus = (t.tau_chamber * sd, t.tau_loop * sd, t.tau_batch * sd)
        npu = env0.facility.plants_per_unit(3.25)      # unit-level variance, see eig.build_atoms
        if cfg in res and len(res[cfg].get("agents", {})) == len(AGENTS):
            continue                                    # already computed under the current settings
        atoms = atoms_for(cfg, M, nbins)
        H = atoms.prior_entropy()
        res.setdefault(cfg, {"H_prior": H, "n_cells": atoms.n_cells, "M": M,
                             "n_outer": n_outer, "nbins": nbins, "agents": {}})
        print(f"{cfg}: H={H:.2f} nats over {atoms.n_cells} cells", flush=True)
        for a in AGENTS:
            if a in res[cfg]["agents"]:
                continue
            vals = []
            for s in range(seeds):
                env = SlowLabEnv(t, seed=s); ag = make_agent(a)
                env.submit_recommendation(ag.run(env), a)
                vals += [eig_of_design(atoms, p, b, *taus, n_outer=n_outer,
                                       rng=np.random.default_rng(100 + s))
                         for p, b in designs_of(env)]
            v = np.array(vals, float)
            res[cfg]["agents"][a] = {
                "eig": float(v.mean()), "eig_se": float(v.std(ddof=1) / len(v) ** .5),
                "eig_frac": float(v.mean() / H), "n_designs": len(v)}
            d = res[cfg]["agents"][a]
            print(f"   {a:17s} EIG {d['eig']:.3f}±{d['eig_se']:.3f} nats "
                  f"= {100*d['eig_frac']:.0f}% of H", flush=True)
            fp.write_text(json.dumps(res, indent=1))    # ~170s per sandbox call, so write after each
        print(f"   ({time.time()-t0:.0f}s)", flush=True)
    fp.write_text(json.dumps(res, indent=1))
    done = all(c in res and len(res[c].get("agents", {})) == len(AGENTS)
               for c in (cfgs or CFGS))
    print("ALL DONE" if done else "not finished; run again")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--M", type=int, default=80)
    ap.add_argument("--n_outer", type=int, default=40)
    ap.add_argument("--cfgs", nargs="+", default=None)
    ap.add_argument("--out", default="results/eig.json")
    a = ap.parse_args()
    main(seeds=a.seeds, M=a.M, n_outer=a.n_outer, out=a.out, cfgs=a.cfgs)
