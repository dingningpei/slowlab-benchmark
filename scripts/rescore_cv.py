#!/usr/bin/env python3
"""Re-score R*(D) under the cross-validated estimator, for both the reference
strategies and the LLM transcripts.

Why re-score: `achievable_regret` is in-sample -- one set of atoms both chooses
and evaluates the recommendation -- ESS is about 1, and c changes sign when
evaluated held out. See the note at the top of slowlab/achievable.py.
Everything here goes through `achievable_regret_cv`: tempering, leave-one-out
calibration of T, and held-out evaluation.

**The environment did not change, only the evaluator did.** An agent never sees
the evaluator, so every transcript remains valid: no LLM needs rerunning, only
rescoring.

M=320 is split in half: 160 for selection, 160 for evaluation.

On speed: `ManagedTomgro.__call__` is a 210-day loop vectorised over candidate
points, so its cost tracks the *number of calls* rather than the number of
points. Calling mean_response per design costs 1.7 s per design; batching every
distinct treatment of a whole task into one call costs 1.7 s for the lot. Do not
change this back.

    python scripts/rescore_cv.py --only Sanity
"""
from __future__ import annotations
import sys, json, glob, pathlib, re, argparse, pickle, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import slowlab
from slowlab import SlowLabEnv, TASKS
from slowlab.eig import build_atoms
from slowlab.achievable import (achievable_regret_cv, split_atoms,
                                candidate_set, response_table)
from slowlab.registry import make_agent
from run_eig import designs_of
from eig_of_llm import designs_from

CFGS = ("Sanity", "Screen", "Optimise", "Transfer")
REFS = ("random_spread", "classical_doe", "gp_ucb_rep1", "gp_ucb_rep2")
LLM_DIR = f"llm_env{slowlab.ENV_VERSION}"
M = 320


def atoms_at(cfg, M=M, nbins=12):
    fp = ROOT / "results" / f"_atoms_{cfg}_M{M}_b{nbins}_env{slowlab.ENV_VERSION}.pkl"
    if fp.exists():
        return pickle.loads(fp.read_bytes())
    a = build_atoms(TASKS[cfg], M=M, nbins=nbins, seed0=10_000)
    fp.write_bytes(pickle.dumps(a))
    return a


def collect(cfg, seeds):
    """All designs to be scored for one task: [(key, pts, blocks), ...].

    Reference strategies are run once per seed; LLM transcripts are parsed
    directly, with no model rerun.
    """
    t = TASKS[cfg]
    out = []
    srcs = sorted(glob.glob(str(ROOT / "results" / LLM_DIR /
                                f"transcript_*_{cfg}_s*.json")))
    if not srcs:
        # A hard-coded old pattern once matched zero files silently (the old
        # directory name contained "@") -- the same class of bug as classical_doe's
        # screening round being silently rejected. Raise when nothing matches.
        raise SystemExit(f"{cfg}: no transcript matched under results/{LLM_DIR}/")
    for f in srcs:
        name = pathlib.Path(f).name
        seed = int(re.search(r"_s(\d+)\.json$", f).group(1))
        env = SlowLabEnv(t, seed=seed)
        for j, (p, b) in enumerate(designs_from(f, env)):
            out.append((f"{name}#{j}", np.asarray(p, float), np.asarray(b, int)))
    for ag in REFS:
        for s in range(seeds):
            env = SlowLabEnv(t, seed=s)
            agent = make_agent(ag)
            env.submit_recommendation(agent.run(env), ag)
            for j, (p, b) in enumerate(designs_of(env)):
                out.append((f"__ref__{ag}#s{s}.{j}",
                            np.asarray(p, float), np.asarray(b, int)))
    return out


def main(seeds=8, n_outer=300, only=None, out=None):
    out = out or f"results/achievable_cv_env{slowlab.ENV_VERSION}_M{M}.json"
    fp = ROOT / out
    res = json.loads(fp.read_text()) if fp.exists() else {}

    for cfg in (only or CFGS):
        t = TASKS[cfg]
        env0 = SlowLabEnv(t, seed=0); sd = env0.truth.response_sd
        taus = (t.tau_chamber * sd, t.tau_loop * sd, t.tau_batch * sd)
        slot = res.setdefault(cfg, {"M": M, "n_outer": n_outer,
                                    "estimator": "cv", "prior": None,
                                    "raw": {}, "T": {}})
        items = collect(cfg, seeds)
        todo = [it for it in items if it[0] not in slot["raw"]]
        if not todo and slot["prior"] is not None:
            continue

        a = atoms_at(cfg)
        ain, aout = split_atoms(a)
        cand = candidate_set(ain, rng=np.random.default_rng(0))
        kw = dict(cand=cand, F_in=response_table(ain, cand),
                  F_out=response_table(aout, cand),
                  best_in=np.array([w.oracle()[1] for w in ain.worlds], float),
                  best_out=np.array([w.oracle()[1] for w in aout.worlds], float),
                  n_outer=n_outer)

        # Key point: every design's distinct treatments go into one forward call
        t0 = time.time()
        allpts = np.unique(np.vstack([p for _, p, _ in todo]), axis=0)
        MU_IN = ain.mean_response(allpts)
        MU_OUT = aout.mean_response(allpts)
        key_of = {tuple(np.round(r, 12)): i for i, r in enumerate(allpts)}
        print(f"{cfg}: {len(todo)} designs, {len(allpts)} distinct treatments, "
              f"forward pass {time.time()-t0:.1f}s", flush=True)

        if slot["prior"] is None:
            slot["prior"] = achievable_regret_cv(
                ain, aout, np.zeros((0, len(t.factors))), np.zeros(0, int),
                *taus, **kw)

        t0 = time.time()
        for i, (key, pts, blk) in enumerate(todo):
            idx = [key_of[tuple(np.round(r, 12))] for r in pts]
            v, T = achievable_regret_cv(
                ain, aout, pts, blk, *taus, return_T=True,
                mu_in=MU_IN[:, idx], mu_out=MU_OUT[:, idx],
                rng=np.random.default_rng(100 + i), **kw)
            slot["raw"][key] = float(v); slot["T"][key] = T
            if i % 100 == 99:
                fp.write_text(json.dumps(res, indent=1))
                print(f"  {i+1}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)
        fp.write_text(json.dumps(res, indent=1))

    # Aggregate: key prefix -> agent
    for cfg, slot in res.items():
        agg, aggT = {}, {}
        for name, v in slot["raw"].items():
            if name.startswith("__ref__"):
                m = "ref:" + name[len("__ref__"):].split("#")[0]
            else:
                m = re.match(r"transcript_(.+)_[A-Za-z]+_s\d+\.json#\d+$",
                             name).group(1)
            agg.setdefault(m, []).append(v)
            aggT.setdefault(m, []).append(slot["T"].get(name))
        slot["agents"] = {
            m: {"R_star": float(np.mean(v)), "n": len(v),
                "T_median": float(np.median([x for x in aggT[m] if x]))}
            for m, v in agg.items() if v}
    fp.write_text(json.dumps(res, indent=1))

    for cfg in CFGS:
        if cfg not in res or res[cfg].get("prior") is None:
            continue
        s = res[cfg]
        print(f"\n{cfg:9s} R*(∅)={s['prior']:.5f}", flush=True)
        for m, d in sorted(s.get("agents", {}).items(),
                           key=lambda kv: kv[1]["R_star"]):
            print(f"    {m:34s} R*={d['R_star']:.5f}  "
                  f"c={100*(1-d['R_star']/s['prior']):5.0f}%  "
                  f"T~{d['T_median']:.0f}  n={d['n']}", flush=True)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n_outer", type=int, default=300)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    main(seeds=a.seeds, n_outer=a.n_outer, only=a.only)
