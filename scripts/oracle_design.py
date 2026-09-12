#!/usr/bin/env python3
"""How much of R*(none) can *any* admissible design remove at this budget?

The results section reports that no agent, scripted or otherwise, gets design
efficiency c above 20%. That invites the obvious objection: if no design in the
admissible set can push R*(D) far below R*(none), then a small c measures the task
rather than the agent, and the denominator of eq:capture carries no information.

This script answers it by searching the admissible set directly. It knows P_Theta and
the forward model, so it is not an agent and its designs are not achievable by one --
it is an upper bound on what the design half of the score could ever report. Nothing
here calls a language model.

The search is over structured families rather than a free optimisation, because the
admissible set is discrete and hierarchical: chamber-level factors take one value per
chamber, so a design is a choice of (how many chambers, how many loops each, how many
distinct chamber-level settings, how many distinct loop-level settings, and where those
settings sit in the factor space).

    python scripts/oracle_design.py --only Sanity
    python scripts/oracle_design.py                 # all four tasks
"""
from __future__ import annotations
import sys, json, pathlib, time, argparse, itertools
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import slowlab
from slowlab import SlowLabEnv, TASKS, Design
from slowlab.achievable import (achievable_regret_cv, split_atoms,
                                candidate_set, response_table)
from rescore_cv import atoms_at

CFGS = ("Sanity", "Screen", "Optimise", "Transfer")


def layouts(B, n_chambers, loops_per_chamber):
    """(g chambers, l loops each) that exactly fill a round's budget."""
    return [(g, B // g) for g in range(1, n_chambers + 1)
            if B % g == 0 and 1 <= B // g <= loops_per_chamber]


def level_sets(dim, k, rng, family):
    """k distinct points in [0,1]^dim under one of the standard design families."""
    if k == 1:
        return np.full((1, dim), 0.5)
    if family == "corners":                       # two-level, pulled off the rails
        rows = np.array(list(itertools.product([0.1, 0.9], repeat=dim)))
        return rows[rng.choice(len(rows), size=min(k, len(rows)), replace=False)]
    if family == "grid3":                         # three-level, for curvature
        rows = np.array(list(itertools.product([0.1, 0.5, 0.9], repeat=dim)))
        return rows[rng.choice(len(rows), size=min(k, len(rows)), replace=False)]
    if family == "lhs":                           # space-filling
        P = np.empty((k, dim))
        for j in range(dim):
            P[:, j] = (rng.permutation(k) + 0.5) / k
        return P
    if family == "maximin":                       # best of 40 LHS draws by min distance
        best, bd = None, -1.0
        for _ in range(40):
            P = np.empty((k, dim))
            for j in range(dim):
                P[:, j] = (rng.permutation(k) + 0.5) / k
            d = np.inf if k < 2 else min(
                np.linalg.norm(P[a] - P[b]) for a in range(k) for b in range(a + 1, k))
            if d > bd:
                best, bd = P, d
        return best
    return rng.random((k, dim))                   # "random"


def build(task, facility, g, l, kc, kl, fam_c, fam_l, rng):
    """One admissible round: pts per unit plus the chamber index of each unit."""
    ci = [j for j, f in enumerate(task.factors) if f.control_level == "chamber"]
    li = [j for j, f in enumerate(task.factors) if f.control_level == "loop"]
    A = level_sets(len(ci), kc, rng, fam_c) if ci else np.zeros((1, 0))
    Bv = level_sets(len(li), kl, rng, fam_l) if li else np.zeros((1, 0))
    pts, blk = [], []
    for gi in range(g):
        a = A[gi % len(A)]
        for li_ in range(l):
            x = np.empty(len(task.factors))
            for s, j in enumerate(ci):
                x[j] = a[s]
            for s, j in enumerate(li):
                x[j] = Bv[li_ % len(Bv)][s]
            pts.append(x); blk.append(gi)
    return np.array(pts), np.array(blk, int)


def admissible(task, facility, pts, blk):
    """Check the round through the environment's own validator, not our reading of it."""
    env = SlowLabEnv(task, seed=0)
    names = [f.name for f in task.factors]
    tre, allo = {}, {}
    seen = {}
    loop_of = {}
    for i, (x, b) in enumerate(zip(pts, blk)):
        key = tuple(np.round(x, 9))
        tid = seen.setdefault(key, f"T{len(seen)}")
        tre[tid] = {n: float(v) for n, v in zip(names, x)}
        loop_of[b] = loop_of.get(b, -1) + 1
        allo.setdefault(tid, []).append(f"c{b}l{loop_of[b]}")
    # Rejection.__bool__ is falsy when the design is rejected, so the object itself is
    # the answer. Testing `.code` instead silently rejects everything: an accepted
    # design carries the truthy string "OK".
    return bool(env.validate_design(Design(treatments=tre, allocation=allo,
                                           randomization_seed=1)))


def pool(cfg, n_random=60, seed=0):
    """Every structured design for this task, plus a random tail."""
    t = TASKS[cfg]
    fac = SlowLabEnv(t, seed=0).facility
    rng = np.random.default_rng(seed)
    ci = [f for f in t.factors if f.control_level == "chamber"]
    li = [f for f in t.factors if f.control_level == "loop"]
    out, seen = [], set()
    fams = ("maximin", "lhs", "corners", "grid3", "random")
    for g, l in layouts(t.units_per_round, fac.n_chambers, fac.loops_per_chamber):
        for kc in range(1, g + 1):
            for kl in range(1, l + 1):
                if not ci and kc > 1: continue
                if not li and kl > 1: continue
                for fc in (fams if ci else ("maximin",)):
                    for fl in (fams if li else ("maximin",)):
                        pts, blk = build(t, fac, g, l, kc, kl, fc, fl, rng)
                        if not admissible(t, fac, pts, blk): continue
                        k = len(np.unique(np.round(pts, 9), axis=0))
                        key = (g, l, kc, kl, fc, fl)
                        if key in seen: continue
                        seen.add(key)
                        out.append({"g": g, "l": l, "kc": kc, "kl": kl,
                                    "fam_c": fc, "fam_l": fl, "k": int(k),
                                    "pts": pts, "blk": blk})
    for _ in range(n_random):
        g, l = layouts(t.units_per_round, fac.n_chambers, fac.loops_per_chamber)[
            rng.integers(len(layouts(t.units_per_round, fac.n_chambers,
                                     fac.loops_per_chamber)))]
        kc = int(rng.integers(1, g + 1)); kl = int(rng.integers(1, l + 1))
        pts, blk = build(t, fac, g, l, kc, kl, "random", "random", rng)
        if not admissible(t, fac, pts, blk): continue
        out.append({"g": g, "l": l, "kc": kc, "kl": kl, "fam_c": "random",
                    "fam_l": "random", "k": int(len(np.unique(np.round(pts, 9), axis=0))),
                    "pts": pts, "blk": blk})
    return out


def main(only=None, n_outer=300, out=None, n_random=60):
    out = out or f"results/oracle_design_env{slowlab.ENV_VERSION}.json"
    fp = ROOT / out
    res = json.loads(fp.read_text()) if fp.exists() else {}

    for cfg in (only or CFGS):
        t = TASKS[cfg]
        env0 = SlowLabEnv(t, seed=0); sd = env0.truth.response_sd
        taus = (t.tau_chamber * sd, t.tau_loop * sd, t.tau_batch * sd)

        a = atoms_at(cfg)
        ain, aout = split_atoms(a)
        cand = candidate_set(ain, rng=np.random.default_rng(0))
        kw = dict(cand=cand, F_in=response_table(ain, cand),
                  F_out=response_table(aout, cand),
                  best_in=np.array([w.oracle()[1] for w in ain.worlds], float),
                  best_out=np.array([w.oracle()[1] for w in aout.worlds], float),
                  n_outer=n_outer)

        P = pool(cfg, n_random=n_random)
        t0 = time.time()
        allpts = np.unique(np.vstack([d["pts"] for d in P]), axis=0)
        MU_IN = ain.mean_response(allpts); MU_OUT = aout.mean_response(allpts)
        key_of = {tuple(np.round(r, 12)): i for i, r in enumerate(allpts)}
        print(f"{cfg}: {len(P)} candidate designs, {len(allpts)} distinct treatments, "
              f"forward pass {time.time()-t0:.1f}s", flush=True)

        prior = achievable_regret_cv(ain, aout, np.zeros((0, len(t.factors))),
                                     np.zeros(0, int), *taus, **kw)
        rows, t0 = [], time.time()
        for i, d in enumerate(P):
            idx = [key_of[tuple(np.round(r, 12))] for r in d["pts"]]
            v = achievable_regret_cv(ain, aout, d["pts"], d["blk"], *taus,
                                     mu_in=MU_IN[:, idx], mu_out=MU_OUT[:, idx],
                                     rng=np.random.default_rng(100 + i), **kw)
            rows.append({kk: d[kk] for kk in ("g", "l", "kc", "kl", "fam_c", "fam_l", "k")}
                        | {"R_star": float(v), "c": float(1 - v / prior)})
            if i % 25 == 24:
                print(f"  {i+1}/{len(P)}  {time.time()-t0:.0f}s", flush=True)
        rows.sort(key=lambda r: -r["c"])
        res[cfg] = {"prior": float(prior), "n_designs": len(rows), "designs": rows}
        fp.write_text(json.dumps(res, indent=1))

        b = rows[0]
        print(f"  best c = {100*b['c']:.0f}%  at k={b['k']} "
              f"({b['g']}x{b['l']}, kc={b['kc']} {b['fam_c']}, kl={b['kl']} {b['fam_l']})",
              flush=True)

    print("\n" + "=" * 60)
    for cfg in CFGS:
        if cfg not in res: continue
        r = res[cfg]["designs"]
        best = r[0]
        print(f"{cfg:9s} R*(none)={res[cfg]['prior']:.4f}   best c {100*best['c']:5.1f}% "
              f"(k={best['k']})   median c {100*np.median([x['c'] for x in r]):5.1f}%   "
              f"worst {100*r[-1]['c']:5.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--n-outer", type=int, default=300)
    ap.add_argument("--n-random", type=int, default=60)
    a = ap.parse_args()
    main(only=a.only, n_outer=a.n_outer, n_random=a.n_random)
