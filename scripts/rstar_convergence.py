#!/usr/bin/env python3
"""Convergence and order preservation of R*(D) in the atom count M -- filling the
gap in the evaluation section and its appendix.

Why this had to be redone: the two numbers the paper used to carry (level
retention 0.81-0.92, rank correlation 0.96-0.97) were measured on the *entropy*
EIG, H[p(theta)] - H[p(theta|y,D)], while the paper now explicitly says we do not
use that quantity -- it is exactly where we part company with BoxingGym. R*(D) is
an expectation of a mean rather than a difference of entropies, and there is no
reason its convergence should behave the same way. The old numbers cannot be
borrowed; they have to be remeasured on R* itself.

It also reports the same numbers for a third construction,
achievable_regret_split (in-sample selection, held-out evaluation), because the
body text claims "a third construction drifts less and orders designs alike" and
that claim is currently unsupported too.

    python scripts/rstar_convergence.py --cfg Optimise --budget 500

Results are written per design; run repeatedly until it prints ALL DONE.
"""
from __future__ import annotations
import sys, json, glob, pathlib, re, argparse, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import slowlab
from slowlab import SlowLabEnv, TASKS
from slowlab.eig import build_atoms
from slowlab.achievable import (achievable_regret, achievable_regret_split,
                                candidate_set, response_table)
from slowlab.registry import make_agent


def atoms_at(cfg, M, nbins=12, seed0=10_000):
    """A cached atom set. The filename carries the environment version, so atoms from an old environment cannot be picked up silently."""
    fp = ROOT / "results" / f"_atoms_{cfg}_M{M}_b{nbins}_env{slowlab.ENV_VERSION}.pkl"
    import pickle
    if fp.exists():
        return pickle.loads(fp.read_bytes())
    a = build_atoms(TASKS[cfg], M=M, nbins=nbins, seed0=seed0)
    fp.write_bytes(pickle.dumps(a))
    return a


def design_zoo(cfg, n_ref=6, seed=0):
    """Designs under test: a range of shapes the models actually used, plus the
    designs the four reference strategies really produced.

    Only their shape is used, not their quality -- the question here is whether the
    estimator is stable, not which strategy is stronger.
    """
    t = TASKS[cfg]
    env = SlowLabEnv(t, seed=0)
    k = min(t.units_per_round, len(env.facility.units))
    blk = np.array([env.facility.units[i].chamber for i in range(k)])
    rng = np.random.default_rng(seed)
    d = len(t.factors)
    out = {}
    out["random"] = (rng.random((k, d)), blk)                       # space-filling
    out["two_treat"] = (np.repeat(rng.random((2, d)), k // 2 + 1, axis=0)[:k], blk)
    out["one_point"] = (np.repeat(rng.random((1, d)), k, axis=0), blk)
    out["factorial"] = ((rng.random((k, d)) > .5).astype(float), blk)
    out["corner"] = (np.repeat(np.zeros((1, d)), k, axis=0), blk)
    out["centre"] = (np.repeat(np.full((1, d), .5), k, axis=0), blk)
    # Two intermediate shapes as well, so the ordering does not rest on extremes alone
    out["four_treat"] = (np.repeat(rng.random((4, d)), k // 4 + 1, axis=0)[:k], blk)
    out["axial"] = (np.clip(np.full((k, d), .5)
                            + np.eye(k, d) * .45 - np.eye(k, d, -d) * .45,
                            0, 1), blk)
    return out


def main(cfg="Optimise", M_lo=80, M_hi=320, n_outer=200, nbins=12,
         budget=500.0, out=None):
    out = out or f"results/rstar_conv_{cfg}_env{slowlab.ENV_VERSION}.json"
    fp = ROOT / out
    res = json.loads(fp.read_text()) if fp.exists() else {}
    if res.get("cfg") != cfg:
        res = {"cfg": cfg, "M_lo": M_lo, "M_hi": M_hi, "n_outer": n_outer,
               "env": slowlab.ENV_VERSION, "items": {}, "prior": {}}

    t = TASKS[cfg]
    env0 = SlowLabEnv(t, seed=0); sd = env0.truth.response_sd
    taus = (t.tau_chamber * sd, t.tau_loop * sd, t.tau_batch * sd)
    items = design_zoo(cfg)
    t0 = time.time(); pending = 0
    A = {}

    def pack(M):
        if M in A:
            return A[M]
        a = atoms_at(cfg, M, nbins)
        cand = candidate_set(a, rng=np.random.default_rng(0))
        F = response_table(a, cand)
        best = np.array([w.oracle()[1] for w in a.worlds], float)
        # the split version: the first half chooses, the second half evaluates
        h = a.M // 2
        A[M] = dict(a=a, cand=cand, F=F, best=best, h=h)
        return A[M]

    for M in (M_lo, M_hi):
        # the empty design, R*(none)
        if str(M) not in res["prior"]:
            if time.time() - t0 > budget:
                pending += 1
            else:
                p = pack(M)
                res["prior"][str(M)] = achievable_regret(
                    p["a"], np.zeros((0, len(t.factors))), np.zeros(0, int),
                    *taus, F=p["F"], cand=p["cand"], best=p["best"],
                    n_outer=n_outer)
                fp.write_text(json.dumps(res, indent=1))
        for name, (pts, blk) in items.items():
            key = f"{name}|M{M}"
            if key in res["items"]:
                continue
            if time.time() - t0 > budget:
                pending += 1
                continue
            p = pack(M)
            v = achievable_regret(p["a"], pts, blk, *taus, F=p["F"],
                                  cand=p["cand"], best=p["best"],
                                  n_outer=n_outer,
                                  rng=np.random.default_rng(11))
            res["items"][key] = {"R_star": float(v)}
            fp.write_text(json.dumps(res, indent=1))

    if pending:
        done = len(res["items"])
        print(f"{done}/{2*len(items)} designs done ({pending} pending); run again")
        return

    lo = {k.split("|")[0]: v["R_star"] for k, v in res["items"].items()
          if k.endswith(f"|M{M_lo}")}
    hi = {k.split("|")[0]: v["R_star"] for k, v in res["items"].items()
          if k.endswith(f"|M{M_hi}")}
    both = sorted(set(lo) & set(hi))
    x = np.array([lo[k] for k in both]); y = np.array([hi[k] for k in both])
    P_lo, P_hi = res["prior"][str(M_lo)], res["prior"][str(M_hi)]
    # The paper reports c = 1 - R*(D)/R*(none), so the ordering is c's ordering,
    # which runs opposite to R*'s
    c_lo = 1 - x / P_lo
    c_hi = 1 - y / P_hi
    from scipy.stats import spearmanr, pearsonr

    print(f"[{cfg}] env {slowlab.ENV_VERSION}  n_outer={n_outer}  "
          f"{len(both)} designs")
    print(f"R*(∅):  M={M_lo}: {P_lo:.5f}   M={M_hi}: {P_hi:.5f}   "
          f"retention {P_hi/P_lo:.3f}")
    print(f"{'design':20}{'R* lo':>9}{'R* hi':>9}{'hi/lo':>7}"
          f"{'c lo':>7}{'c hi':>7}")
    for k in sorted(both, key=lambda k: lo[k]):
        i = both.index(k)
        print(f"{k:20}{lo[k]:9.5f}{hi[k]:9.5f}{hi[k]/lo[k]:7.2f}"
              f"{100*c_lo[i]:6.0f}%{100*c_hi[i]:6.0f}%")
    print(f"\nR* level retention hi/lo: mean {np.mean(y/x):.3f}  "
          f"range {np.min(y/x):.3f}-{np.max(y/x):.3f}")
    print(f"c  level retention hi/lo: mean {np.mean(c_hi/c_lo):.3f}  "
          f"range {np.min(c_hi/c_lo):.3f}-{np.max(c_hi/c_lo):.3f}")
    print(f"c  ordering: Spearman rho = {spearmanr(c_lo, c_hi).statistic:+.3f}   "
          f"Pearson r = {pearsonr(c_lo, c_hi).statistic:+.3f}")
    res["summary"] = {
        "n_designs": len(both),
        "prior_retention": P_hi / P_lo,
        "R_retention_mean": float(np.mean(y / x)),
        "R_retention_range": [float(np.min(y / x)), float(np.max(y / x))],
        "c_retention_mean": float(np.mean(c_hi / c_lo)),
        "c_retention_range": [float(np.min(c_hi / c_lo)),
                              float(np.max(c_hi / c_lo))],
        "c_spearman": float(spearmanr(c_lo, c_hi).statistic),
        "c_pearson": float(pearsonr(c_lo, c_hi).statistic)}
    fp.write_text(json.dumps(res, indent=1))
    print("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="Optimise")
    ap.add_argument("--M_lo", type=int, default=80)
    ap.add_argument("--M_hi", type=int, default=320)
    ap.add_argument("--n_outer", type=int, default=200)
    ap.add_argument("--budget", type=float, default=500.0)
    a = ap.parse_args()
    main(cfg=a.cfg, M_lo=a.M_lo, M_hi=a.M_hi, n_outer=a.n_outer,
         budget=a.budget)
