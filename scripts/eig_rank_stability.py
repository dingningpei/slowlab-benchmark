#!/usr/bin/env python3
"""EIG does not converge in M, so do the paper's comparisons still hold?

EIG/H decreases monotonically in M (Appendix G), so its absolute value is biased.
But every comparison the paper draws is *between designs at the same M*. If the
bias has roughly the same direction and magnitude across designs, the ordering
survives; if it does not, every sentence in that section has to be withdrawn.

This script takes both kinds of design -- those of the reference strategies and
those three models actually submitted -- computes each at M = 80 and M = 320, and
reports the rank correlation and how well the gaps are preserved.

About 170 s per sandbox call, so results are written per design; run repeatedly
until it prints ALL DONE.
"""
from __future__ import annotations
import sys, json, glob, pathlib, re, argparse, pickle, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from slowlab import SlowLabEnv, TASKS
from slowlab.eig import build_atoms, eig_of_design
from eig_of_llm import designs_from, atoms_for


def reference_designs(task, n=6, seed=0):
    """Space-filling / two-treatment / single-point -- the range of shapes models actually used."""
    env = SlowLabEnv(task, seed=0)
    k = min(task.units_per_round, len(env.facility.units))
    blk = np.array([env.facility.units[i].chamber for i in range(k)])
    rng = np.random.default_rng(seed)
    d = len(task.factors)
    out = {}
    out["random"] = (rng.random((k, d)), blk)                    # space-filling
    two = np.repeat(rng.random((2, d)), k // 2 + 1, axis=0)[:k]  # two treatments replicated
    out["two_treat"] = (two, blk)
    out["one_point"] = (np.repeat(rng.random((1, d)), k, axis=0), blk)
    corners = ((rng.random((k, d)) > .5).astype(float))          # a factorial design
    out["factorial"] = (corners, blk)
    return out


def main(cfg="Screen", M_lo=80, M_hi=320, n_outer=120, n_llm=6,
         out=None, budget=140.0):
    out = out or f"results/eig_rank_{cfg}.json"   # one file per task, so they do not overwrite each other
    fp = ROOT / out
    res = json.loads(fp.read_text()) if fp.exists() else {}
    if res.get("cfg") != cfg:
        res = {"cfg": cfg, "M_lo": M_lo, "M_hi": M_hi, "items": {}}
    t = TASKS[cfg]
    env0 = SlowLabEnv(t, seed=0); sd = env0.truth.response_sd
    taus = (t.tau_chamber * sd, t.tau_loop * sd, t.tau_batch * sd)

    # Designs under test: the reference shapes plus a few real designs per model
    items = dict(reference_designs(t))
    for slug in ("z-ai_glm-5.3-flash", "deepseek_deepseek-v4-flash", "xiaomi_mimo-v2.5"):
        got = 0
        for f in sorted(glob.glob(str(ROOT / "results" / "llm" /
                                      f"transcript_{slug}@{cfg}_{cfg}_s*.json"))):
            s = int(re.search(r"_s(\d+)", f).group(1))
            for j, (p, b) in enumerate(designs_from(f, SlowLabEnv(t, seed=s))):
                if got >= n_llm:
                    break
                items[f"{slug.split('_')[-1]}#{s}.{j}"] = (p, b)
                got += 1
            if got >= n_llm:
                break

    t0 = time.time(); left = 0
    A = {}
    for M in (M_lo, M_hi):
        for name, (pts, blk) in items.items():
            key = f"{name}|M{M}"
            if key in res["items"]:
                continue
            if time.time() - t0 > budget:
                left += 1
                continue
            if M not in A:
                A[M] = atoms_for(cfg, M, 12)
            a = A[M]
            v = float(eig_of_design(a, pts, blk, *taus, n_outer=n_outer,
                                    rng=np.random.default_rng(11)))
            res["items"][key] = {"eig": v, "H": float(a.prior_entropy()),
                                 "frac": v / float(a.prior_entropy())}
            fp.write_text(json.dumps(res, indent=1))
    fp.write_text(json.dumps(res, indent=1))

    lo = {k.split("|")[0]: v["frac"] for k, v in res["items"].items()
          if k.endswith(f"|M{M_lo}")}
    hi = {k.split("|")[0]: v["frac"] for k, v in res["items"].items()
          if k.endswith(f"|M{M_hi}")}
    both = sorted(set(lo) & set(hi))
    if left or len(both) < len(items):
        print(f"{len(both)}/{len(items)} designs done; run again")
        return
    x = np.array([lo[k] for k in both]); y = np.array([hi[k] for k in both])
    from scipy.stats import spearmanr, pearsonr
    print(f"[{cfg}]  M={M_lo} vs M={M_hi}, {len(both)} designs")
    print(f"{'design':26}{f'M={M_lo}':>9}{f'M={M_hi}':>9}{'ratio':>8}")
    for k in sorted(both, key=lambda k: -lo[k]):
        print(f"{k:26}{100*lo[k]:>8.0f}%{100*hi[k]:>8.0f}%{hi[k]/lo[k]:>8.2f}")
    print(f"\nSpearman ρ = {spearmanr(x, y).statistic:+.3f}    "
          f"Pearson r = {pearsonr(x, y).statistic:+.3f}")
    print(f"shrinkage hi/lo: mean {np.mean(y/x):.2f}, range {np.min(y/x):.2f}-{np.max(y/x):.2f}")
    print("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="Screen")
    ap.add_argument("--budget", type=float, default=140.0)
    ap.add_argument("--n_llm", type=int, default=6)
    a = ap.parse_args()
    main(cfg=a.cfg, budget=a.budget, n_llm=a.n_llm)
