#!/usr/bin/env python3
"""Quantify, one at a time, the three approximations in the EIG estimator.

  (a) Gaussian moment matching of the per-plant response -- reports skewness and
      kurtosis, to show how much work the approximation is doing
  (b) s^2 computed only at the nominal theta -- reports its variation across atoms
  (c) discretising P_Theta into M atoms -- reports EIG's convergence in M
Plus convergence in the outer sample size N.
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab import SlowLabEnv, TASKS
from slowlab.eig import build_atoms, eig_of_design
sys.path.insert(0, str(ROOT / "scripts"))
from eig_of_llm import atoms_for   # on-disk cache; do not pay for build_atoms twice
from slowlab.world import ManagedTomgro


def moments(task, n_unit=400, n_grid=8, seed=0):
    """(a) Third and fourth standardised moments of the unit-level response. A unit is the mean of N plants on one loop."""
    env = SlowLabEnv(task, seed=0)
    npu = env.facility.plants_per_unit(3.25)
    w = ManagedTomgro(seed=0, factors=task.factors, cycle_days=task.cycle_days)
    rng = np.random.default_rng(seed)
    out, skipped = [], 0
    for g in range(n_grid):
        x = rng.random((1, len(task.factors)))
        # Same vectorisation as env.py: compute n_unit x npu plants at once, then average per unit
        big = np.repeat(x, n_unit * npu, axis=0)
        pp = w.sample_plant_params(n_unit * npu, rng, task.plant_cv)
        v = np.asarray(w(big, plant_params=pp), float).reshape(n_unit, npu).mean(1)
        if v.std(ddof=1) < 1e-12:
            # This grid point kills the crop and every plant returns the same floor
            # value. That is a real property of the world model rather than a
            # defect, but there is no distribution to speak of, so skip and count it.
            skipped += 1
            continue
        z = (v - v.mean()) / v.std(ddof=1)
        out.append((float((z ** 3).mean()), float((z ** 4).mean() - 3.0)))
    sk = np.array([o[0] for o in out]); ku = np.array([o[1] for o in out])
    return {"skew_mean": float(sk.mean()), "skew_max_abs": float(np.abs(sk).max()),
            "exkurt_mean": float(ku.mean()), "exkurt_max_abs": float(np.abs(ku).max()),
            "n_unit": n_unit, "n_grid": n_grid, "n_used": len(out),
            "n_degenerate": skipped, "plants_per_unit": npu}


def s2_spread(task, n_atom=12, n_unit=120, seed=1):
    """(b) How much the per-plant variance differs across theta -- we use only the one at the nominal theta."""
    rng = np.random.default_rng(seed)
    x = np.full((1, len(task.factors)), 0.5)
    env = SlowLabEnv(task, seed=0); npu = env.facility.plants_per_unit(3.25)
    sds = []
    for m in range(n_atom):
        w = ManagedTomgro(seed=10_000 + m, factors=task.factors, cycle_days=task.cycle_days)
        big = np.repeat(x, n_unit * npu, axis=0)
        pp = w.sample_plant_params(n_unit * npu, rng, task.plant_cv)
        v = np.asarray(w(big, plant_params=pp), float).reshape(n_unit, npu).mean(1)
        sds.append(float(np.std(v, ddof=1)))
    sds = np.asarray(sds)
    return {"sd_min": float(sds.min()), "sd_max": float(sds.max()),
            "sd_cv": float(sds.std(ddof=1) / sds.mean()), "n_atom": n_atom}


def convergence(task, cfg, Ms=(40, 80, 160, 320), Ns=(40, 50, 100, 200, 400), seed=3,
                have=None, on_step=None):
    """(c) Convergence of EIG in M (atom count) and N (outer sample), at a fixed design."""
    env = SlowLabEnv(task, seed=0); sd = env.truth.response_sd
    taus = (task.tau_chamber * sd, task.tau_loop * sd, task.tau_batch * sd)
    npu = env.facility.plants_per_unit(3.25)
    rng = np.random.default_rng(seed)
    n = min(task.units_per_round, len(env.facility.units))
    pts = rng.random((n, len(task.factors)))
    blk = np.array([env.facility.units[i].chamber for i in range(n)])
    have = have or {"vs_M": [], "vs_N": []}
    # A JSON round trip turns tuples into lists; normalise before comparing, otherwise sorted blows up
    outM = [list(r) for r in have.get("vs_M", [])]
    outN = [list(r) for r in have.get("vs_N", [])]
    doneM = {r[0] for r in outM}; doneN = {r[0] for r in outN}
    for M in Ms:                              # ~170s per sandbox call, so write after each point
        if M in doneM:
            continue
        a = atoms_for(cfg, M, 12)
        outM.append([M, float(eig_of_design(a, pts, blk, *taus, n_outer=200,
                                            rng=np.random.default_rng(7))),
                     float(a.prior_entropy())])
        if on_step:
            on_step({"vs_M": outM, "vs_N": outN})
    if set(Ns) - doneN:
        a = atoms_for(cfg, max(Ms), 12)
        for N in Ns:
            if N in doneN:
                continue
            outN.append([N, float(eig_of_design(a, pts, blk, *taus, n_outer=N,
                                                rng=np.random.default_rng(7)))])
            if on_step:
                on_step({"vs_M": outM, "vs_N": outN})
    return {"vs_M": sorted(outM), "vs_N": sorted(outN)}


def main(cfg="Optimise", out="results/eig_validation.json"):
    """Resumable in stages: finished stages are skipped. Run repeatedly until it prints ALL DONE."""
    t = TASKS[cfg]
    fp = ROOT / out
    res = json.loads(fp.read_text()) if fp.exists() else {}
    if res.get("task") != cfg:
        res = {"task": cfg}
    save = lambda: fp.write_text(json.dumps(res, indent=1))
    if "moments" not in res:
        res["moments"] = moments(t); save()
    if "s2_spread" not in res:
        res["s2_spread"] = s2_spread(t); save()

    def step(partial):
        res["convergence"] = partial; save()

    res["convergence"] = convergence(t, cfg, have=res.get("convergence"), on_step=step)
    save()
    m, s2, c = res["moments"], res["s2_spread"], res["convergence"]
    print(f"[{cfg}]  one unit = the mean of {m['plants_per_unit']} plants")
    print(f"(a) Gaussian moment match   skew {m['skew_mean']:+.3f} (|max| {m['skew_max_abs']:.3f})   "
          f"excess kurtosis {m['exkurt_mean']:+.3f} (|max| {m['exkurt_max_abs']:.3f})   "
          f"[{m['n_used']}/{m['n_grid']} grid points, {m['n_degenerate']} degenerate]")
    print(f"(b) s^2 across atoms   sd {s2['sd_min']:.5f}-{s2['sd_max']:.5f}, CV {s2['sd_cv']:.3f}")
    print(f"(c) EIG vs M     " + "  ".join(f"M={M}:{v:.3f}" for M, v, _ in c["vs_M"]))
    print(f"    EIG vs N     " + "  ".join(f"N={N}:{v:.3f}" for N, v in c["vs_N"]))
    done = (len(c["vs_M"]) == 4 and len(c["vs_N"]) == 4)
    print("ALL DONE" if done else "not finished; run again")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Optimise")
