#!/usr/bin/env python3
"""Extract the designs a model actually submitted from its transcripts and compute
their EIG/H.

This answers the fork that matters most:
  * low EIG with poor regret -- the design carried no information
  * decent EIG with poor regret -- the information was collected and not used
Those are two entirely different papers.

A sandbox call lasts about 170 seconds and does not keep background processes
alive, so this script:
  * caches the atom set on disk (build_atoms is the expensive step and should not
    be paid for twice)
  * writes out after every transcript
Call it repeatedly until it prints ALL DONE.
"""
from __future__ import annotations
import sys, json, glob, pathlib, re, argparse, pickle, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import slowlab
from slowlab import SlowLabEnv, TASKS

# The LLM results directory carries the environment version, so transcripts from
# an old environment cannot be picked up silently.
LLM_DIR = f"llm_env{slowlab.ENV_VERSION}"
from slowlab.llm import _extract_json
from slowlab.eig import build_atoms, eig_of_design

CFGS = ("Sanity", "Screen", "Optimise", "Transfer")


def designs_from(path, env):
    """transcript -> [(pts, blocks)]. Skips the final recommendation and the no-experiment questions."""
    names = [f.name for f in env.task.factors]
    out = []
    for turn in json.loads(pathlib.Path(path).read_text()):
        try:
            b = _extract_json(turn["assistant"])
        except Exception:
            continue
        if "treatments" not in b or "allocation" not in b:
            continue
        pts, blk = [], []
        for tid, uids in b["allocation"].items():
            t = b["treatments"].get(tid)
            if not t:
                continue
            v, ok = [], True
            for j, n in enumerate(names):
                f = env.task.factors[j]
                try:
                    v.append((float(t[n]) - f.low) / (f.high - f.low))
                except Exception:
                    ok = False; break
            if not ok:
                continue
            for u in (uids if isinstance(uids, list) else [uids]):
                m = re.match(r"c(\d+)l(\d+)", str(u))
                if m and str(u) in env.facility._by_id:
                    pts.append(v); blk.append(int(m.group(1)))
        if pts:
            out.append((np.clip(np.asarray(pts, float), 0, 1), np.asarray(blk, int)))
    return out


def atoms_for(cfg, M, nbins):
    """Atom-set cache. M, nbins and **the environment version** all go into the key.

    Putting the version in the key is part of the freeze discipline: the atom set
    is a function of ground truth, so it must be rebuilt whenever the environment
    changes. The key used to contain only M and nbins, and stale atom sets sitting
    in results/ were silently reused.
    """
    import slowlab
    cache = (ROOT / "results" /
             f"_atoms_{cfg}_M{M}_b{nbins}_env{slowlab.ENV_VERSION}.pkl")
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    t = TASKS[cfg]
    env0 = SlowLabEnv(t, seed=0)
    a = build_atoms(t, M=M, nbins=nbins,
                    plants_per_unit=env0.facility.plants_per_unit(3.25))
    cache.write_bytes(pickle.dumps(a))
    return a


def main(M=80, n_outer=40, nbins=12, out="results/eig_llm.json",
         only=None, models=None, budget=140.0):
    fp = ROOT / out
    res = json.loads(fp.read_text()) if fp.exists() else {}
    t0 = time.time()
    remaining = 0
    for cfg in (only or CFGS):
        t = TASKS[cfg]
        sd = SlowLabEnv(t, seed=0).truth.response_sd
        taus = (t.tau_chamber * sd, t.tau_loop * sd, t.tau_batch * sd)
        slot = res.setdefault(cfg, {"H_prior": None, "raw": {}})
        atoms = None
        for f in sorted(glob.glob(str(ROOT / "results" / LLM_DIR /
                                      f"transcript_*@{cfg}_{cfg}_s*.json"))):
            name = pathlib.Path(f).name
            model = re.search(r"transcript_(.+?)@", name).group(1)
            if models and model not in models:
                continue
            if name in slot["raw"]:                    # already computed
                continue
            if time.time() - t0 > budget:              # leave enough time to write out
                remaining += 1
                continue
            if atoms is None:
                atoms = atoms_for(cfg, M, nbins)
                slot["H_prior"] = atoms.prior_entropy()
            seed = int(re.search(r"_s(\d+)\.json$", f).group(1))
            env = SlowLabEnv(t, seed=seed)
            vals = [eig_of_design(atoms, p, b, *taus, n_outer=n_outer,
                                  rng=np.random.default_rng(100 + seed))
                    for p, b in designs_from(f, env)]
            slot["raw"][name] = [float(v) for v in vals]
            fp.write_text(json.dumps(res, indent=1))   # written out after every transcript

    # The aggregate view (raw is the single source of truth; models is derived and recomputed each time)
    for cfg, slot in res.items():
        H = slot.get("H_prior")
        agg = {}
        for name, vals in slot.get("raw", {}).items():
            m = re.search(r"transcript_(.+?)@", name).group(1)
            agg.setdefault(m, []).extend(vals)
        slot["models"] = {m: {"eig": float(np.mean(v)), "n": len(v),
                              "frac": float(np.mean(v) / H) if H else None}
                          for m, v in agg.items() if v}
    fp.write_text(json.dumps(res, indent=1))

    for cfg in CFGS:
        if cfg not in res:
            continue
        s = res[cfg]
        line = f"{cfg:9s} H={s['H_prior'] or float('nan'):.2f}"
        for m, d in sorted(s["models"].items()):
            line += f"   {m} {100*d['frac']:.0f}%({d['n']})"
        print(line, flush=True)
    print("ALL DONE" if remaining == 0 else
          f"{remaining} transcripts remaining; run again", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=80)
    ap.add_argument("--n_outer", type=int, default=40)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--budget", type=float, default=140.0)
    a = ap.parse_args()
    main(M=a.M, n_outer=a.n_outer, only=a.only, models=a.models, budget=a.budget)
