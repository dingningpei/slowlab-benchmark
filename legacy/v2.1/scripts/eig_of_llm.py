#!/usr/bin/env python3
"""Compute EIG/H from designs with accepted and completed execution events.

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
import sys, json, glob, pathlib, re, argparse, pickle, time, os
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import slowlab
from slowlab import SlowLabEnv, TASKS

# The LLM results directory carries the environment version, so transcripts from
# an old environment cannot be picked up silently.
LLM_DIR = f"llm_env{slowlab.ENV_VERSION}"
TRANSCRIPT_DIR = pathlib.Path(os.environ.get(
    "SLOWLAB_TRANSCRIPT_DIR", ROOT / "results" / LLM_DIR))
CACHE_DIR = pathlib.Path(os.environ.get(
    "SLOWLAB_CACHE_DIR", ROOT / "results"))
from slowlab.eventlog import design_arrays_from_events
from slowlab.eig import build_atoms, eig_of_design

CFGS = ("Sanity", "Screen", "Optimise", "Transfer")


def event_path_for(path):
    """Resolve a transcript identity to its recovered event file.

    ``SLOWLAB_EVENT_DIR`` may point at a private recovery directory. Missing
    events are fatal so a scoring run cannot silently fall back to transcripts.
    """
    source = pathlib.Path(path)
    if source.name.startswith("events_"):
        candidate = source
    elif source.name.startswith("transcript_"):
        directory = pathlib.Path(os.environ.get("SLOWLAB_EVENT_DIR", source.parent))
        candidate = directory / source.name.replace("transcript_", "events_", 1)
    else:
        raise ValueError(f"not a transcript or event path: {source}")
    if not candidate.exists():
        raise FileNotFoundError(
            f"completed-event source missing: {candidate}; run "
            "scripts/recover_execution_events.py first")
    return candidate


def designs_from(path, env):
    """Return only accepted designs with a matching completed event."""
    return design_arrays_from_events(event_path_for(path), env)


def atoms_for(cfg, M, nbins):
    """Atom-set cache. M, nbins and **the environment version** all go into the key.

    Putting the version in the key is part of the freeze discipline: the atom set
    is a function of ground truth, so it must be rebuilt whenever the environment
    changes. The key used to contain only M and nbins, and stale atom sets sitting
    in results/ were silently reused.
    """
    import slowlab
    cache = (CACHE_DIR /
             f"_atoms_{cfg}_M{M}_b{nbins}_env{slowlab.ENV_VERSION}.pkl")
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    t = TASKS[cfg]
    env0 = SlowLabEnv(t, seed=0)
    a = build_atoms(t, M=M, nbins=nbins,
                    plants_per_unit=env0.facility.plants_per_unit(3.25))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(a))
    return a


def main(M=80, n_outer=40, nbins=12,
         out="results/eig_llm_completed_events.json",
         only=None, models=None, budget=140.0):
    fp = ROOT / out
    res = json.loads(fp.read_text()) if fp.exists() else {}
    t0 = time.time()
    remaining = 0
    for cfg in (only or CFGS):
        t = TASKS[cfg]
        sd = SlowLabEnv(t, seed=0).truth.response_sd
        taus = (t.tau_chamber * sd, t.tau_loop * sd, t.tau_batch * sd)
        slot = res.setdefault(cfg, {"H_prior": None, "raw": {},
                                    "design_source": "completed_events"})
        atoms = None
        for f in sorted(glob.glob(str(
                TRANSCRIPT_DIR / f"transcript_*_{cfg}_s*.json"))):
            name = pathlib.Path(f).name
            model = re.match(
                rf"transcript_(.+)_{re.escape(cfg)}_s\d+\.json$", name
            ).group(1)
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
            m = re.match(
                rf"transcript_(.+)_{re.escape(cfg)}_s\d+\.json$", name
            ).group(1)
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
