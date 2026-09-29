#!/usr/bin/env python3
"""Convert the designs each model (and each reference strategy) submitted into the
achievable regret R*(D).

This produces the decomposition the results section needs, entirely in one
currency:

    R_actual  =  R*(D)  +  ( R_actual - R*(D) )
                 design limit   inference limit

A sandbox call lasts about 170 s and does not keep background processes alive, so
results are written per transcript; run repeatedly until it prints ALL DONE.
"""
from __future__ import annotations
import sys, json, glob, pathlib, re, argparse, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import slowlab
from slowlab import SlowLabEnv, TASKS

# The LLM results directory carries the environment version, so transcripts from
# an old environment cannot be picked up silently.
LLM_DIR = f"llm_env{slowlab.ENV_VERSION}"
from slowlab.achievable import achievable_regret, candidate_set, response_table
from slowlab.registry import make_agent
from eig_of_llm import designs_from, atoms_for

CFGS = ("Sanity", "Screen", "Optimise", "Transfer")
REFS = ("random_spread", "classical_doe", "gp_ucb_rep1", "gp_ucb_rep2")


from run_eig import designs_of   # reuse the same parser so the two cannot drift apart


def main(M=80, n_outer=200, nbins=12, seeds=8,
         out=None, budget=130.0, only=None, refs_only=False):
    # The output path carries the environment version, so stale results sitting in
    # results/ are not silently extended.
    import slowlab
    out = out or f"results/achievable_env{slowlab.ENV_VERSION}_M{M}.json"
    fp = ROOT / out
    res = json.loads(fp.read_text()) if fp.exists() else {}
    t0 = time.time(); left = 0

    for cfg in (only or CFGS):
        t = TASKS[cfg]
        env0 = SlowLabEnv(t, seed=0); sd = env0.truth.response_sd
        taus = (t.tau_chamber * sd, t.tau_loop * sd, t.tau_batch * sd)
        slot = res.setdefault(cfg, {"M": M, "n_outer": n_outer,
                                    "prior": None, "raw": {}})
        pack = None

        def prep():
            """The atom set and candidate table are built only if this cfg actually has work to do."""
            a = atoms_for(cfg, M, nbins)
            cand = candidate_set(a, rng=np.random.default_rng(0))
            return a, dict(F=response_table(a, cand), cand=cand, n_outer=n_outer)

        if slot["prior"] is None:
            pack = prep(); a, kw = pack
            slot["prior"] = achievable_regret(
                a, np.zeros((0, len(t.factors))), np.zeros(0, int), *taus, **kw)
            fp.write_text(json.dumps(res, indent=1))

        # ── models under test ──
        # refs_only: transcripts from an old environment cannot be scored against a
        # new world's ceiling -- the agent chose those designs in a different world.
        # Re-enable after a rerun.
        for f in ([] if refs_only else sorted(glob.glob(str(ROOT / "results" / LLM_DIR /
                                      f"transcript_*@{cfg}_{cfg}_s*.json")))):
            name = pathlib.Path(f).name
            if name in slot["raw"]:
                continue
            if time.time() - t0 > budget:
                left += 1; continue
            if pack is None:
                pack = prep()
            a, kw = pack
            seed = int(re.search(r"_s(\d+)\.json$", f).group(1))
            env = SlowLabEnv(t, seed=seed)
            vals = [achievable_regret(a, p, b, *taus,
                                      rng=np.random.default_rng(100 + seed), **kw)
                    for p, b in designs_from(f, env)]
            slot["raw"][name] = [float(v) for v in vals]
            fp.write_text(json.dumps(res, indent=1))

        # ── reference strategies ──
        for ag in REFS:
            key = f"__ref__{ag}"
            if key in slot["raw"]:
                continue
            if time.time() - t0 > budget:
                left += 1; continue
            if pack is None:
                pack = prep()
            a, kw = pack
            vals = []
            for s in range(seeds):
                env = SlowLabEnv(t, seed=s); agent = make_agent(ag)
                env.submit_recommendation(agent.run(env), ag)
                vals += [achievable_regret(a, p, b, *taus,
                                           rng=np.random.default_rng(100 + s), **kw)
                         for p, b in designs_of(env)]
            slot["raw"][key] = [float(v) for v in vals]
            fp.write_text(json.dumps(res, indent=1))

    # Aggregate (raw is the single source of truth)
    for cfg, slot in res.items():
        agg = {}
        for name, vals in slot["raw"].items():
            m = (name.replace("__ref__", "ref:") if name.startswith("__ref__")
                 else re.search(r"transcript_(.+?)@", name).group(1))
            agg.setdefault(m, []).extend(vals)
        slot["agents"] = {m: {"R_star": float(np.mean(v)), "n": len(v)}
                          for m, v in agg.items() if v}
    fp.write_text(json.dumps(res, indent=1))

    for cfg in CFGS:
        if cfg not in res:
            continue
        s = res[cfg]
        print(f"{cfg:9s} R*(none)={s['prior']:.4f}", flush=True)
        for m, d in sorted(s["agents"].items(), key=lambda kv: kv[1]["R_star"]):
            print(f"    {m:30s} R*={d['R_star']:.4f}  n={d['n']}", flush=True)
    print("ALL DONE" if left == 0 else f"{left} items remaining; run again", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=80)
    ap.add_argument("--n_outer", type=int, default=200)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--budget", type=float, default=130.0)
    ap.add_argument("--refs-only", action="store_true")
    a = ap.parse_args()
    main(M=a.M, n_outer=a.n_outer, only=a.only, budget=a.budget,
         refs_only=a.refs_only)
