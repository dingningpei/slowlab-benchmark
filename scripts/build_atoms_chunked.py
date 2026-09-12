#!/usr/bin/env python3
"""Build the atom set in chunks. M=320 takes about 185s under the new oracle, past the
sandbox's per-call limit.

Run repeatedly until it prints ALL DONE. Intermediate results land in
_atomseed_{cfg}_M{M}.pkl; the final product has the same name and structure as the
cache file of eig_of_llm.atoms_for.
"""
from __future__ import annotations
import sys, pickle, pathlib, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import slowlab
from slowlab import SlowLabEnv, TASKS
from slowlab.eig import build_atoms

CFGS = ("Sanity", "Optimise", "Screen", "Transfer")


def main(M=320, nbins=12, budget=140.0, chunk=40):
    for cfg in CFGS:
        final = (ROOT / "results" /
                 f"_atoms_{cfg}_M{M}_b{nbins}_env{slowlab.ENV_VERSION}.pkl")
        if final.exists():
            continue
        part = (ROOT / "results" /
                f"_atomseed_{cfg}_M{M}_env{slowlab.ENV_VERSION}.pkl")
        done = pickle.loads(part.read_bytes()) if part.exists() else 0
        t = TASKS[cfg]
        t0 = time.time()
        while done < M and time.time() - t0 < budget:
            done = min(M, done + chunk)
            part.write_bytes(pickle.dumps(done))
            # Warm the oracle cache: build_atoms later reuses the same seeds
            from slowlab.world import ManagedTomgro
            for s in range(done - chunk, done):
                ManagedTomgro(seed=10_000 + s, factors=t.factors,
                              cycle_days=t.cycle_days).oracle()
        if done < M:
            print(f"{cfg}: warmed {done}/{M}, run again", flush=True)
            return
        env0 = SlowLabEnv(t, seed=0)
        a = build_atoms(t, M=M, nbins=nbins,
                        plants_per_unit=env0.facility.plants_per_unit(3.25))
        final.write_bytes(pickle.dumps(a))
        try:                       # some mounts forbid deletion; a failed cleanup
                                   # must not fail the whole build
            part.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"{cfg}: built M={M}", flush=True)
        return
    print("ALL DONE")


if __name__ == "__main__":
    main(M=int(sys.argv[1]) if len(sys.argv) > 1 else 320)
