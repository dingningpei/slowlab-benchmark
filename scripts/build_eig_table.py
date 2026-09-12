#!/usr/bin/env python3
"""Build the M x G response table (the precomputed part of eq. (4) in the paper). Done
once; every later evaluation is a table lookup."""
from __future__ import annotations
import sys, pathlib, argparse, pickle, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab import TASKS
from slowlab.world import ManagedTomgro
from slowlab.eig import ResponseTable


def sobol(d, n, seed=0):
    try:
        from scipy.stats import qmc
        return qmc.Sobol(d, scramble=True, seed=seed).random(n)
    except Exception:
        return np.random.default_rng(seed).random((n, d))


def build(cfg="Optimise", M=400, G=1681, n_plant=24, seed=0):
    t = TASKS[cfg]; d = len(t.factors)
    if d <= 2:                                   # regular grid in low dimensions: easier to read off
        k = int(round(G ** (1 / d))); G = k ** d
        ax = [np.linspace(0, 1, k)] * d
        X = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, d)
    else:
        X = sobol(d, G, seed)
    t0 = time.time()
    F = np.empty((M, len(X)), float)
    for m in range(M):
        F[m] = ManagedTomgro(seed=10_000 + m, factors=t.factors)(X)
        if m % 50 == 0:
            print(f"  atom {m}/{M}  {time.time()-t0:.0f}s", flush=True)
    w = ManagedTomgro(seed=0, factors=t.factors)
    rng = np.random.default_rng(1)
    pp = w.sample_plant_params(n_plant, rng, t.plant_cv)
    draws = np.stack([w(X, plant_params={k: (v[i] if np.ndim(v) else v)
                                         for k, v in pp.items()})
                      for i in range(n_plant)])
    s2 = draws.var(0, ddof=1)
    tab = ResponseTable(X=X, F=F, s2=s2, xstar=np.argmax(F, axis=1), task_name=cfg)
    out = ROOT / "results" / f"eig_table_{cfg}.pkl"
    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as fh:
        pickle.dump(tab, fh)
    print(f"{cfg}: M={tab.M} G={tab.G}  H[p(x*)]={tab.prior_entropy():.2f} nats "
          f"(e^H={np.exp(tab.prior_entropy()):.1f})  mean s={np.sqrt(s2).mean():.4f}  "
          f"{time.time()-t0:.0f}s", flush=True)
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="Optimise")
    ap.add_argument("--M", type=int, default=400)
    ap.add_argument("--G", type=int, default=1681)
    a = ap.parse_args()
    build(a.cfg, a.M, a.G)
