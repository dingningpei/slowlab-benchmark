#!/usr/bin/env python3
"""Two things that are easy to conflate: the cost of setting a factor *wrongly*,
and the value of *locating* it by experiment.

The paper uses these numbers to say that a sensitivity analysis points at the
wrong factor, and until now they had been computed once with no script behind
them.

  cost of midrange   = this factor at the midpoint of its range with the others at
                       the site's own optimum, against the site's optimum.
  value of locating  = this factor at the *across-site mean* optimum with the
                       others at the site's own optimum, against the site's
                       optimum. If the factor's optimum is the same at every site
                       this is 0: no experiment is needed, the literature suffices.

A factor can have a large first quantity and a zero second one -- it matters, but
it is not worth an experiment to find.

    python scripts/factor_value.py
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import slowlab
from slowlab import SlowLabEnv, TASKS

N_SITES = 24


def per_site_optima(cfg, n=N_SITES):
    """Each site's optimum point and optimal value."""
    X, V, envs = [], [], []
    for s in range(n):
        e = SlowLabEnv(TASKS[cfg], seed=s)
        x, v = e.truth.oracle()
        X.append(np.asarray(x, float).ravel()); V.append(float(v)); envs.append(e)
    return np.array(X), np.array(V), envs


def main():
    out = {}
    for cfg in ("Screen", "Optimise"):
        t = TASKS[cfg]
        names = [f.name for f in t.factors]
        X, V, envs = per_site_optima(cfg)
        xbar = X.mean(axis=0)                      # the across-site mean optimum
        rows = []
        for j, nm in enumerate(names):
            mid, loc = [], []
            for k, e in enumerate(envs):
                x = X[k].copy(); x[j] = 0.5        # set to the midpoint of the range
                mid.append(V[k] - float(e.truth(x.reshape(1, -1))[0]))
                x = X[k].copy(); x[j] = xbar[j]    # set to the across-site mean optimum
                loc.append(V[k] - float(e.truth(x.reshape(1, -1))[0]))
            rows.append({"factor": nm, "cost_of_midrange": float(np.mean(mid)),
                         "value_of_locating": float(np.mean(loc)),
                         "optimum_mean": float(xbar[j]),
                         "optimum_sd": float(X[:, j].std())})
        tot = sum(r["value_of_locating"] for r in rows)
        for r in rows:
            r["share_of_locating"] = r["value_of_locating"] / tot if tot else 0.0
        out[cfg] = {"n_sites": len(X), "factors": rows, "total_locating": tot}

        print(f"--- {cfg}  ({len(X)} sites)")
        print(f"{'factor':16s}{'cost_midrange':>15}{'value_locating':>16}{'share':>7}"
              f"{'opt_mean':>10}{'opt_sd':>9}")
        for r in sorted(rows, key=lambda r: -r["cost_of_midrange"]):
            print(f"{r['factor']:16s}{r['cost_of_midrange']:12.4f}"
                  f"{r['value_of_locating']:12.4f}{100*r['share_of_locating']:6.0f}%"
                  f"{r['optimum_mean']:11.2f}{r['optimum_sd']:10.3f}")
        top = max(rows, key=lambda r: r["cost_of_midrange"])
        best = sorted(rows, key=lambda r: -r["value_of_locating"])[:2]
        print(f"  steepest factor is {top['factor']} (cost of midrange {top['cost_of_midrange']:.4f}), "
              f"but locating it is worth only {top['value_of_locating']:.4f}")
        print(f"  the two most worth locating: {best[0]['factor']} {best[0]['value_of_locating']:.4f}, "
              f"{best[1]['factor']} {best[1]['value_of_locating']:.4f}, together "
              f"{100*(best[0]['share_of_locating']+best[1]['share_of_locating']):.0f}%\n")

    p = ROOT / "results" / f"factor_value_env{slowlab.ENV_VERSION}.json"
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p.name)


if __name__ == "__main__":
    main()
