"""The best fixed recommendation -- the oracle bound for the realised-regret column.

    x_fixed = argmin_x  E_{θ~P_Θ} [ f_θ(x*_θ) − f_θ(x) ]

It answers "how well can you do without experimenting at all". Three properties:

  * computable, uncited and free of selection bias -- unlike a hand-written
    "literature value", where it sits is not decided by which paper an author
    picked;
  * it is an oracle, not a strategy -- computing it needs P_Theta and each
    instance's f_theta, which no agent can read, so it is a bound rather than an
    opponent;
  * it is evaluated on held-out instances -- fitted on one set of seeds and
    reported on another, otherwise it is in-sample.

A candidate set of uniform random points is nowhere near dense enough in high
dimension: 20000 uniform points have converged in 1-2 dimensions but overstate
the bound by 78% in 4 and 81% in 6 (Screen 0.0029 against 0.0006). Uniform
spacing shrinks as n^(-1/d) while the optima concentrate near a low-dimensional
manifold, so random points almost never hit them.
The candidate set is therefore uniform points union the true optima of the
fitting worlds. The latter are still data-independent -- they use P_Theta, not
the agent's observations -- so this remains an oracle bound rather than a
strategy; it just computes that bound more accurately. Convergence is guarded by
a check in tests: adding more candidates should not change the answer.

An agent that beats it has demonstrated that it actually used its data.
"""
from __future__ import annotations
import json, pathlib
import numpy as np

_CACHE = pathlib.Path(__file__).resolve().parents[1] / "results" / "fixed_reference.json"


def _worlds(task, seeds):
    from .world import ManagedTomgro
    return [ManagedTomgro(seed=s, factors=task.factors,
                          cycle_days=task.cycle_days) for s in seeds]


def best_fixed(task, n_fit=40, n_cand=20_000, seed=0, cache=True):
    """Returns (x_fixed, regret on the held-out set)."""
    key = f"{task.name}|{n_fit}|{n_cand}|{seed}|v2"   # v2: candidate set includes the fitting optima
    if cache and _CACHE.exists():
        d = json.loads(_CACHE.read_text())
        if key in d:
            return np.asarray(d[key]["x"], float), float(d[key]["regret"])
    rng = np.random.default_rng(seed)
    Wf = _worlds(task, range(n_fit))
    We = _worlds(task, range(1000, 1000 + n_fit))          # held out, disjoint from the fitting seeds
    bf = np.array([w.oracle()[1] for w in Wf])
    be = np.array([w.oracle()[1] for w in We])
    # Uniform points plus the true optima of the fitting worlds. See the module
    # docstring: pure uniform sampling loosens this bound by 78-81% in 4-6
    # dimensions, and a loose bound is the direction that flatters the agent.
    C = np.vstack([rng.random((n_cand, len(task.factors))),
                   np.array([w.oracle()[0] for w in Wf], float).reshape(-1, len(task.factors))])
    R = np.zeros(len(C))
    for w, b in zip(Wf, bf):
        R += b - w(C)
    x = C[int(np.argmin(R / len(Wf)))]
    out = float(np.mean([b - float(w(x.reshape(1, -1))[0]) for w, b in zip(We, be)]))
    if cache:
        d = json.loads(_CACHE.read_text()) if _CACHE.exists() else {}
        d[key] = {"x": x.tolist(), "regret": out}
        _CACHE.parent.mkdir(exist_ok=True)
        _CACHE.write_text(json.dumps(d, indent=1))
    return x, out
