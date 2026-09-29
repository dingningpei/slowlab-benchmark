"""Estimator for EIG_{x*} -- equations (3) and (4) of the paper.

  1. Discretise P_Theta into M atoms theta_m with weights pi_m = 1/M.
  2. Get each atom's true optimum x*_{theta_m} from the oracle -- not by argmax
     over a coarse grid. In 7 dimensions, 4096 Sobol points collapse the optima
     of different atoms into the same cell and crush the prior entropy to
     e^H = 1.3, which is an artefact of the discretisation.
  3. Bin those M optima at nbins per dimension to get the support of p(x*).
  4. An observation only reweights the atoms: w_m(y) proportional to
     pi_m p(y | theta_m, D), with the likelihood evaluated *exactly* at the
     design points, never interpolated.
  5. EIG = H[Φ[π]] − (1/N) Σ_n H[Φ[w(y_n)]]

The inner expectation is an exact finite sum rather than a Monte Carlo estimate,
so the O(1/M) bias of a nested estimator does not arise.

The likelihood is a moment-matched Gaussian:
    Σ(D) = diag(s²(x_u) + τ_γ²) + τ_β² Z Zᵀ + τ_ε² 1 1ᵀ
Z is the unit-to-block membership matrix -- this is where a design's *allocation*
enters the measure.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


def _H(p):
    p = np.asarray(p, float)
    p = p[p > 1e-300]
    return float(-(p * np.log(p)).sum())


@dataclass
class AtomSet:
    """M site atoms: each one's ground-truth backend, true optimum, and the bin that optimum falls in."""
    worlds: list                      # M ManagedTomgro instances
    xstar: np.ndarray                 # (M, d) true optima in normalised coordinates
    cell: np.ndarray                  # (M,) integer bin labels
    n_cells: int
    nbins: int
    task_name: str
    s2_ref: float = 0.0               # representative value of the per-plant pushed-forward variance

    @property
    def M(self): return len(self.worlds)

    def prior_hist(self):
        h = np.bincount(self.cell, minlength=self.n_cells).astype(float)
        return h / h.sum()

    def prior_entropy(self):
        return _H(self.prior_hist())

    def mean_response(self, pts: np.ndarray) -> np.ndarray:
        """(M, n): each atom's mean response at the design points, evaluated exactly.

        Deduplicate by *distinct treatment* before evaluating. This is not a
        micro-optimisation: the paper itself measures that most submitted designs
        use two distinct treatments replicated across the whole facility, and
        without deduplication TOMGRO runs once per unit -- which accounted for 99%
        of the R* estimator's time. The results are bit-identical.
        """
        pts = np.atleast_2d(np.asarray(pts, float))
        if len(pts) == 0:
            return np.zeros((self.M, 0))
        uniq, inv = np.unique(pts, axis=0, return_inverse=True)
        if len(uniq) == len(pts):
            return np.stack([w(pts) for w in self.worlds])
        return np.stack([w(uniq) for w in self.worlds])[:, inv]


def build_atoms(task, M=400, nbins=12, n_plant=24, seed0=10_000,
                plants_per_unit=None) -> AtomSet:
    """plants_per_unit: plants on one experimental unit. The s^2 in the likelihood
    must be the *unit-level* variance, i.e. the variance after averaging those N
    plants, not the single-plant variance -- otherwise EIG comes out
    systematically low by sqrt(N). (Same root cause as the "one unit is one
    plant" error in env.)"""
    from .world import ManagedTomgro
    ws, xs = [], []
    for m in range(M):
        w = ManagedTomgro(seed=seed0 + m, factors=task.factors,
                          cycle_days=task.cycle_days)
        x, _ = w.oracle()
        ws.append(w); xs.append(x)
    X = np.asarray(xs, float)
    b = np.clip((X * nbins).astype(int), 0, nbins - 1)
    lab = np.zeros(len(X), dtype=np.int64)
    for j in range(X.shape[1]):
        lab = lab * nbins + b[:, j]
    uniq, cell = np.unique(lab, return_inverse=True)
    rng = np.random.default_rng(1)
    npu = int(plants_per_unit or 39)          # default: mid-range density x a 12 m^2 loop
    probe = np.full((1, X.shape[1]), 0.5)
    draws = []
    for _ in range(n_plant):                  # each draw is one *unit*: the mean of npu plants
        pp = ws[0].sample_plant_params(npu, rng, task.plant_cv)
        draws.append(float(np.mean([
            ws[0](probe, plant_params={k: (v[i] if np.ndim(v) else v)
                                       for k, v in pp.items()})[0]
            for i in range(npu)])))
    draws = np.asarray(draws, float)
    return AtomSet(worlds=ws, xstar=X, cell=cell, n_cells=len(uniq),
                   nbins=nbins, task_name=task.name, s2_ref=float(draws.var(ddof=1)))


def eig_of_design(atoms: AtomSet, pts, blocks, tau_block, tau_unit, tau_batch,
                  n_outer=600, rng=None, mu=None) -> float:
    """EIG_{x*} of one design, in nats."""
    rng = rng or np.random.default_rng(0)
    pts = np.atleast_2d(np.asarray(pts, float))
    n = len(pts)
    if n == 0:
        return 0.0
    mu = atoms.mean_response(pts) if mu is None else mu       # (M, n)
    nb = int(np.max(blocks)) + 1
    Z = np.eye(nb)[np.asarray(blocks, int)]
    S = (np.eye(n) * (atoms.s2_ref + tau_unit ** 2)
         + tau_block ** 2 * Z @ Z.T
         + tau_batch ** 2 * np.ones((n, n)))
    L = np.linalg.cholesky(S + 1e-14 * np.eye(n))
    Si = np.linalg.inv(S + 1e-14 * np.eye(n))
    H0 = atoms.prior_entropy()
    m_n = rng.integers(0, atoms.M, n_outer)
    Y = mu[m_n].T + L @ rng.standard_normal((n, n_outer))
    Hs = np.empty(n_outer)
    for k in range(n_outer):
        Rr = Y[:, k][None, :] - mu
        ll = -0.5 * np.einsum('mi,ij,mj->m', Rr, Si, Rr)
        w = np.exp(ll - ll.max()); w /= w.sum()
        h = np.bincount(atoms.cell, weights=w, minlength=atoms.n_cells)
        Hs[k] = _H(h / h.sum())
    return H0 - float(Hs.mean())


# ── Information regret (following BoxingGym's EIRegret) ─────────────────
# They compare the EIG of the agent's chosen experiment against the best of 100
# random experiments and take the difference. The anchor is the best score random
# search achieves, not the theoretical optimum and not a BOED maximiser.
# We reuse the construction, but note an asymmetry: one of their experiments is a
# single design point, which random sampling covers well; one of ours is a whole
# allocation of 12 units over blocks, a far larger space, so best-of-K is a *much
# weaker* upper reference. The paper has to say so.


def random_feasible_design(task, facility, rng, n_units=None):
    """Draw a random *feasible* design: respects the hierarchy, the width cap, and no repeated units."""
    n = n_units or task.units_per_round
    units = list(facility.units)[:n]
    d = len(task.factors)
    lvl = [f.control_level for f in task.factors]
    by_ch = {}
    for u in units:
        by_ch.setdefault(u.chamber, []).append(u)
    ch_vals = {c: rng.random(d) for c in by_ch}          # one set of chamber-level values per chamber
    pts, blk = [], []
    for c, us in by_ch.items():
        for u in us:
            v = ch_vals[c].copy()
            for j, L in enumerate(lvl):
                if L != "chamber":
                    v[j] = rng.random()                   # loop-level factors are free per unit
            pts.append(v); blk.append(c)
    return np.asarray(pts, float), np.asarray(blk, int)


def reference_eig(atoms, task, facility, taus, K=100, rng=None, **kw) -> float:
    """Largest EIG among K random feasible designs.

    **No longer used for scoring.** We first followed BoxingGym and reported
    information as the gap to best-of-K, then moved to a self-contained EIG/H:
    under the gap form the choice of K changes the score (K=12 produced negative
    values), and a reader should not have to run a batch of reference designs to
    interpret a number. It is kept because the appendix explains that trade-off,
    and this is the reproducible basis for that claim.
    """
    rng = rng or np.random.default_rng(0)
    best = -np.inf
    for _ in range(K):
        pts, blk = random_feasible_design(task, facility, rng)
        best = max(best, eig_of_design(atoms, pts, blk, *taus, rng=rng, **kw))
    return float(best)
