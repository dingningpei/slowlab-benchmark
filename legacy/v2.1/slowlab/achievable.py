"""The regret a design supports: the ceiling on the decision it can carry,
expressed in the currency of the main score.

Note that this is *not achievable by any agent*: computing it needs P_Theta and
the forward model, neither of which the interface exposes. Calling it
"achievable" would mislead a reader into thinking some agent could reach it.

    R*(D) = E_y[ min_x  sum_m w_m(y) ( f_m(x*_m) - f_m(x) ) ]

It reads exactly like eq:regret, in the same units of EUR/m^2/d; the only
difference is that the expectation is taken under "if the inference were Bayes
optimal". Realised regret therefore splits into two addends:

    R_actual  =  R*(D)              +  ( R_actual - R*(D) )
                 ceiling set by the design      lost by the inference

Both endpoints are computable without any reference strategy:
  * the empty design gives R*(none) = min_x E_theta[f(x*)-f(x)], which is exactly
    best_fixed's regret;
  * perfect identification gives R* = 0.

Why this rather than EIG_{x*}: EIG is a difference of entropies, and entropy is
systematically biased at finite M and does not converge over the range of M we
can afford (Appendix G). R* is an expectation of a mean, means converge far
faster, and the result does not have to be divided by H to be readable.

────────────────────────────────────────────────────────────────────────
2026-09-11: achievable_regret has a severe in-sample bias. Do not use it for
reporting. Use achievable_regret_cv.

The diagnosis: one set of atoms simultaneously (a) forms the posterior, (b)
chooses a recommendation from the candidate set, and (c) evaluates that
recommendation. Sharing atoms between (b) and (c) is in-sample selection.
Measured ESS was 1.1 out of 160 -- the posterior collapses onto essentially one
atom and the recommendation is that atom's own optimum in 98-100% of draws, which
of course scores perfectly in sample, because that atom really did generate y.
Evaluated on held-out atoms, c falls from +97% to -64%.

This is model misspecification rather than estimation noise: the true site is
never one of the M atoms, while the likelihood -- which carries observation noise
but no discretisation error -- is sharp enough to put all the weight on the
nearest one.

The fix is described in achievable_regret_cv's docstring.
────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import numpy as np


def candidate_set(atoms, n_rand=2000, rng=None):
    """Candidate recommendations. Must include each atom's true optimum, otherwise R* has a floor it cannot reach."""
    rng = rng or np.random.default_rng(0)
    d = atoms.xstar.shape[1]
    return np.vstack([atoms.xstar, rng.random((n_rand, d))])


def response_table(atoms, cand):
    """F[m, c] = atom m's response at candidate c. With a fixed atom set this is computed once."""
    return np.stack([w(cand) for w in atoms.worlds])          # (M, C)


def _sigma(atoms, pts, blocks, tau_block, tau_unit, tau_batch):
    n = len(pts)
    nb = int(np.max(blocks)) + 1
    Z = np.eye(nb)[np.asarray(blocks, int)]
    return (np.eye(n) * (atoms.s2_ref + tau_unit ** 2)
            + tau_block ** 2 * Z @ Z.T
            + tau_batch ** 2 * np.ones((n, n)))


def posterior_weights(atoms, pts, blocks, y, tau_block, tau_unit, tau_batch,
                      prior_w=None):
    """Fold a round's *actual observations* into the atom weights, so later rounds
    can condition on them.

    Without this, round r's design is scored against the prior: a design that
    correctly narrows in to confirm the optimum it already believes in would look
    uninformative, when that is exactly the right thing to do.
    """
    pts = np.atleast_2d(np.asarray(pts, float))
    if len(pts) == 0:
        return (np.full(atoms.M, 1.0 / atoms.M) if prior_w is None
                else np.asarray(prior_w, float))
    mu = atoms.mean_response(pts)
    S = _sigma(atoms, pts, blocks, tau_block, tau_unit, tau_batch)
    Si = np.linalg.inv(S + 1e-14 * np.eye(len(pts)))
    R = np.asarray(y, float)[None, :] - mu
    ll = -0.5 * np.einsum('mi,ij,mj->m', R, Si, R)
    w = np.exp(ll - ll.max())
    if prior_w is not None:
        w = w * np.asarray(prior_w, float)
    return w / w.sum()


def achievable_regret(atoms, pts, blocks, tau_block, tau_unit, tau_batch,
                      F=None, cand=None, best=None, n_outer=200, rng=None,
                      mu=None, prior_w=None) -> float:
    """R*(D) for one design, in the units of the objective.

    pts/blocks have the same structure as in eig_of_design and share the same
    Sigma(D) noise structure, so the allocation -- which treatment goes in which
    chamber -- enters this quantity too.
    """
    rng = rng or np.random.default_rng(0)
    if cand is None:
        cand = candidate_set(atoms, rng=np.random.default_rng(0))
    if F is None:
        F = response_table(atoms, cand)
    if best is None:
        best = np.array([w.oracle()[1] for w in atoms.worlds], float)   # (M,)

    pts = np.atleast_2d(np.asarray(pts, float))
    n = len(pts)
    prior = (np.full(atoms.M, 1.0 / atoms.M) if prior_w is None
             else np.asarray(prior_w, float))
    if n == 0:                                    # empty design: the best constant under current belief
        return float(np.min(prior @ (best[:, None] - F)))

    mu = atoms.mean_response(pts) if mu is None else mu                 # (M, n)
    nb = int(np.max(blocks)) + 1
    Z = np.eye(nb)[np.asarray(blocks, int)]
    S = (np.eye(n) * (atoms.s2_ref + tau_unit ** 2)
         + tau_block ** 2 * Z @ Z.T
         + tau_batch ** 2 * np.ones((n, n)))
    L = np.linalg.cholesky(S + 1e-14 * np.eye(n))
    Si = np.linalg.inv(S + 1e-14 * np.eye(n))

    # y is drawn from the predictive distribution of the *current* belief, not the prior
    m_n = rng.choice(atoms.M, size=n_outer, p=prior)
    Y = mu[m_n].T + L @ rng.standard_normal((n, n_outer))
    D = best[:, None] - F                         # (M, C) per-atom regret at each candidate
    out = np.empty(n_outer)
    for k in range(n_outer):
        Rr = Y[:, k][None, :] - mu
        ll = -0.5 * np.einsum('mi,ij,mj->m', Rr, Si, Rr)
        w = np.exp(ll - ll.max()) * prior; w /= w.sum()
        out[k] = np.min(w @ D)                    # posterior expected regret of the Bayes-optimal recommendation
    return float(out.mean())


# ────────────── Tempering, leave-one-out calibration, held-out evaluation ──────────────

TEMP_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)


def _design_cov(atoms, pts, blocks, tau_block, tau_unit, tau_batch):
    n = len(pts)
    Z = np.eye(int(np.max(blocks)) + 1)[np.asarray(blocks, int)]
    return (np.eye(n) * (atoms.s2_ref + tau_unit ** 2)
            + tau_block ** 2 * Z @ Z.T
            + tau_batch ** 2 * np.ones((n, n)))


def _weights(resid, Si, T):
    ll = -0.5 * np.einsum('mi,ij,mj->m', resid, Si, resid) / T
    w = np.exp(ll - ll.max())
    return w / w.sum()


def calibrate_temperature(atoms, pts, blocks, tau_block, tau_unit, tau_batch,
                          F=None, cand=None, best=None, grid=TEMP_GRID,
                          n_cal=150, rng=None, mu=None):
    """Choose the tempering temperature T for this design by leave-one-out.
    **Uses in-sample atoms only; the held-out set is never touched.**

    For each atom m: generate y from m, form the posterior over the *other* atoms
    with m excluded, pick the recommendation x-hat, and evaluate x-hat's regret on
    m. Average, then take the T on the grid that minimises it.

    Why this is not circular: R*(D) is defined as the regret a perfectly reasoning
    decision-maker still carries. Under a correctly specified model that is the
    Bayes rule; under a misspecified one -- the true site is not in the atom set --
    "perfect" can only mean the best within a family of inference rules. T is that
    family's parameter, and the calibration data are disjoint from the evaluation
    data, so choosing T does not move bias into the result.
    """
    rng = rng or np.random.default_rng(7)
    if cand is None:
        cand = candidate_set(atoms, rng=np.random.default_rng(0))
    if F is None:
        F = response_table(atoms, cand)
    if best is None:
        best = np.array([w.oracle()[1] for w in atoms.worlds], float)
    D = best[:, None] - F
    pts = np.atleast_2d(np.asarray(pts, float))
    n = len(pts)
    if n == 0:
        return 1.0                                   # an empty design has no likelihood to temper
    mu = atoms.mean_response(pts) if mu is None else mu
    S = _design_cov(atoms, pts, blocks, tau_block, tau_unit, tau_batch)
    L = np.linalg.cholesky(S + 1e-14 * np.eye(n))
    Si = np.linalg.inv(S + 1e-14 * np.eye(n))
    ms = rng.integers(0, atoms.M, n_cal)
    Y = mu[ms].T + L @ rng.standard_normal((n, n_cal))
    # The log-likelihood depends only on (k, m) and not on T, so it is computed
    # once and shared across temperatures.
    # Leave-one-out masks the atom itself with -inf, which avoids re-indexing D.
    R = Y.T[:, None, :] - mu[None, :, :]                  # (n_cal, M, n)
    LL = -0.5 * np.einsum('kmi,ij,kmj->km', R, Si, R)     # (n_cal, M)
    LL[np.arange(n_cal), ms] = -np.inf
    scores = []
    for T in grid:
        z = LL / T
        w = np.exp(z - z.max(axis=1, keepdims=True))
        w /= w.sum(axis=1, keepdims=True)
        pick = np.argmin(w @ D, axis=1)                   # (n_cal,)
        scores.append(float(D[ms, pick].mean()))
    return float(grid[int(np.argmin(scores))])


def achievable_regret_cv(atoms_in, atoms_out, pts, blocks,
                         tau_block, tau_unit, tau_batch,
                         cand=None, F_in=None, F_out=None, best_in=None,
                         best_out=None, T=None, n_outer=300, rng=None,
                         return_T=False, mu_in=None, mu_out=None):
    """R*(D) with the in-sample bias repaired. Everything reported uses this.

    Three differences from achievable_regret:

    1. **Tempering.** The likelihood is raised to the power 1/T, equivalent to
       inflating Sigma by a factor T -- acknowledging that the true site falls
       between atoms, so no atom explains y exactly. Untempered, ESS is about 1,
       the posterior collapses onto a single atom, and the recommendation is that
       atom's own optimum.
    2. **T is chosen by leave-one-out** (calibrate_temperature), using in-sample
       atoms only.
    3. **Evaluation on held-out atoms**: ground truth comes from atoms_out while
       the posterior sees only atoms_in, so what is reported is this inference
       rule's risk under the true distribution rather than the atom model's
       opinion of itself.

    Measured at M=320, 160 for selection and 160 for evaluation: untempered
    held-out c is +15% / -64% / -8% on Optimise and -22% / -75% / +20% on Screen;
    this function gives 32/8/16 and 20/5/22.
    """
    rng = rng or np.random.default_rng(11)
    if cand is None:
        cand = candidate_set(atoms_in, rng=np.random.default_rng(0))
    if F_in is None:
        F_in = response_table(atoms_in, cand)
    if F_out is None:
        F_out = response_table(atoms_out, cand)
    if best_in is None:
        best_in = np.array([w.oracle()[1] for w in atoms_in.worlds], float)
    if best_out is None:
        best_out = np.array([w.oracle()[1] for w in atoms_out.worlds], float)
    D_in = best_in[:, None] - F_in
    D_out = best_out[:, None] - F_out

    pts = np.atleast_2d(np.asarray(pts, float))
    n = len(pts)
    if n == 0:                                   # empty design: pick the constant in sample, evaluate held out
        p = np.full(atoms_in.M, 1.0 / atoms_in.M)
        v = float(D_out[:, int(np.argmin(p @ D_in))].mean())
        return (v, 1.0) if return_T else v

    # mu may be precomputed in bulk by the caller and sliced.
    # ManagedTomgro.__call__ is a 210-day loop vectorised over candidate points,
    # so its cost tracks the *number of calls*, not the number of points --
    # batching a whole task's design points into one call is three orders of
    # magnitude faster than calling per design.
    mu_in = atoms_in.mean_response(pts) if mu_in is None else mu_in
    mu_out = atoms_out.mean_response(pts) if mu_out is None else mu_out
    if T is None:
        T = calibrate_temperature(atoms_in, pts, blocks, tau_block, tau_unit,
                                  tau_batch, F=F_in, cand=cand, best=best_in,
                                  mu=mu_in)
    S = _design_cov(atoms_in, pts, blocks, tau_block, tau_unit, tau_batch)
    L = np.linalg.cholesky(S + 1e-14 * np.eye(n))
    Si = np.linalg.inv(S + 1e-14 * np.eye(n))
    m_out = rng.integers(0, atoms_out.M, n_outer)
    Y = mu_out[m_out].T + L @ rng.standard_normal((n, n_outer))
    # Same vectorisation as calibrate_temperature: the likelihood is computed for all (k, m) at once.
    R = Y.T[:, None, :] - mu_in[None, :, :]               # (n_outer, M_in, n)
    z = -0.5 * np.einsum('kmi,ij,kmj->km', R, Si, R) / T
    w = np.exp(z - z.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    pick = np.argmin(w @ D_in, axis=1)                    # (n_outer,)
    v = float(D_out[m_out, pick].mean())
    return (v, float(T)) if return_T else v


def split_atoms(atoms):
    """Split an atom set in half into (select, evaluate). The first half chooses the recommendation, the second is ground truth."""
    import copy
    h = atoms.M // 2

    def sub(idx):
        b = copy.copy(atoms)
        b.worlds = [atoms.worlds[i] for i in idx]
        b.xstar = atoms.xstar[idx]
        b.cell = atoms.cell[idx]
        return b
    return sub(np.arange(h)), sub(np.arange(h, atoms.M))


def achievable_regret_split(atoms_in, atoms_out, pts, blocks,
                            tau_block, tau_unit, tau_batch,
                            cand=None, F_in=None, F_out=None,
                            best_out=None, n_outer=200, rng=None):
    """An unbiased R*(D): in-sample atoms choose the recommendation, held-out atoms
    evaluate it.

    Using one set of atoms both to choose x and to evaluate it understates regret
    -- taking a minimum over 2000 candidates chases noise whenever the signal is
    weak. On Screen and Transfer that optimism reaches 76%, which are exactly the
    two tasks with the smallest spread. best_fixed avoids it with 40 fitting / 40
    held-out seeds, and this does the same:

      theta ~ held-out atom  ->  y | theta, D  ->  posterior over in-sample atoms  ->  x-hat  ->  b_theta - f_theta(x-hat)

    The choice of x-hat sees only in-sample atoms and the evaluation only held-out
    atoms, and the two do not overlap.
    """
    rng = rng or np.random.default_rng(0)
    if cand is None:
        cand = candidate_set(atoms_in, rng=np.random.default_rng(0))
    if F_in is None:
        F_in = response_table(atoms_in, cand)
    if F_out is None:
        F_out = response_table(atoms_out, cand)
    if best_out is None:
        best_out = np.array([w.oracle()[1] for w in atoms_out.worlds], float)

    best_in = np.array([w.oracle()[1] for w in atoms_in.worlds], float)
    D_in = best_in[:, None] - F_in                       # (M_in, C)
    D_out = best_out[:, None] - F_out                    # (M_out, C)

    pts = np.atleast_2d(np.asarray(pts, float))
    n = len(pts)
    if n == 0:                                           # empty design: the best constant
        prior = np.full(atoms_in.M, 1.0 / atoms_in.M)
        c = int(np.argmin(prior @ D_in))                 # chosen in sample
        return float(D_out[:, c].mean())                 # evaluated held out

    mu_in = atoms_in.mean_response(pts)
    mu_out = atoms_out.mean_response(pts)
    nb = int(np.max(blocks)) + 1
    Z = np.eye(nb)[np.asarray(blocks, int)]
    S = (np.eye(n) * (atoms_in.s2_ref + tau_unit ** 2)
         + tau_block ** 2 * Z @ Z.T
         + tau_batch ** 2 * np.ones((n, n)))
    L = np.linalg.cholesky(S + 1e-14 * np.eye(n))
    Si = np.linalg.inv(S + 1e-14 * np.eye(n))

    m_out = rng.integers(0, atoms_out.M, n_outer)        # ground truth from the held-out set
    Y = mu_out[m_out].T + L @ rng.standard_normal((n, n_outer))
    tot = np.empty(n_outer)
    for k in range(n_outer):
        Rr = Y[:, k][None, :] - mu_in                    # the posterior sees in-sample atoms only
        ll = -0.5 * np.einsum('mi,ij,mj->m', Rr, Si, Rr)
        w = np.exp(ll - ll.max()); w /= w.sum()
        c = int(np.argmin(w @ D_in))                     # the Bayes-optimal recommendation
        tot[k] = D_out[m_out[k], c]                      # evaluated on the truth that generated y
    return float(tot.mean())
