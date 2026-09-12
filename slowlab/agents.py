"""Non-LLM baselines. The three strategies differ in *how they design*, not only in where they sample."""
from __future__ import annotations
import numpy as np
import itertools
from .design import Design


# ---------------- Allocation helpers ----------------
def respect_hierarchy(P, task, facility, rng, screening: bool = False,
                      reserve: int = 0):
    """Quantise the *chamber-level* coordinates of candidate points to at most n_chambers levels.

    This is the direct consequence of the control hierarchy for design: four
    chambers means a chamber-level factor can carry at most four levels at once;
    loop-level factors are not constrained this way. Without this step the extra
    treatments have nowhere to go, which wastes parallelism and breaks the
    replication structure.
    """
    nc = facility.n_chambers
    for j, f in enumerate(task.factors):
        if f.control_level != "chamber":
            continue
        col = P[:, j]
        # Pick nc representative levels by quantile, then snap each point to the
        # nearest one.
        # The key choice: use nc//2 levels, not nc.
        # With nc levels of a chamber-level factor and exactly nc chambers, each
        # level occupies one chamber and the main plot has no replication -- the
        # loop-level replicates are then all crammed into one chamber and are
        # necessarily confounded with it. Leaving at least two chambers per
        # main-plot level is what makes the split-plot analysable.
        # screening=True: a screening round uses two levels with an unreplicated
        # main plot. The "at least two chambers per main-plot level" rule above is
        # right for a response-surface round, but applied to a screening round it
        # collapses Plackett-Burman rows: measured on the six-factor Screen, 10
        # rows became 6 treatments at rank 5/6 -- our reference for good practice
        # would itself have been rank-deficient, and rank deficiency in the models
        # is one of the paper's central criticisms. An unreplicated screening
        # design is standard: main-plot error is estimated from effect sparsity
        # (high-order interactions negligible), not from replication.
        n_lv = 2 if screening else max(2, nc // 2)
        levels = np.quantile(col, np.linspace(0.1, 0.9, min(n_lv, len(col))))
        P[:, j] = levels[np.argmin(np.abs(col[:, None] - levels[None, :]), axis=1)]

    # Still not enough: two levels per factor individually still allows up to
    # 2^k *combinations*, and there are only nc chambers. The real constraint is
    # that the number of distinct chamber-level combinations must not exceed the
    # number of chambers. Take the first nc distinct combinations as main-plot
    # treatments and snap the rest to the nearest one.
    ci = [j for j, f in enumerate(task.factors) if f.control_level == "chamber"]
    if ci:
        C = P[:, ci]
        uniq, first = np.unique(C, axis=0, return_index=True)
        # reserve: the caller will add further treatments afterwards (centre
        # points after a PB design are the example), and those consume
        # chamber-level combinations of their own. Without reserving, the
        # combination count exceeds the chamber count and the design is rejected
        # at submission with INFEASIBLE_GRANULARITY -- measured on Screen, 8 PB
        # combinations plus 1 centre point = 9, against 8 chambers.
        keep = C[np.sort(first)[:max(1, nc - reserve)]]
        d2 = ((C[:, None, :] - keep[None, :, :]) ** 2).sum(-1)
        P[:, ci] = keep[np.argmin(d2, axis=1)]
    return P


def chamber_safe_allocation(tids, reps, facility, rng, treatments, task, avail=None):
    """Within-block randomisation that respects the control hierarchy.

    A chamber-level factor can take only one value inside a chamber, so
    treatments are first grouped by their *chamber-level values* -- only members
    of a group may share a chamber -- and randomised within block after that.

    reps may be a scalar or a {tid: count} dict. A PB screening round needs the
    latter: screening points once each, centre points replicated.

    **All treatments must be passed in one call.** DoEAgent's screening round
    used to call this once per treatment, so the grouping saw a single treatment
    and the function could not know that a chamber was already held by *another*
    chamber-level setting -- it could only see whether units were free. The
    design was then rejected with INFEASIBLE_GRANULARITY. A function that groups
    by chamber-level value cannot maintain its constraint if it is fed one
    treatment at a time.
    """
    rep_of = (lambda t: int(reps)) if np.isscalar(reps) else (lambda t: int(reps[t]))
    ch_names = [f.name for f in task.factors if f.control_level == "chamber"]
    key = lambda t: tuple(round(treatments[t][n], 9) for n in ch_names)
    groups: dict = {}
    for t in tids:
        groups.setdefault(key(t), []).append(t)

    chambers = list(range(facility.n_chambers))
    rng.shuffle(chambers)
    ok = None if avail is None else set(avail)
    pool = {c: [u for u in facility.units
                if u.chamber == c and (ok is None or u.id in ok)] for c in chambers}
    chambers = [c for c in chambers if pool[c]]          # skip chambers wholly in recovery
    for c in pool:
        rng.shuffle(pool[c])

    # Split-plot allocation: each main-plot level (chamber-level value) gets
    # *several* chambers, and the treatments inside that group spread their
    # replicates across them -- which is what makes loop-level replication cross
    # chambers instead of confounding with them.
    n_groups = len(groups)
    per = max(1, len(chambers) // max(1, n_groups))
    alloc, ci = {}, 0
    for _, ts in groups.items():
        mine = chambers[ci:ci + per]
        ci += per
        if not mine:              # out of chambers: drop the group rather than
            continue              # squeeze it into an occupied one (that would
                                  # violate the hierarchy and fail the whole design)
        for t in ts:
            take = []
            for r in range(rep_of(t)):
                c = mine[r % len(mine)]                 # rotate across chambers
                if pool[c]:
                    take.append(pool[c].pop().id)
                else:
                    alt = [c2 for c2 in mine if pool[c2]]
                    if alt:
                        take.append(pool[alt[0]].pop().id)
            if take:
                alloc[t] = take
    return alloc


def spread_allocation(tids, reps, facility, rng, spread=True, avail=None):
    """Randomised complete block allocation.

    Both naive approaches are wrong:
      * pure randomisation -- with 2 replicates over 4 chambers there is about a
        27% chance that both replicates of a treatment land in the same chamber,
        so with 6 treatments at least one is almost certainly confounded;
      * taking chambers in order -- chambers are then spread, but treatment order
        becomes monotonically related to *loop index*, so the confounding just
        moves to another dimension. (That bug was caught by the
        allocation_balanced check.)

    The correct approach: treat chambers as blocks, put each treatment's
    replicates in *different* chambers, and draw at random both which chambers
    and which loop inside each.
    """
    if not spread:                       # kept deliberately: confound with chamber, for tests
        units = sorted(facility.units, key=lambda u: (u.chamber, u.loop))
        alloc, k = {}, 0
        for t in tids:
            alloc[t] = [units[(k + i) % len(units)].id for i in range(reps)]
            k += reps
        return alloc

    # Each treatment needs reps *distinct* chambers without exceeding capacity.
    # Naive greedy exhausts chamber diversity on the last few treatments: with 4
    # chambers, 6 treatments and 2 replicates a solution exists -- it is exactly
    # the six edges of K4 -- but greedy does not find it.
    # Choosing the chamber with the most remaining capacity, ties broken at
    # random, is enough.
    cap = {c: sum(1 for u in facility.units if u.chamber == c)
           for c in range(facility.n_chambers)}
    pool = {c: [u for u in facility.units if u.chamber == c]
            for c in range(facility.n_chambers)}
    for c in pool:
        rng.shuffle(pool[c])
    alloc = {}
    for t in tids:
        order = sorted(cap, key=lambda c: (-cap[c], rng.random()))
        chosen = []
        for c in order:
            if len(chosen) == reps:
                break
            if cap[c] > 0:
                chosen.append(pool[c].pop().id)
                cap[c] -= 1
        while len(chosen) < reps:                      # degraded path when capacity runs out
            cs = [c for c in cap if cap[c] > 0]
            if not cs:
                break
            c = cs[0]; chosen.append(pool[c].pop().id); cap[c] -= 1
        alloc[t] = chosen
    return alloc


# ---------------- Lightweight GP ----------------
class GP:
    def __init__(self, ls=0.28, noise=0.3):
        self.ls, self.noise = ls, noise

    def _k(self, A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / self.ls ** 2)

    def fit(self, X, y):
        self.X = X
        self.ym = y.mean() if len(y) else 0.0
        self.ys = float(np.std(y)) if len(y) > 1 and np.std(y) > 1e-12 else 1.0
        self.y = (y - self.ym) / self.ys        # standardise so noise and kernel amplitude share a scale
        K = self._k(X, X) + (self.noise ** 2) * np.eye(len(X))
        self.L = np.linalg.cholesky(K + 1e-8 * np.eye(len(X)))
        self.a = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))
        return self

    def predict(self, Xs):
        """Returns (mean, latent variance) in the original units."""
        Ks = self._k(Xs, self.X)
        mu = (Ks @ self.a) * self.ys + self.ym
        v = np.linalg.solve(self.L, Ks.T)
        var = np.clip(1.0 - (v ** 2).sum(0), 1e-9, None) * self.ys ** 2
        return mu, var

    def predict_obs(self, Xs):
        """Predictive distribution of a single *observation*, noise included, for reporting intervals."""
        mu, var = self.predict(Xs)
        return mu, np.sqrt(var + (self.noise * self.ys) ** 2)


def _gp_posterior_max(env, n_cand: int = 8000, seed: int = 7):
    """Posterior-mean maximiser of the GP -- the recommendation rule shared by DoE, GP-UCB and Transfer."""
    X, y = env.as_arrays()
    if not len(y):
        return np.full(env.task.d, 0.5)
    g = GP(noise=max(0.15, env.task.plant_cv)).fit(X, y)
    C = _cands(env.task.d, n_cand, np.random.default_rng(seed))
    return C[int(np.argmax(g.predict(C)[0]))]


def _best_so_far(env, agent=None):
    """The recommendation logged at the end of each round.

    **This must use the same rule the agent submits at the end of the episode.**
    Otherwise the trajectory and the endpoint are two different quantities and
    the claim that we read the same recommendation every round is false. This
    used to take the best observed point for DoEAgent while the agent finally
    submitted the GP posterior maximum; on Optimise seed 0 the two differed by
    47%.
    """
    if hasattr(agent, "recommend"):
        return agent.recommend(env)
    X, y = env.as_arrays()
    if not len(y):
        return np.full(env.task.d, 0.5)
    return X[int(np.argmax(y))]


def _cands(d, n, rng):
    return rng.random((n, d))


# ---------------- Baseline 1: random spread ----------------

Z95 = 1.959964


def _wide_prior(n, y_seen):
    """Default interval before any data: centred on what has been seen, if anything, and very wide."""
    c = float(np.mean(y_seen)) if len(y_seen) else 0.0
    h = 4.0 * (float(np.std(y_seen)) if len(y_seen) > 1 else 0.25)
    return {"centre": c, "half": h}


def _preds_empirical(tids, y_seen):
    """Model-free forecast (used by random search): mean of all observations so
    far plus/minus 1.96 sd. It gives every treatment the same interval, which is
    the honest consequence of having no model."""
    if len(y_seen) > 1:
        c, h = float(np.mean(y_seen)), Z95 * float(np.std(y_seen, ddof=1))
    else:
        w = _wide_prior(0, y_seen); c, h = w["centre"], w["half"]
    return {t: (c - h, c, c + h) for t in tids}


def _preds_gp(tids, P, X_seen, y_seen, noise):
    """GP posterior predictive interval -- BO has one already, no extra assumption needed."""
    if len(y_seen) < 3:
        return _preds_empirical(tids, y_seen)
    g = GP(noise=noise).fit(X_seen, y_seen)
    mu, sd = g.predict_obs(np.asarray(P, float))
    return {t: (float(m - Z95 * s_), float(m), float(m + Z95 * s_))
            for t, m, s_ in zip(tids, mu, sd)}


def _preds_ols(tids, P, X_seen, y_seen):
    """Classical DoE forecast: first-order linear model plus a residual-s prediction interval."""
    if len(y_seen) < len(np.atleast_2d(P)[0]) + 2:
        return _preds_empirical(tids, y_seen)
    A = np.column_stack([np.ones(len(X_seen)), X_seen])
    coef, *_ = np.linalg.lstsq(A, y_seen, rcond=None)
    resid = y_seen - A @ coef
    dof = max(1, len(y_seen) - A.shape[1])
    s = float(np.sqrt(np.sum(resid ** 2) / dof))
    B = np.column_stack([np.ones(len(P)), np.asarray(P, float)])
    mu = B @ coef
    return {t: (float(m - Z95 * s), float(m), float(m + Z95 * s))
            for t, m in zip(tids, mu)}



def probe_intervals(env, kind, noise=0.3):
    """95% intervals on the common probe set at the end of an episode. Each method uses its own natural rule."""
    P = env.probe_set()
    X, y = env.as_arrays()
    if kind == "empirical" or len(y) < 3:
        d = _preds_empirical([f"p{i}" for i in range(len(P))], y)
        return np.array([d[f"p{i}"] for i in range(len(P))], float)
    if kind == "ols":
        d = _preds_ols([f"p{i}" for i in range(len(P))], P, X, y)
        return np.array([d[f"p{i}"] for i in range(len(P))], float)
    g = GP(noise=noise).fit(X, y)
    mu, sd = g.predict_obs(P)
    return np.column_stack([mu - Z95 * sd, mu, mu + Z95 * sd])


class RandomAgent:
    def recommend(self, env):
        X, y = env.as_arrays()
        return X[int(np.argmax(y))] if len(y) else np.full(env.task.d, 0.5)

    """Random points, spread wide, one replicate each, no randomisation seed
    declared. A deliberate model of the naive "just try a few more" instinct."""
    name = "random_spread"

    def run(self, env):
        rng = np.random.default_rng(env.seed + 991)
        t = env.task
        while env.rounds_left > 0:
            q = t.units_per_round
            P = respect_hierarchy(rng.random((q, t.d)), t, env.facility, rng)
            tids = [f"t{i}" for i in range(q)]
            trt = {tid: {f.name: float(P[i][j]) for j, f in enumerate(t.factors)}
                   for i, tid in enumerate(tids)}
            alloc = chamber_safe_allocation(tids, 1, env.facility, rng, trt, t, avail=env.available_units())
            _, y_seen = env.as_arrays()
            r = env.submit_design(Design(trt, alloc, randomization_seed=None,
                                         question="random spread",
                                         predictions=_preds_empirical(tids, y_seen)))
            if isinstance(r, int):
                env.advance()
                env.interim_recommendation(_best_so_far(env, self))
            else:
                break
        X, y = env.as_arrays()
        self.probes = probe_intervals(env, "empirical")
        return X[int(np.argmax(y))] if len(y) else np.full(t.d, 0.5)


# ---------------- Baseline 2: the classical DoE pipeline ----------------
def _pb12():
    """Plackett-Burman 12-run design (screens up to 11 factors)."""
    g = [1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1]
    rows = [g[-i:] + g[:-i] for i in range(11)]
    rows.append([-1] * 11)
    return np.array(rows, float)


class DoEAgent:
    def recommend(self, env):
        return _gp_posterior_max(env)

    """What an agronomist would do: screening design, then response surface.
    Always replicated, always across chambers, always with a declared seed."""
    name = "classical_doe"

    def run(self, env):
        t = env.task
        rng = np.random.default_rng(env.seed + 7)
        d, q = t.d, t.units_per_round

        # ---- Round 1: screening / factorial design ----
        # In high dimension use Plackett-Burman (unreplicated) plus replicated
        # centre points to estimate pure error, which is standard practice; in low
        # dimension use a full factorial plus centre points.
        if d >= 4:
            n_screen = min(12, q - 2)
            M = _pb12()[:n_screen, :d]
            P = (M + 1) / 2 * 0.8 + 0.1
            P = np.vstack([P, [[0.5] * d] * 2])       # 2 centre points -> pure_error_df >= 1
            reps_plan = [1] * n_screen + [2]          # centre point replicated twice
            best_x = self._round_mixed(env, P[:n_screen], np.array([[0.5] * d]), rng)
        else:
            n_pts = max(2, q // 2)
            lv = list(itertools.product([0.15, 0.85], repeat=d))
            P = np.array(lv + [tuple([0.5] * d)] * max(0, n_pts - len(lv)))[:n_pts]
            best_x = self._round(env, np.clip(P, 0, 1), 2, rng)

        # ---- Later rounds: response surface around the best point, still replicated ----
        guard = 0
        while env.rounds_left > 0:
            guard += 1
            if guard > env.task.n_rounds + 2:      # backstop when submissions keep being rejected
                break
            X, y = env.as_arrays()
            if len(y):
                best_x = X[int(np.argmax(y))]
            n_pts = max(2, q // 2)
            P = np.clip(best_x + rng.normal(0, 0.12, (n_pts, d)), 0, 1)
            P[0] = best_x                            # keep the current best as an anchor
            best_x = self._round(env, P, 2, rng)

        self.probes = probe_intervals(env, "gp", noise=max(0.15, t.plant_cv))
        return self.recommend(env)

    def _round_mixed(self, env, P_single, P_center, rng):
        # reserve=1: the centre point needs a chamber-level combination of its own.
        P_single = respect_hierarchy(P_single, env.task, env.facility, rng,
                                     screening=True, reserve=1)
        """Screening points once each plus replicated centre points -- what a standard PB design actually looks like."""
        t = env.task
        n_left = t.units_per_round - len(P_single)
        n_center = max(2, n_left)
        pts = list(P_single) + list(P_center)
        reps = [1] * len(P_single) + [n_center]
        tids = [f"t{i}" for i in range(len(pts))]
        trt = {tid: {f.name: float(pts[i][j]) for j, f in enumerate(t.factors)}
               for i, tid in enumerate(tids)}
        # A past bug: this called chamber_safe_allocation once per treatment.
        # Two consequences:
        #   * each call was handed the same env.available_units() (the
        #     environment's free units change only on submission), so one unit
        #     was given to several treatments -> UNIT_CONFLICT;
        #   * more fundamentally, the grouping saw a single treatment and could
        #     not know a chamber was already held by *another* chamber-level
        #     setting, only whether units were free -> INFEASIBLE_GRANULARITY.
        # The effect was that classical_doe never once completed a screening
        # round on any task with d >= 4: round 1 was rejected and what it actually
        # submitted were the later response-surface designs, with every value
        # bunched near 0.5. That is where the paper's "good practice" reference
        # numbers came from.
        # The fix: pass all treatments in one call, with replication as a
        # per-treatment dict.
        alloc = chamber_safe_allocation(tids, dict(zip(tids, reps)), env.facility,
                                        rng, trt, t, avail=env.available_units())
        Xs, ys = env.as_arrays()
        r = env.submit_design(Design(trt, alloc, randomization_seed=int(rng.integers(1e6)),
                                     question="PB screening + center replicates",
                                     predictions=_preds_ols(tids, pts, Xs, ys)))
        if isinstance(r, int):
            env.advance()
            env.interim_recommendation(_best_so_far(env, self))
        X, y = env.as_arrays()
        self.probes = probe_intervals(env, "ols")
        return X[int(np.argmax(y))] if len(y) else np.full(t.d, 0.5)

    def _round(self, env, P, reps, rng):
        P = respect_hierarchy(P, env.task, env.facility, rng)
        tids = [f"t{i}" for i in range(len(P))]
        trt = {tid: {f.name: float(P[i][j]) for j, f in enumerate(env.task.factors)}
               for i, tid in enumerate(tids)}
        alloc = chamber_safe_allocation(tids, reps, env.facility, rng, trt, env.task, avail=env.available_units())
        Xs, ys = env.as_arrays()
        r = env.submit_design(Design(trt, alloc, randomization_seed=int(rng.integers(1e6)),
                                     question="screening / RSM",
                                     predictions=_preds_ols(tids, P, Xs, ys)))
        if isinstance(r, int):
            env.advance()
            env.interim_recommendation(_best_so_far(env, self))
        else:
            self._last_rejected = True
        X, y = env.as_arrays()
        self.probes = probe_intervals(env, "ols")
        return X[int(np.argmax(y))] if len(y) else np.full(env.task.d, 0.5)


# ---------------- Baseline 3: GP-UCB (two allocation styles) ----------------
class GPUCBAgent:
    def recommend(self, env):
        return _gp_posterior_max(env)

    def __init__(self, reps: int = 1, beta: float = 1.8):
        self.reps, self.beta = reps, beta
        self.name = f"gp_ucb_rep{reps}"

    def run(self, env):
        t = env.task
        rng = np.random.default_rng(env.seed + 13)
        d, q = t.d, t.units_per_round
        while env.rounds_left > 0:
            n_pts = max(1, q // self.reps)
            X, y = env.as_arrays()
            if len(y) < 2:
                P = rng.random((n_pts, d))
            else:
                g = GP(noise=max(0.15, t.plant_cv)).fit(X, y)
                C = _cands(d, 2500, rng)
                mu, var = g.predict(C)
                sc = mu + self.beta * np.sqrt(var)
                order = np.argsort(-sc)
                P, chosen = [], []
                for i in order:                        # greedy with a minimum spacing, for diversity
                    if all(np.linalg.norm(C[i] - c) > 0.05 for c in chosen):
                        chosen.append(C[i]); P.append(C[i])
                    if len(P) == n_pts:
                        break
                P = np.array(P) if P else rng.random((n_pts, d))
            P = respect_hierarchy(P, t, env.facility, rng)
            tids = [f"t{i}" for i in range(len(P))]
            trt = {tid: {f.name: float(P[i][j]) for j, f in enumerate(t.factors)}
                   for i, tid in enumerate(tids)}
            alloc = chamber_safe_allocation(tids, self.reps, env.facility, rng, trt, t, avail=env.available_units())
            Xs, ys = env.as_arrays()
            r = env.submit_design(Design(trt, alloc,
                                         randomization_seed=int(rng.integers(1e6)),
                                         question="GP-UCB batch",
                                         predictions=_preds_gp(tids, P, Xs, ys,
                                                               max(0.15, t.plant_cv))))
            if isinstance(r, int):
                env.advance()
                env.interim_recommendation(_best_so_far(env, self))
            else:
                break
        self.probes = probe_intervals(env, "gp", noise=max(0.15, t.plant_cv))
        return self.recommend(env)


AGENTS = {
    "random_spread": RandomAgent,
    "classical_doe": DoEAgent,
    "gp_ucb_rep1": lambda: GPUCBAgent(reps=1),
    "gp_ucb_rep2": lambda: GPUCBAgent(reps=2),
}


# ---------------- T4: the controlled pair for economic transfer ----------------
# The two agents are identical -- same surrogate, same acquisition function,
# same budget, same observations. The only difference is whether each models
# *profit itself* or *revenue and cost separately*.
class TransferAgent(GPUCBAgent):
    """The profit-only / component-wise modelling choice.

    **The design loop is exactly GPUCBAgent's** -- inherited, not rewritten. It
    used to have its own loop that took only n_chambers treatments per round
    (written for Transfer's four chamber-level factors) and was then applied
    unconditionally to all four tasks: on Sanity a budget of 8 units per round
    produced only 4 distinct treatments, measured regret 0.043 against
    GPUCBAgent's 0.016, a factor of 2.7. That was not the price of a modelling
    choice, it was one task's logic used in the wrong place.

    Inheriting makes this pair share the design loop with the GP-UCB reference
    and differ from each other only in the last step -- which is exactly what the
    paper claims about the pair: same surrogate, same acquisition function, same
    budget, same observations, differing only in whether they modelled profit or
    modelled revenue and the cost streams separately.
    """

    def __init__(self, mode: str = "profit", beta: float = 1.8, reps: int = 1):
        assert mode in ("profit", "components")
        super().__init__(reps=reps, beta=beta)
        self.mode = mode
        self.name = f"gp_ucb_{mode}"

    def run(self, env):
        t = env.task
        d = t.d
        noise = max(0.15, t.plant_cv)
        best = super().run(env)                 # same design loop, same recommendation

        X, y = env.as_arrays()
        if not len(y):
            self.x_transfer = np.full(d, 0.5)
            return np.full(d, 0.5)
        C = _cands(d, 8000, np.random.default_rng(7))

        # Only a task that *declares* a price shock has a transfer question. This
        # used to be called unconditionally, so float(None) crashed outright on a
        # task like Optimise.
        if getattr(env.task, "transfer_energy_shock", None) is None:
            self.x_transfer = best
            return best

        q_ = env.transfer_query()
        if self.mode == "profit":
            # It learned only this profit surface. When prices move it has no way
            # to re-solve and can only reuse its original recommendation -- which
            # is exactly the price of having memorised the surface.
            self.x_transfer = best
        else:
            # Learn revenue rate and cost rate separately, then recompose the
            # objective at the new prices. No new experiments are run.
            Xo = np.array([[env._designs[o.design_id].treatments[o.treatment][f.name]
                            for f in t.factors] for o in env.observations()])
            # Energy cost and other cost must be learned *separately*: the shock
            # multiplies energy alone, and a merged cost_rate cannot be recomposed
            # at the new prices.
            rev = np.array([o.rev_rate for o in env.observations()])
            e_c = np.array([o.energy_cost_rate for o in env.observations()])
            o_c = np.array([o.other_cost_rate for o in env.observations()])
            gr = GP(noise=noise).fit(Xo, rev)
            ge = GP(noise=0.02).fit(Xo, e_c)
            go = GP(noise=0.02).fit(Xo, o_c)
            mu_r, _ = gr.predict(C)
            mu_e, _ = ge.predict(C)
            mu_o, _ = go.predict(C)
            s_new = q_["energy_shock"] / max(q_["old_energy_shock"], 1e-9)
            self.x_transfer = C[int(np.argmax(mu_r - (s_new * mu_e + mu_o)))]
        return best


AGENTS["gp_ucb_profit"] = lambda: TransferAgent("profit")
AGENTS["gp_ucb_components"] = lambda: TransferAgent("components")

# Removed: LiteraturePriorAgent / LITERATURE_VALUES.
# Those six "textbook values" were written from the author's memory with no
# source, and their purpose was to contrast "is the agent experimenting or
# reciting" -- using something recited as the control for recitation does not
# stand up. Their two functions were replaced by:
#   * the ceiling on "what you get without experimenting" ->
#     slowlab.fixed_reference.best_fixed, which is computable, uncited, evaluated
#     on held-out instances, and strictly dominates any hand-picked constant;
#   * "what recall alone gets you" -> ask the model under test directly: with no
#     data, what do you recommend? That is a per-model measurement (see
#     llm.zero_shot_recommendation), not a line we assume on its behalf.
# PriorAnchoredAgent went with it: it shrank towards those values by an
# arbitrary amount and never once won on any of the four tasks.
