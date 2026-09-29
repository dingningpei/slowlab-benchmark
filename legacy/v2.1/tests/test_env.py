import numpy as np
import pytest

from slowlab.env import SlowLabEnv
from slowlab.tasks import TASKS, Task
from slowlab.design import Design, RejectCode
from slowlab.factors import FactorSpec
from slowlab.validity import score_validity

def mk(task="T3", seed=0):
    return SlowLabEnv(TASKS[task], seed=seed)


def simple_design(env, n=4, reps=2, seed=1, spread=True):
    """Build a legal design.

    It must respect the control hierarchy: a chamber-level factor takes one value
    inside a chamber, so each treatment gets its own chamber (n <= number of
    chambers) with replicates on different loops of that chamber.
    spread=False deliberately crowds the replicates together to trip the
    confounding check.
    """
    t = env.task
    n = min(n, env.facility.n_chambers)
    tids = [f"t{i}" for i in range(n)]
    trt = {tid: {f.name: 0.2 + 0.15 * i for f in t.factors} for i, tid in enumerate(tids)}
    alloc = {}
    for i, tid in enumerate(tids):
        us = [u.id for u in env.facility.units if u.chamber == i]
        alloc[tid] = us[:reps]
    return Design(trt, alloc, randomization_seed=seed)


# ---------- Rejection semantics ----------
def test_unknown_unit():
    env = mk(); d = simple_design(env); d.allocation["t0"] = ["c99l99"]
    assert env.validate_design(d).code == RejectCode.UNKNOWN_UNIT

def test_unit_conflict():
    env = mk(); d = simple_design(env)
    d.allocation["t1"] = list(d.allocation["t0"])
    assert env.validate_design(d).code == RejectCode.UNIT_CONFLICT

def test_budget():
    env = mk('T2')   # T2 has 24 units, enough to build an over-budget submission
    units = [u.id for u in env.facility.units]
    n = env.task.units_per_round + 2                 # over this round's cap
    d = Design({f"t{i}": {f.name: 0.5 for f in env.task.factors} for i in range(n)},
               {f"t{i}": [units[i]] for i in range(n)}, randomization_seed=1)
    assert env.validate_design(d).code == RejectCode.BUDGET_EXCEEDED

def test_factor_range():
    env = mk(); d = simple_design(env)
    d.treatments["t0"][env.task.factors[0].name] = 1.7
    assert env.validate_design(d).code == RejectCode.FACTOR_OUT_OF_RANGE

def test_granularity():
    """A chamber-level factor taking different values inside one chamber must be rejected.
    day_temp is chamber-level (lamps and heating serve a chamber), density is loop-level."""
    env = mk("T2")                       # T2 has both chamber-level and loop-level factors
    u = [x.id for x in env.facility.units]
    c0 = [x.id for x in env.facility.units if x.chamber == 0][:2]
    c1 = [x.id for x in env.facility.units if x.chamber == 1][:1]
    base = {f.name: 0.5 for f in env.task.factors}
    a = dict(base); a["day_temp"] = 0.2
    b = dict(base); b["day_temp"] = 0.8
    bad = Design({"a": a, "b": b}, {"a": [c0[0]], "b": [c0[1]]}, randomization_seed=1)
    assert env.validate_design(bad).code == RejectCode.INFEASIBLE_GRANULARITY
    ok = Design({"a": a, "b": b}, {"a": [c0[0]], "b": [c1[0]]}, randomization_seed=1)
    assert env.validate_design(ok).code == RejectCode.OK

def test_rejection_costs_no_budget():
    env = mk('T2')
    units = [u.id for u in env.facility.units]
    n = env.task.units_per_round + 2
    bad = Design({f"t{i}": {f.name: 0.5 for f in env.task.factors} for i in range(n)},
                 {f"t{i}": [units[i]] for i in range(n)}, randomization_seed=1)
    env.submit_design(bad)
    assert env.rounds_left == env.task.n_rounds     # no round was consumed
    assert env._rejections == [RejectCode.BUDGET_EXCEEDED]


# ---------- Validity criteria ----------
def test_validity_replication():
    env = mk(); ok = simple_design(env, reps=2); bad = simple_design(env, reps=1)
    assert score_validity(ok, env.facility, env.task).checks["replication"]
    assert not score_validity(bad, env.facility, env.task).checks["replication"]

def test_validity_randomization():
    env = mk(); d = simple_design(env); d.randomization_seed = None
    assert not score_validity(d, env.facility, env.task).passed

def test_validity_confounding():
    """A loop-level factor whose replicates all sit in one chamber is confounded
    with the chamber effect.

    This applies to *loop-level* factors only: if every factor were chamber-level,
    a treatment would be a chamber setting and they could not be separated anyway,
    so the check does not apply. T3 deliberately mixes the two levels.
    """
    env = mk("T3")
    assert any(f.control_level == "loop" for f in env.task.factors)
    lo = next(f.name for f in env.task.factors if f.control_level == "loop")
    c0 = [u.id for u in env.facility.units if u.chamber == 0]
    c1 = [u.id for u in env.facility.units if u.chamber == 1]
    base = {f.name: 0.5 for f in env.task.factors}
    a = dict(base); a[lo] = 0.2
    b = dict(base); b[lo] = 0.8
    bad = Design({"a": a, "b": b}, {"a": c0[:2], "b": c1[:2]}, randomization_seed=1)
    good = Design({"a": a, "b": b}, {"a": [c0[0], c1[0]], "b": [c0[1], c1[1]]},
                  randomization_seed=1)
    assert not score_validity(bad, env.facility, env.task).checks["no_chamber_confounding"]
    assert score_validity(good, env.facility, env.task).checks["no_chamber_confounding"]

def test_no_feedback_before_advance():
    env = mk(); d = simple_design(env)
    n = len(d.unit_ids)
    env.submit_design(d)
    assert len(env.observations()) == 0          # nothing visible after submit and before advance
    env.advance()
    assert len(env.observations()) == n

def test_rounds_exhaust():
    env = mk()
    for _ in range(env.task.n_rounds):
        r = env.submit_design(simple_design(env))
        assert isinstance(r, int), r
        env.advance()
    assert env.rounds_left == 0
    assert env.submit_design(simple_design(env)).code == RejectCode.BUDGET_EXCEEDED


# ---------- The generative model ----------
def test_regret_nonnegative():
    env = mk()
    r = env.submit_recommendation(np.full(env.task.d, 0.5))
    assert r.simple_regret >= -1e-9


# ---------- The substantive randomisation test (allocation_balanced) ----------
def _six_by_two(env, alloc):
    """Six treatments with two replicates. Only the *loop-level* factor varies and
    the chamber-level one is held fixed, otherwise the design is rejected by the
    hierarchy first and the randomisation itself is never tested."""
    lv = np.linspace(0.05, 0.95, 6)
    lo = next(f.name for f in env.task.factors if f.control_level == "loop")
    base = {f.name: 0.5 for f in env.task.factors}
    return Design({f"t{i}": {**base, lo: float(lv[i])} for i in range(6)},
                  alloc, randomization_seed=7,
                  predictions={f"t{i}": (-1.0, 0.0, 1.0) for i in range(6)})


def test_balance_catches_systematic_by_chamber():
    """Treatments laid out in chamber order must fail, even with a seed declared."""
    env = mk("T3")
    by_c = [u.id for u in sorted(env.facility.units, key=lambda u: (u.chamber, u.loop))]
    d = _six_by_two(env, {f"t{i}": [by_c[2*i], by_c[2*i+1]] for i in range(6)})
    assert not score_validity(d, env.facility, env.task).checks["allocation_balanced"]


def test_balance_catches_systematic_by_loop():
    """Laid out in loop order: chambers are spread, but the confounding just moves
    to another dimension. This was exactly spread_allocation's original bug."""
    env = mk("T3")
    by_l = [u.id for u in sorted(env.facility.units, key=lambda u: (u.loop, u.chamber))]
    d = _six_by_two(env, {f"t{i}": [by_l[2*i], by_l[2*i+1]] for i in range(6)})
    assert not score_validity(d, env.facility, env.task).checks["allocation_balanced"]


def test_blocked_randomisation_passes():
    """Within-block randomisation should pass all four checks at a high rate."""
    from slowlab.agents import spread_allocation
    env = mk("T3")
    ok = 0
    for s in range(30):
        rng = np.random.default_rng(s)
        alloc = spread_allocation([f"t{i}" for i in range(6)], 2, env.facility, rng)
        if score_validity(_six_by_two(env, alloc), env.facility, env.task).passed:
            ok += 1
    assert ok >= 27          # allows the few false positives a 99th-percentile threshold produces


def test_balance_threshold_is_deterministic():
    """Scoring the same design twice must give the same threshold -- scoring may not be random."""
    from slowlab.agents import spread_allocation
    env = mk("T3")
    alloc = spread_allocation([f"t{i}" for i in range(6)], 2, env.facility,
                              np.random.default_rng(0))
    d = _six_by_two(env, alloc)
    a = score_validity(d, env.facility, env.task).advisory["assoc_threshold"]
    b = score_validity(d, env.facility, env.task).advisory["assoc_threshold"]
    assert a == b


def test_declared_seed_alone_is_not_enough():
    """A declared seed with a systematic allocation must still fail overall."""
    env = mk("T3")
    by_c = [u.id for u in sorted(env.facility.units, key=lambda u: (u.chamber, u.loop))]
    d = _six_by_two(env, {f"t{i}": [by_c[2*i], by_c[2*i+1]] for i in range(6)})
    r = score_validity(d, env.facility, env.task)
    assert r.checks["randomization_declared"]      # the seed was declared
    assert not r.passed                            # but the design still fails


def test_trajectory_endpoint_matches_submitted_recommendation():
    """The last point of the trajectory must equal the regret of the submitted
    recommendation.

    Otherwise the claim that "eliciting x_hat_r changes nothing; it reads out a
    trajectory instead of an endpoint" is false. DoEAgent's interim once used the
    best observed point while it finally submitted the GP posterior maximum; on
    Optimise seed 0 the two differed by 47%.
    """
    from slowlab import SlowLabEnv, TASKS
    from slowlab.registry import make_agent
    # Run all four named tasks against all four reference strategies. This used to
    # run only the legacy alias TASKS["T3"] and three of the four, missing
    # gp_ucb_rep2 and never touching the v1.0 task set.
    for cfg in ("Sanity", "Screen", "Optimise", "Transfer"):
        for name in ("random_spread", "classical_doe", "gp_ucb_rep1", "gp_ucb_rep2"):
            env = SlowLabEnv(TASKS[cfg], seed=0)
            ag = make_agent(name)
            r = env.submit_recommendation(ag.run(env), name)
            assert r.regret_trace, (cfg, name)
            assert abs(r.regret_trace[-1] - r.simple_regret) < 1e-9, (
                cfg, name, r.regret_trace[-1], r.simple_regret)


def test_information_regret_reference_beats_a_degenerate_design():
    """The best-of-K random reference must beat a degenerate "one treatment,
    replicated everywhere" design.

    Otherwise information regret has no meaningful anchor -- it must be reachable
    but non-trivial, not something any design exceeds.
    """
    import numpy as np
    from slowlab import SlowLabEnv, TASKS
    from slowlab.eig import build_atoms, eig_of_design, reference_eig
    t = TASKS["Optimise"]
    env = SlowLabEnv(t, seed=0)
    sd = env.truth.response_sd
    taus = (t.tau_chamber * sd, t.tau_loop * sd, t.tau_batch * sd)
    atoms = build_atoms(t, M=60, nbins=12)
    ref = reference_eig(atoms, t, env.facility, taus, K=8,
                        rng=np.random.default_rng(0), n_outer=60)
    n = t.units_per_round
    flat = eig_of_design(atoms, np.full((n, len(t.factors)), 0.5),
                         np.repeat(np.arange(t.n_chambers), n // t.n_chambers),
                         *taus, n_outer=60, rng=np.random.default_rng(1))
    assert ref > flat, (ref, flat)
    assert 0.0 < ref < atoms.prior_entropy()


def test_unit_is_a_population_of_plants_not_one_plant():
    """A unit is the whole stand on one loop, and the observation is their mean.

    We used to draw a single plant per unit, overstating unit-level noise by
    sqrt(N): noise of 0.0165 swamped the between-site spread of 0.0066, making
    experimenting unprofitable in principle.
    """
    import numpy as np
    from slowlab import SlowLabEnv, TASKS
    from slowlab.design import Design
    t = TASKS["Optimise"]
    env = SlowLabEnv(t, seed=0)
    assert env.facility.plants_per_unit(3.25) > 20
    trt = {"a": {f.name: 0.5 for f in t.factors}}
    alloc = {"a": [u.id for u in env.facility.units[:12]]}
    env.submit_design(Design(trt, alloc, randomization_seed=1))
    env.advance()
    v = np.array([o.value for o in env.observations()])
    assert v.std(ddof=1) < 0.012, v.std(ddof=1)


def test_best_fixed_reference_beats_any_hand_picked_constant():
    """The best fixed recommendation must beat any hand-picked constant, otherwise
    it is not a bound on "without experimenting".

    It replaces the hand-written literature prior: those six textbook values had no
    source, whereas this quantity is computable, uncited, and evaluated on held-out
    instances.
    """
    import numpy as np
    from slowlab import TASKS
    from slowlab.fixed_reference import best_fixed
    from slowlab.world import ManagedTomgro
    t = TASKS["Optimise"]
    _, ref = best_fixed(t)
    rng = np.random.default_rng(3)
    W = [ManagedTomgro(seed=s, factors=t.factors, cycle_days=t.cycle_days)
         for s in range(1000, 1010)]
    b = [w.oracle()[1] for w in W]
    for _ in range(5):
        x = rng.random((1, len(t.factors)))
        r = float(np.mean([bi - float(w(x)[0]) for w, bi in zip(W, b)]))
        assert ref <= r + 1e-9, (ref, r)


def test_best_fixed_search_has_converged():
    """Given a stronger set of candidates, this oracle bound should not fall further.

    This test was once written as raising n_cand from 20k to 40k and checking that
    nothing moved -- which compares a biased estimator against itself. Uniform
    sampling converges as n^(-1/d), so in six dimensions doubling barely moves it:
    the test passed while the answer was wrong by a factor of five (Screen reported
    0.0029 against a true 0.0005).

    The candidate set now includes the fitting worlds' true optima. This adds two
    independent strong candidates -- the held-out worlds' optima, and local
    perturbations around the current solution -- and if the answer still drops
    appreciably then it has not converged. A loose bound is the direction that
    flatters the agent, so this must be guarded.
    """
    import numpy as np
    from slowlab import TASKS
    from slowlab.fixed_reference import best_fixed, _worlds
    for name in ("Screen",):        # only the highest-dimensional task; this check is expensive
        t = TASKS[name]
        d = len(t.factors)
        x0, ref = best_fixed(t, cache=False)
        rng = np.random.default_rng(1)
        Wf = _worlds(t, range(40))
        We = _worlds(t, range(1000, 1040))
        bf = np.array([w.oracle()[1] for w in Wf])
        be = np.array([w.oracle()[1] for w in We])
        extra = np.vstack([
            np.array([w.oracle()[0] for w in We], float).reshape(-1, d),
            np.clip(x0 + 0.05 * rng.standard_normal((1000, d)), 0, 1),
            np.clip(x0 + 0.20 * rng.standard_normal((1000, d)), 0, 1),
            rng.random((8_000, d))])
        R = np.zeros(len(extra))
        for w, b in zip(Wf, bf):
            R += b - w(extra)
        x = extra[int(np.argmin(R / len(Wf)))]
        alt = float(np.mean([b - float(w(x.reshape(1, -1))[0])
                             for w, b in zip(We, be)]))
        assert ref - alt < 0.30 * max(ref, alt), (
            f"{name}: the current candidate set reports {ref:.4f}, but adding local "
            f"perturbations and held-out optima gives {alt:.4f} -- the search in {d} "
            f"dimensions has not converged and this bound is loose")




def test_oracle_is_converged():
    """An independent, stronger search should not appreciably beat the oracle.

    The oracle appears in the definition of regret, f(x*) - f(x_hat), so
    understating it understates every agent's regret and flatters the agent. The
    old implementation refined from the single best uniform sample; in six
    dimensions 20% of instances were beaten by a candidate set of only 2320 points,
    by 0.00062 -- comparable to Screen's entire spread. With multi-start the
    residual fell to 0.000035.

    This attacks with 38000 points (uniform plus local perturbation at two scales)
    and allows an excess of 20% of the task's spread.
    """
    import numpy as np
    from slowlab import TASKS
    from slowlab.world import ManagedTomgro
    from slowlab.fixed_reference import best_fixed
    for name in ("Screen", "Transfer"):          # the two highest-dimensional tasks
        t = TASKS[name]
        d = len(t.factors)
        _, spread = best_fixed(t)
        worst = 0.0
        for s in range(8):
            w = ManagedTomgro(seed=10_000 + s, factors=t.factors,
                              cycle_days=t.cycle_days)
            x, b = w.oracle()
            rng = np.random.default_rng(777 + s)
            C = np.vstack([rng.random((30_000, d)),
                           np.clip(x + 0.02 * rng.standard_normal((4000, d)), 0, 1),
                           np.clip(x + 0.08 * rng.standard_normal((4000, d)), 0, 1)])
            worst = max(worst, float(np.max(w(C)) - b))
        assert worst < 0.20 * spread, (
            f"{name}: the oracle was beaten by {worst:.6f}, which is "
            f"{100*worst/spread:.0f}% of this task's spread of {spread:.5f}; "
            f"the baseline for regret cannot be trusted")
