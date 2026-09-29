

def test_reference_and_model_eig_use_the_same_ruler():
    """References and models under test must be scored at the same M / n_outer / nbins.

    EIG/H is strongly sensitive to M (Appendix G), so different settings on the
    two sides turn "model vs reference" into a comparison between settings rather
    than between designs. This bug happened once: the reference at M=60 and the
    model at M=80 changed which strategies the model beat on Optimise.
    """
    import inspect, scripts.run_eig as R, scripts.eig_of_llm as L
    a = inspect.signature(R.main).parameters
    b = inspect.signature(L.main).parameters
    for k in ("M", "n_outer", "nbins"):
        assert a[k].default == b[k].default, (
            f"run_eig.main has {k}={a[k].default} but eig_of_llm.main has "
            f"{k}={b[k].default}; changing one requires changing the other and rerunning both sides")


def test_achievable_regret_anchors_at_the_best_constant():
    """R*(empty design) must equal the regret of the best fixed recommendation.

    This is the whole basis for eq:eig being self-contained: with no design, the
    Bayes-optimal decision is that constant. The two are computed by entirely
    different code paths (one through atom weighting, one through a held-out set),
    so a mismatch means one of them is wrong.
    """
    import numpy as np
    from slowlab import TASKS
    from slowlab.achievable import achievable_regret, candidate_set, response_table
    from slowlab.fixed_reference import best_fixed
    from scripts.eig_of_llm import atoms_for
    for name in ("Sanity", "Optimise"):          # low-dimensional, where best_fixed has converged
        t = TASKS[name]
        a = atoms_for(name, 80, 12)
        cand = candidate_set(a, rng=np.random.default_rng(0))
        r = achievable_regret(a, np.zeros((0, len(t.factors))), np.zeros(0, int),
                              0.0, 0.0, 0.0, F=response_table(a, cand), cand=cand)
        _, bf = best_fixed(t)
        assert abs(r - bf) < 0.25 * bf, (name, r, bf)


def test_achievable_regret_never_exceeds_the_empty_design():
    """No design should be worse than running nothing -- data cannot have negative value."""
    import numpy as np
    from slowlab import SlowLabEnv, TASKS
    from slowlab.achievable import achievable_regret, candidate_set, response_table
    from scripts.eig_of_llm import atoms_for
    t = TASKS["Optimise"]
    env = SlowLabEnv(t, seed=0)
    sd = env.truth.response_sd
    taus = (t.tau_chamber * sd, t.tau_loop * sd, t.tau_batch * sd)
    n = min(t.units_per_round, len(env.facility.units))
    blk = np.array([env.facility.units[i].chamber for i in range(n)])
    a = atoms_for("Optimise", 80, 12)
    cand = candidate_set(a, rng=np.random.default_rng(0))
    kw = dict(F=response_table(a, cand), cand=cand)
    empty = achievable_regret(a, np.zeros((0, len(t.factors))), np.zeros(0, int),
                              *taus, **kw)
    for seed in range(3):
        pts = np.random.default_rng(seed).random((n, len(t.factors)))
        v = achievable_regret(a, pts, blk, *taus,
                              rng=np.random.default_rng(7), **kw)
        assert v <= empty + 1e-9, (seed, v, empty)
