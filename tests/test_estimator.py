"""Pin down the R* estimator fix so nobody quietly reverts to the in-sample version.

Background is in the 2026-09-11 note at the top of achievable.py:
achievable_regret lets one set of atoms both choose and evaluate the
recommendation, ESS is about 1, and c changes sign when evaluated held out.
Everything reported must go through achievable_regret_cv.

These tests need a cached atom set (about 2 minutes to build once) and skip
without one.
    python scripts/rstar_convergence.py --cfg Optimise   # builds it as a side effect
"""
from __future__ import annotations
import pathlib, pickle, sys
import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import slowlab
from slowlab import SlowLabEnv, TASKS
from slowlab.achievable import (achievable_regret, achievable_regret_cv,
                                calibrate_temperature, split_atoms,
                                candidate_set, response_table)

CFG = "Optimise"
ATOMS = ROOT / "results" / f"_atoms_{CFG}_M320_b12_env{slowlab.ENV_VERSION}.pkl"
needs_atoms = pytest.mark.skipif(not ATOMS.exists(), reason=f"{ATOMS.name} is missing")


@pytest.fixture(scope="module")
def fix():
    a = pickle.loads(ATOMS.read_bytes())
    t = TASKS[CFG]
    env = SlowLabEnv(t, seed=0)
    sd = env.truth.response_sd
    taus = (t.tau_chamber * sd, t.tau_loop * sd, t.tau_batch * sd)
    ain, aout = split_atoms(a)
    cand = candidate_set(ain, rng=np.random.default_rng(0))
    k = min(t.units_per_round, len(env.facility.units))
    blk = np.array([env.facility.units[i].chamber for i in range(k)])
    d = len(t.factors)
    pts = np.random.default_rng(3).random((k, d))          # a space-filling design
    return dict(ain=ain, aout=aout, taus=taus, cand=cand, blk=blk, pts=pts,
                d=d, F_in=response_table(ain, cand),
                F_out=response_table(aout, cand))


@needs_atoms
def test_leave_one_out_calibration_picks_a_temperature_above_one(fix):
    """Untempered, the likelihood gives ESS about 1. Calibration must notice and temper."""
    T = calibrate_temperature(fix["ain"], fix["pts"], fix["blk"], *fix["taus"],
                              F=fix["F_in"], cand=fix["cand"], n_cal=80)
    assert T > 1.0, f"calibration chose T={T}, which is no fix at all -- check calibrate_temperature"


@needs_atoms
def test_in_sample_version_is_visibly_optimistic(fix):
    """This is the reason for the fix. If the two are close, the bias was removed
    somewhere else and this file needs rewriting."""
    ain, aout, taus = fix["ain"], fix["aout"], fix["taus"]
    kw = dict(cand=fix["cand"], n_outer=200)
    e = (np.zeros((0, fix["d"])), np.zeros(0, int))
    P_in = achievable_regret(ain, *e, *taus, F=fix["F_in"], **kw)
    P_cv = achievable_regret_cv(ain, aout, *e, *taus, F_in=fix["F_in"],
                                F_out=fix["F_out"], **kw)
    v_in = achievable_regret(ain, fix["pts"], fix["blk"], *taus,
                             F=fix["F_in"], rng=np.random.default_rng(11), **kw)
    v_cv = achievable_regret_cv(ain, aout, fix["pts"], fix["blk"], *taus,
                                F_in=fix["F_in"], F_out=fix["F_out"],
                                rng=np.random.default_rng(11), **kw)
    c_in, c_cv = 1 - v_in / P_in, 1 - v_cv / P_cv
    assert c_in > 0.80, f"in-sample c={c_in:.2f} does not match the recorded 0.97"
    assert c_cv < c_in - 0.30, (
        f"c in sample {c_in:.2f} vs cross-validated {c_cv:.2f} -- the gap should exceed 30 points")


@needs_atoms
def test_repaired_c_lands_in_a_plausible_range(fix):
    """The point of the fix is to pull c back into a credible range; a negative value
    means the tempering did not take effect."""
    ain, aout, taus = fix["ain"], fix["aout"], fix["taus"]
    kw = dict(cand=fix["cand"], F_in=fix["F_in"], F_out=fix["F_out"],
              n_outer=200)
    P = achievable_regret_cv(ain, aout, np.zeros((0, fix["d"])),
                             np.zeros(0, int), *taus, **kw)
    v = achievable_regret_cv(ain, aout, fix["pts"], fix["blk"], *taus,
                             rng=np.random.default_rng(11), **kw)
    c = 1 - v / P
    assert 0.0 < c < 0.60, f"c={c:.2f} is outside the range"


@needs_atoms
def test_the_empty_design_ceiling_is_insensitive_to_the_estimator(fix):
    """R*(none) is the denominator of the abstract's headline. If it drifts with the
    estimator too, that claim has to be withdrawn as well."""
    ain, aout, taus = fix["ain"], fix["aout"], fix["taus"]
    e = (np.zeros((0, fix["d"])), np.zeros(0, int))
    P_in = achievable_regret(ain, *e, *taus, F=fix["F_in"], cand=fix["cand"],
                             n_outer=200)
    P_cv = achievable_regret_cv(ain, aout, *e, *taus, F_in=fix["F_in"],
                                F_out=fix["F_out"], cand=fix["cand"],
                                n_outer=200)
    assert abs(P_cv - P_in) / P_in < 0.15, (
        f"R*(none) differs by {100*abs(P_cv-P_in)/P_in:.0f}% between the two estimators -- the ceiling itself is unstable")
