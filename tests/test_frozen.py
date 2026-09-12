"""The environment freeze guard -- v1.0, 2026-09-10. See ENVIRONMENT_v1.0.md.

These tests **do not check that the environment is correct.** They check that it
**has not changed.**

Why this is needed: the previous working pattern was find a defect, fix it, the
ceiling moves, the tasks get redefined, find another defect -- and it did not
converge. With the environment frozen, any change to ground truth must be
*deliberate*: bump slowlab.ENV_VERSION, update the fingerprints here, and accept
that every episode already run is invalidated.

If you are seeing these tests fail because you changed world / economics /
tomgro / tasks: the tests are not broken, you are invalidating experiments that
have already been run.
"""
from __future__ import annotations
import numpy as np
import pytest

import slowlab
from slowlab.economics import EconomicModel, sample_site_econ
from slowlab.tasks import TASKS
from slowlab.world import ManagedTomgro, MANAGEMENT_FACTORS

TOL = 1e-9


def test_env_version_pinned():
    assert slowlab.ENV_VERSION == "1.0.0", (
        "The environment version changed. If ground truth really did change that is "
        "correct -- but update the fingerprints in this file too, and record in "
        "ENVIRONMENT_v1.0.md what changed, why, and which results are invalidated.")


def test_lambda_is_gone():
    """lambda is no longer a task design parameter. Anything != 1 requires an explicit strict=False."""
    assert EconomicModel().cost_weight == 1.0
    with pytest.raises(ValueError):
        EconomicModel(cost_weight=0.5)
    EconomicModel(cost_weight=0.5, strict=False)      # the explicit exemption still works


def test_factor_ranges():
    """Factor ranges -- density must stay inside the commercial band."""
    got = {f.name: (f.low, f.high, f.control_level) for f in MANAGEMENT_FACTORS}
    assert got == {
        "day_temp":   (18.0, 32.0, "chamber"),
        "night_temp": (12.0, 22.0, "chamber"),
        "co2":       (350.0, 1200.0, "chamber"),
        "par":        (18.0, 32.0, "chamber"),
        "density":     (2.2, 4.5, "loop"),
        "lai_max":     (2.0, 5.0, "loop"),
    }


def test_task_geometry():
    """The geometry and budget of the four tasks."""
    for k, (nf, nc, lpc, nr, upr) in {
        "T1": (1, 4, 4, 2, 8),
        "T2": (6, 8, 3, 2, 12),
        "T3": (2, 4, 3, 3, 12),
        "T4": (4, 8, 3, 3, 24),
    }.items():
        t = TASKS[k]
        assert (len(t.factors), t.n_chambers, t.loops_per_chamber,
                t.n_rounds, t.units_per_round) == (nf, nc, lpc, nr, upr), k


def test_transfer_is_an_energy_shock_not_a_lambda_shock():
    t = TASKS["T4"]
    assert not hasattr(t, "transfer_cost_weight")
    assert t.transfer_energy_shock == 2.2
    assert t.train_energy_shock is None


def test_forward_model_fingerprint():
    """Ground-truth output at fixed points. Any change to tomgro / economics / world breaks it."""
    w = ManagedTomgro(seed=0, econ=EconomicModel())
    X = np.array([[0.30, 0.55, 0.20, 0.00, 0.35, 0.70],
                  [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
                  [0.70, 0.30, 0.80, 0.20, 0.60, 0.40]])
    got = np.asarray(w(X), float)
    want = np.array([0.04497318, -0.16932811, -0.16039058])
    assert np.allclose(got, want, atol=1e-6), f"ground-truth output changed: {got.tolist()}"


def test_site_distribution_fingerprint():
    """A site is a place. Climate and prices vary with the seed, and the distribution may not move."""
    e = sample_site_econ(3)
    for k, v in {"elec_price": 0.17987601, "heat_price": 0.05803167,
                 "led_efficacy": 1.85656492, "u_value": 4.49981079,
                 "price_per_kg_fw": 1.53752976,
                 "t_out_day": 13.94325531, "t_out_night": 6.29550354}.items():
        assert abs(getattr(e, k) - v) < 1e-6, f"{k}: {getattr(e, k)}"


def test_physics_corrections_are_in():
    """The three physics fixes must still be in place: solar gain, ventilation-driven CO2 loss, flue-gas CO2."""
    e = EconomicModel()
    # Solar gain: transmitted solar far exceeds daytime heat loss, so no daytime heating
    assert e.transmitted_solar() > 6 * e.u_value * (22.0 - e.t_out_day) * 12.0
    assert float(e.heating_kwh(22.0, 18.0)) == pytest.approx(0.48, abs=1e-3)
    # Ventilation: a higher day setpoint means less air exchange
    assert float(e.ventilation_volume(20.0)) > float(e.ventilation_volume(28.0))
    # Flue gas: the free CO2 that comes with heating makes enrichment cheaper
    assert float(e.co2_cost(494.0, 22.0, 18.0)) < float(e.co2_cost(494.0, 22.0))


def test_photosynthesis_reads_site_parameters():
    """_Pg must use the parameter dict passed to it, not the global nominal values.

    This was once a bug: every site had an identical photosynthetic response, which
    made the appendix's claim that sampling the light and CO2 parameters would
    double the ceiling a no-op.
    """
    from slowlab.tomgro import TomgroModel, TomgroParams
    m = TomgroModel(params=TomgroParams(values={}, provenance="j99"),
                    factors=MANAGEMENT_FACTORS[:4], days=1)
    lo = float(m._Pg(2.0, 20.0, 600.0, {"alpha": 0.081, "sigma": 0.05976}))
    hi = float(m._Pg(2.0, 20.0, 600.0, {"alpha": 0.099, "sigma": 0.07304}))
    assert hi > lo * 1.05, "_Pg does not vary with site parameters -- the bug has regressed"


def test_gross_margin_is_not_called_net_profit_silently():
    """Gross margin is not net profit. The name must be honest, and cost_weight no longer enters the formula."""
    e = EconomicModel()
    assert hasattr(e, "gross_margin")
    a = float(e.gross_margin(300.0, 22.0, 18.0, 600.0, 18.0, 210.0, 3.0))
    e2 = EconomicModel(cost_weight=0.5, strict=False)
    b = float(e2.gross_margin(300.0, 22.0, 18.0, 600.0, 18.0, 210.0, 3.0))
    assert a == pytest.approx(b), "cost_weight still affects gross margin -- it should have been removed from the formula"


def test_irreversibility_is_commitment_only():
    """Irreversibility in v1.0 is about *commitment*, not *damage*.

    The heat-damage recovery mechanism exists but is set to 0 on all four tasks.
    This test pins that fact down: whoever turns it on must bump ENV_VERSION at the
    same time, because it invalidates every episode already run.

    It is not dead code: 38-45% of the units the reference strategies submit cross
    the heat-damage threshold, by 3.4-4.3 C on average. Leaving it off is our
    choice, not a mechanism that never fires.
    """
    for k in ("T1", "T2", "T3", "T4"):
        t = TASKS[k]
        assert t.recovery_days == 0.0, (
            f"{k}: recovery_days={t.recovery_days}. Turning on heat-damage recovery "
            "changes ground truth -- bump slowlab.ENV_VERSION and update the two "
            "passages in the paper saying irreversibility is commitment, not damage.")
        assert t.damage_margin == 2.0 and t.damage_scale == 6.0


def test_damage_threshold_would_actually_fire():
    """Guards the premise of the test above: the threshold really is somewhere
    agents go.

    If the day_temp range or the TCRIT distribution ever changes so that crossing
    becomes impossible, the claim that we deliberately disabled a mechanism that
    would have fired no longer holds and the paper must be rewritten.
    """
    from slowlab.world import ManagedTomgro
    t = TASKS["T3"]
    hi = max(f.high for f in t.factors if f.name == "day_temp")
    tcrit = [ManagedTomgro(seed=s).params.values["TCRIT"] for s in range(20)]
    assert hi > max(tcrit) + t.damage_margin, (
        "the upper rail of day_temp can no longer reach the heat-damage threshold -- "
        "the mechanism has become untriggerable and the 38-45% claim must be remeasured.")
