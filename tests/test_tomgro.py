import numpy as np
import pytest

from slowlab.tomgro import (TomgroModel, TomgroParams, APPROXIMATED,
                            PlaceholderParametersError)
from slowlab.factors import FactorSpec


def test_placeholder_guard_blocks_invented_params():
    """Placeholder parameters must refuse to construct, so invented numbers cannot
    reach the paper. The default constructor goes through from_j99(), so this
    passes a placeholder parameter object explicitly."""
    with pytest.raises(PlaceholderParametersError):
        TomgroModel(params=TomgroParams(values={"Nm": 1.0}, provenance="placeholder"))


def test_j99_params_run():
    m = TomgroModel(params=TomgroParams.from_j99())
    y = m(np.random.default_rng(0).random((16, m.d)))
    assert y.shape == (16,) and np.all(np.isfinite(y)) and y.max() > 0


def test_j99_provenance_carries_source():
    p = TomgroParams.from_j99()
    p.require_real()
    assert "Jones" in p.source and p.site == "Gainesville"


def test_backend_interface():
    """A ground-truth backend must provide __call__ / oracle / response_sd."""
    tg = TomgroModel(params=TomgroParams.from_j99())
    for attr in ("__call__", "oracle", "response_sd"):
        assert hasattr(tg, attr)
    x, best = tg.oracle()
    assert x.shape == (tg.d,) and np.isfinite(best)
    assert tg.response_sd > 0


def test_deterministic():
    m1 = TomgroModel(params=TomgroParams.from_j99())
    m2 = TomgroModel(params=TomgroParams.from_j99())
    X = np.random.default_rng(3).random((32, m1.d))
    assert np.allclose(m1(X), m2(X))


def test_plant_param_perturbation_varies():
    """Plant-to-plant perturbation must actually produce variation, and only touch parameters VC14 gives a nominal value for."""
    from slowlab.tomgro_params import VC14_NOMINAL
    m = TomgroModel(params=TomgroParams.from_j99())
    rng = np.random.default_rng(0)
    a, b = m.sample_plant_params(rng), m.sample_plant_params(rng)
    varied = [k for k in a if a[k] != b[k]]
    assert varied
    assert all(k in VC14_NOMINAL for k in varied)


def test_approximations_are_declared():
    """Sub-functions that are still approximations must be registered one by one, never passed off as J99's originals."""
    assert "Pg" in APPROXIMATED and "fN(T)" in APPROXIMATED
    assert all(isinstance(v, str) and v for v in APPROXIMATED.values())


# ---------- Published parameters ----------
def test_published_params_loaded():
    from slowlab.tomgro_params import (VC14_NOMINAL, G21_SEARCH_RANGE,
                                       nominal_params, perturbation_intervals)
    assert len(VC14_NOMINAL) == 17 and len(G21_SEARCH_RANGE) == 14
    n = nominal_params()
    assert n["Nm"] == 0.495 and n["E"] == 0.7 and n["alphaF"] == 0.95
    for k, (lo, hi) in perturbation_intervals("VC14").items():
        assert lo < n[k] < hi, f"{k}'s nominal value is not between its bounds"


def test_two_sources_disagree_and_it_is_recorded():
    """The two sources disagree on the same parameter. That is a fact, and it must be recorded explicitly rather than quietly resolved."""
    from slowlab.tomgro_params import VC14_NOMINAL, G21_SEARCH_RANGE, DISAGREEMENTS
    real = []
    for k in set(VC14_NOMINAL) & set(G21_SEARCH_RANGE):
        nom = VC14_NOMINAL[k][0]; lo, hi = G21_SEARCH_RANGE[k][:2]
        if not (lo <= nom <= hi):
            real.append(k)
    assert set(real) <= set(DISAGREEMENTS), f"unrecorded disagreements: {set(real)-set(DISAGREEMENTS)}"
    assert len(real) >= 5


def test_missing_functions_are_declared():
    """Functions that are still missing must be listed explicitly, never silently
    replaced by a placeholder. With J99 in hand GRnet is no longer missing
    (equation 5), but the Pg / fR(N) it depends on still are."""
    from slowlab.tomgro_params import MISSING_FUNCTIONS, SOURCE_FOR_MISSING
    assert "fN(T)" in MISSING_FUNCTIONS and "Pg" in MISSING_FUNCTIONS
    assert "GRnet" not in MISSING_FUNCTIONS      # equation (5) is known
    assert "Acock" in SOURCE_FOR_MISSING


# ---------- Jones et al. (1999), the original paper ----------
def test_j99_table3_three_sites():
    from slowlab.tomgro_params import J99_TABLE3, J99_CARRIED_OVER, J99_LAIMAX
    assert len(J99_TABLE3) == 10
    assert J99_TABLE3["beta"] == (0.169, 0.169, 0.169)      # identical at all three sites
    assert J99_TABLE3["NFF"] == (22.0, 10.0, 19.0)          # hugely different at the three sites
    assert J99_CARRIED_OVER["rm"] == 0.016 and J99_CARRIED_OVER["p1"] == 2.0
    assert J99_LAIMAX["LakeCity"] == 4.0


def test_g_daytime_matches_eq8():
    """Equation (8): g = 1 - 0.154 . (Tday - TCRIT), and exactly 1 below TCRIT."""
    from slowlab.tomgro_params import g_daytime
    assert g_daytime(20.0, 24.4) == 1.0
    assert g_daytime(24.4, 24.4) == 1.0
    assert abs(g_daytime(30.0, 24.4) - (1 - 0.154 * 5.6)) < 1e-9
    assert g_daytime(45.0, 24.4) == 0.0                     # clipped at 0


def test_vegetative_params_stable_fruit_params_not():
    """J99's central finding: vegetative parameters are stable across sites and
    fruit parameters are not. This is the direct basis for constructing T6
    (between-season non-stationarity)."""
    from slowlab.tomgro_params import J99_TABLE3
    veg = ["beta", "Nb"]
    for k in veg:
        v = [x for x in J99_TABLE3[k] if x is not None]
        assert max(v) / min(v) < 1.05, f"{k} should be stable across sites"
    v = [x for x in J99_TABLE3["NFF"] if x is not None]
    assert max(v) / min(v) > 2.0, "NFF should differ greatly across sites"


# ---------- Structural checks now that the J99 equations are wired in ----------
def _m(site="Gainesville", days=120):
    from slowlab.tomgro import TomgroModel, TomgroParams
    return TomgroModel(params=TomgroParams.from_j99(site), days=days)


def _x(m, day_t, night_t, co2, par):
    f = m.factors
    return np.array([[(day_t-f[0].low)/(f[0].high-f[0].low),
                      (night_t-f[1].low)/(f[1].high-f[1].low),
                      (co2-f[2].low)/(f[2].high-f[2].low),
                      (par-f[3].low)/(f[3].high-f[3].low)]])


def test_temperature_response_is_non_monotonic():
    """The entire reason for the mechanistic skeleton: structure a stationary GP
    cannot produce. Yield against day temperature should be unimodal, not
    monotone."""
    m = _m()
    ys = [m(_x(m, T, T-7, 700, 30))[0] for T in (20, 22, 24, 26, 28, 30, 32)]
    peak = int(np.argmax(ys))
    assert 0 < peak < len(ys) - 1, "the optimal temperature is on a boundary, so there is no unimodal structure"
    assert ys[peak] > 2 * ys[-1], "insufficient decline at the hot end"


def test_lai_caps_at_laimax():
    m = _m(); p = m.params.values
    lai = 0.01; N = 3.0
    for _ in range(m.days):
        dN = p["Nm"] * float(m._fN(np.array([25.0]))[0])
        e = np.exp(np.clip(p["beta"]*(N-p["Nb"]), -30, 30))
        if lai <= p["LAImax"]:
            lai += p["rho"]*p["delta"]*float(m._lambda(np.array([21.5]))[0])*e/(1+e)*dN
        N += dN
    assert lai <= p["LAImax"] * 1.05


def test_fruit_lags_behind_vegetative():
    """WF does not start until N > NFF, and WM lags a further kappaF nodes."""
    m = _m(days=40)
    assert m(_x(m, 25, 18, 700, 30))[0] == 0.0        # no mature fruit within 40 days
    m2 = _m(days=120)
    assert m2(_x(m2, 25, 18, 700, 30))[0] > 0.0


def test_source_sink_limit_binds():
    """The minimum in equation (10) must actually bind, otherwise the source-sink limit does nothing."""
    m = _m(); p = m.params.values
    GR = np.array([50.0]); dWF = np.array([1.0]); dN = np.array([0.5])
    dW9 = GR
    dW10 = dWF + (p["Vmax"] - p["p1"]) * p["rho"] * dN
    assert min(float(dW9[0]), float(dW10[0])) == float(dW10[0])


def test_sites_differ_mainly_in_fruit_parameters():
    from slowlab.tomgro import TomgroParams
    g = TomgroParams.from_j99("Gainesville").values
    l = TomgroParams.from_j99("LakeCity").values
    assert g["beta"] == l["beta"] and g["Nb"] == l["Nb"]     # vegetative parameters agree
    assert g["NFF"] != l["NFF"] and g["alphaF"] != l["alphaF"]  # fruit parameters differ


def test_j99_params_pass_guard():
    from slowlab.tomgro import TomgroModel, TomgroParams
    TomgroModel(params=TomgroParams.from_j99())      # should not raise
    assert TomgroParams.from_j99().provenance == "j99"


# ---------- The economic model and the margin backend ----------
def test_cost_components_nonnegative_and_monotone():
    from slowlab.economics import EconomicModel
    e = EconomicModel()
    assert e.lighting_cost(18.0) == 0.0                     # equal to natural light, so no supplemental cost
    assert e.lighting_cost(40.0) > e.lighting_cost(25.0) > 0
    assert e.heating_cost(24, 17) > e.heating_cost(16, 10) >= 0
    assert e.co2_cost(400.0) == 0.0 and e.co2_cost(1000.0) > 0


def test_free_resources_would_rail_out_the_resource_factors():
    """With resources priced at zero, CO2 and supplemental light go to their upper
    rails -- which is why the cost term cannot be dropped.

    This test used to build the counterfactual with cost_weight=0. lambda is now
    not merely pinned at 1 but *removed from the formula* (see
    test_frozen.test_gross_margin_naming), so that route can no longer construct
    any counterfactual -- it silently does nothing, and the test would fail for the
    wrong reason. It now zeroes the electricity, gas and CO2 prices directly.
    """
    from slowlab.tomgro import TomgroModel, TomgroParams
    from slowlab.economics import EconomicModel, TomgroProfitModel
    crop = TomgroModel(params=TomgroParams.from_j99(), days=120)
    free = EconomicModel(elec_price=0.0, heat_price=0.0, co2_price=0.0)
    x, _ = TomgroProfitModel(crop, free).oracle(n_sample=8000)
    assert x[2] > 0.9, f"CO2 is free yet did not reach the upper rail: {x[2]:.3f}"
    assert x[3] > 0.9, f"supplemental light is free yet did not reach the upper rail: {x[3]:.3f}"


def test_default_lambda_gives_interior_optima():
    """All four dimensions should have an interior optimum."""
    from slowlab.tomgro import TomgroModel, TomgroParams
    from slowlab.economics import TomgroProfitModel
    crop = TomgroModel(params=TomgroParams.from_j99(), days=120)
    m = TomgroProfitModel(crop)
    x, _ = m.oracle(n_sample=20000)
    assert all(0.03 < xi < 0.97 for xi in x), f"some dimension is still on a boundary: {x}"


def test_profit_backend_matches_truthmodel_interface():
    from slowlab.tomgro import TomgroModel, TomgroParams
    from slowlab.economics import TomgroProfitModel
    m = TomgroProfitModel(TomgroModel(params=TomgroParams.from_j99(), days=120))
    for attr in ("__call__", "oracle", "response_sd"):
        assert hasattr(m, attr)
    assert m.response_sd > 0
    y = m(np.random.default_rng(0).random((8, m.d)))
    assert y.shape == (8,) and np.all(np.isfinite(y))
