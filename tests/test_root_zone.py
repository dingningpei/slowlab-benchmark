import math

import pytest

from slowlab.root_zone import (
    RockwoolRootZone,
    RootZoneForcing,
    RootZoneParams,
    RootZoneState,
)


def _model(**overrides):
    values = dict(
        pore_volume_l_m2=10.0,
        drain_threshold_fraction=0.8,
        drain_rate_per_hour=0.5,
        ec_to_concentration_mmol_l_per_ds_m=10.0,
        crop_uptake_concentration_mmol_l=5.0,
    )
    values.update(overrides)
    return RockwoolRootZone(RootZoneParams(**values))


def test_no_flow_preserves_state():
    model = _model()
    state = RootZoneState(7.0, 350.0, 21.0)
    nxt, flux = model.step(state, RootZoneForcing(0, 0, 0), dt_hours=1 / 12)
    assert nxt == state
    assert flux.drain_l_m2 == 0


def test_irrigation_adds_water_and_feed_solute_with_declared_conversion():
    model = _model(drain_rate_per_hour=0)
    state = RootZoneState(5.0, 200.0, 20.0)
    nxt, flux = model.step(
        state, RootZoneForcing(1.0, 3.0, 0.0), dt_hours=1 / 12
    )
    assert nxt.water_l_m2 == pytest.approx(6.0)
    assert flux.feed_solute_mmol_m2 == pytest.approx(30.0)
    assert nxt.solute_mmol_m2 == pytest.approx(230.0)
    assert model.ec_ds_m(nxt) == pytest.approx((230 / 6) / 10)


def test_water_and_solute_balances_close_with_uptake_and_drain():
    model = _model(drain_rate_per_hour=1.0)
    state = RootZoneState(9.0, 450.0, 20.0)
    forcing = RootZoneForcing(
        irrigation_l_m2=3.0,
        feed_ec_ds_m=4.0,
        potential_root_uptake_l_m2=0.5,
        substrate_evaporation_l_m2=0.1,
    )
    nxt, flux = model.step(state, forcing, dt_hours=1.0)
    water_in = state.water_l_m2 + flux.irrigation_l_m2
    water_out = (
        nxt.water_l_m2 + flux.root_uptake_l_m2
        + flux.substrate_evaporation_l_m2 + flux.drain_l_m2
    )
    solute_in = state.solute_mmol_m2 + flux.feed_solute_mmol_m2
    solute_out = (
        nxt.solute_mmol_m2 + flux.crop_solute_uptake_mmol_m2
        + flux.drained_solute_mmol_m2
    )
    assert water_out == pytest.approx(water_in)
    assert solute_out == pytest.approx(solute_in)
    assert 0 <= model.water_content_pct(nxt) <= 100


def test_requested_uptake_is_capped_by_available_water():
    model = _model(drain_rate_per_hour=0)
    state = RootZoneState(0.2, 1.0, 20.0)
    nxt, flux = model.step(
        state, RootZoneForcing(0.0, 0.0, 1.0, 0.1), dt_hours=1.0
    )
    assert flux.substrate_evaporation_l_m2 == pytest.approx(0.1)
    assert flux.root_uptake_l_m2 == pytest.approx(0.1)
    assert nxt.water_l_m2 == pytest.approx(0.0)
    assert math.isnan(model.ec_ds_m(nxt))


def test_parameters_reject_nonphysical_values():
    with pytest.raises(ValueError):
        _model(pore_volume_l_m2=0)
    with pytest.raises(ValueError):
        _model(drain_threshold_fraction=1.1)
    with pytest.raises(ValueError):
        _model(drain_rate_per_hour=-1)
