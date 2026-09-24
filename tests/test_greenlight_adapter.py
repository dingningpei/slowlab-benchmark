import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from slowlab.greenlight_adapter import (
    assert_greenlight_solution_complete,
    greenlight_canopy_water_demand,
    greenlight_initial_climate_override,
    integrate_canopy_water_demand,
    read_greenlight_series,
)


def test_observed_climate_initials_use_greenlight_state_units():
    state = greenlight_initial_climate_override(20.4, 80.8, 569)
    temp = float(state["tAir"]["init"])
    vapour_pressure = float(state["vpAir"]["init"])
    co2_density = float(state["co2Air"]["init"])
    saturation_pressure = 610.78 * math.exp(17.2694 * temp / (temp + 238.3))
    assert 100 * vapour_pressure / saturation_pressure == pytest.approx(80.8)
    assert (8.3144598e6 * (temp + 273.15) * 1e-6 * co2_density
            / (101325 * 44.01e-3)) == pytest.approx(569)


def test_unphysical_observed_climate_cannot_initialize_greenlight():
    with pytest.raises(ValueError, match="physical ranges"):
        greenlight_initial_climate_override(20, 101, 569)


def test_partial_solver_result_fails_even_if_output_file_was_written():
    assert_greenlight_solution_complete(SimpleNamespace(success=True, t=[0, 86400]), 86400)
    with pytest.raises(ValueError, match="solver failed"):
        assert_greenlight_solution_complete(
            SimpleNamespace(success=False, t=[0, 16263], message="step size too small"), 86400)
    with pytest.raises(ValueError, match="stopped"):
        assert_greenlight_solution_complete(
            SimpleNamespace(success=True, t=[0, 16263]), 86400)


def test_interval_start_integration_converts_kg_per_m2_to_litres_per_m2():
    result = integrate_canopy_water_demand([0, 10, 20], [0, 1, 0])
    assert result.root_uptake_l_m2 == pytest.approx(10.0)
    assert result.canopy_condensation_l_m2 == 0


def test_sign_change_separates_root_demand_and_condensation():
    result = integrate_canopy_water_demand([0, 1, 2], [-1, 1, 0])
    assert result.root_uptake_l_m2 == pytest.approx(1.0)
    assert result.canopy_condensation_l_m2 == pytest.approx(1.0)


def test_nonmonotone_time_is_rejected():
    with pytest.raises(ValueError):
        integrate_canopy_water_demand([0, 1, 1], [0, 1, 2])


def test_greenlight_metadata_rows_and_units_are_checked(tmp_path: Path):
    path = tmp_path / "output.csv"
    path.write_text(
        "Time,mvCanAir\n"
        "Time since start,Canopy transpiration\n"
        "s,kg m**-2 s**-1\n"
        "0,0.0\n"
        "3600,0.000001\n"
    )
    series = read_greenlight_series(path, "mvCanAir")
    assert series.description == "Canopy transpiration"
    assert series.time_seconds == (0.0, 3600.0)
    result = greenlight_canopy_water_demand(path)
    assert result.root_uptake_l_m2 == pytest.approx(0.0036)


def test_changed_greenlight_unit_fails_closed(tmp_path: Path):
    path = tmp_path / "output.csv"
    path.write_text(
        "Time,mvCanAir\n"
        "Time since start,Canopy transpiration\n"
        "s,g m**-2 s**-1\n"
        "0,0.0\n"
        "3600,0.001\n"
    )
    with pytest.raises(ValueError, match="unit changed"):
        greenlight_canopy_water_demand(path)


def test_pipe_initials_use_positive_observation_and_never_zero_off_code():
    from slowlab.greenlight_adapter import greenlight_initial_pipe_override
    assert greenlight_initial_pipe_override(20, 45.9, 0) == {
        "tPipe": {"init": "45.9"}, "tGroPipe": {"init": "20.0"}}


def test_pipe_initials_reject_negative_or_nonfinite_codes():
    from slowlab.greenlight_adapter import greenlight_initial_pipe_override
    import pytest
    with pytest.raises(ValueError):
        greenlight_initial_pipe_override(20, -1, 0)
    with pytest.raises(ValueError):
        greenlight_initial_pipe_override(float("nan"), 40, 0)
