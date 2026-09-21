from pathlib import Path

import pytest

from slowlab.greenlight_adapter import (
    greenlight_canopy_water_demand,
    integrate_canopy_water_demand,
    read_greenlight_series,
)


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
