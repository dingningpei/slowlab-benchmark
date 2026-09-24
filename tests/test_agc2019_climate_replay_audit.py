from datetime import date
import json

import pytest

from scripts.audit_agc2019_climate_replay import audit


def test_climate_audit_separates_complete_solve_from_physical_validity(tmp_path):
    simulation = tmp_path / "simulation.csv"
    simulation.write_text(
        "Time,tAir,rhIn,co2InPpm\n"
        "Description,Air,RH,CO2\n"
        "Units,C,%,ppm\n"
        + "".join(f"{hour * 3600},20,80,{'-1' if hour == 1 else '500'}\n"
                  for hour in range(24))
    )
    observations = tmp_path / "observations.csv"
    observations.write_text(
        "timestamp,air_temperature_c,relative_humidity_pct,co2_ppm\n"
        "2020-03-01T00:00:00,19,79,480\n"
        "2020-03-01T23:00:00,19,79,480\n"
    )
    solver = tmp_path / "solver.json"
    solver.write_text(json.dumps({"LSODA": {"success": True, "end": 86400}}))

    result = audit(simulation, observations, date(2020, 3, 1), solver, "LSODA")
    assert result["solver_complete"] is True
    assert result["physical_violations"]["co2InPpm"] == 1
    assert result["metrics"]["tAir"]["mae"] == pytest.approx(1.0)
    assert result["status"] == "diagnostic_only_not_physical_validation"

    simulation.write_text(simulation.read_text().replace("82800,20,80,500\n", ""))
    with pytest.raises(ValueError, match="does not cover the day"):
        audit(simulation, observations, date(2020, 3, 1), solver, "LSODA")


def test_holdout_climate_is_not_inspected(tmp_path):
    with pytest.raises(ValueError, match="calibration dates"):
        audit(tmp_path / "missing.csv", tmp_path / "missing2.csv",
              date(2020, 4, 1), tmp_path / "missing3.json")
