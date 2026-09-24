import csv

import pytest

from scripts.audit_agc2019_co2_rate_units import audit_compartment


def test_co2_rate_unit_hypotheses_are_not_interchanged(tmp_path):
    room = tmp_path / "Reference"
    room.mkdir()
    with (room / "GreenhouseClimate.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("%time", "co2_dos"))
        for index in range(288):
            writer.writerow((f"{43815 + index / 288:.8f}", 1))
    (room / "Resources.csv").write_text("%Time ,CO2_cons\n43815,24\n")
    result = audit_compartment(tmp_path, "Reference")
    assert result["days"] == 1
    assert result["median_relative_error_if_rate_kg_m2_h"] == pytest.approx(0, abs=1e-12)
    assert result["median_relative_error_if_rate_kg_ha_h"] > 0.999
