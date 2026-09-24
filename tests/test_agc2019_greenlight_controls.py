import csv

import pytest

from scripts.export_agc2019_greenlight_controls import FIELDS, build
from slowlab.agc_lighting import LED_FIELDS


def make_trace(path, missing_led=False):
    fields = ("Time", "BlackScr", "EnScr", "VentLee", "Ventwind", "AssimLight",
              "co2_dos", "PipeLow", "PipeGrow", *LED_FIELDS)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for index in range(2):
            row = {field: 1000 for field in LED_FIELDS}
            if missing_led and index == 1: row["int_white_vip"] = ""
            row.update({"Time": index * 300, "BlackScr": 80, "EnScr": 60,
                        "VentLee": 20, "Ventwind": 40, "AssimLight": index * 100,
                        "co2_dos": .0072, "PipeLow": 45, "PipeGrow": 0})
            writer.writerow(row)


def test_export_maps_only_defensible_control_proxies(tmp_path):
    trace = tmp_path / "trace.csv"; out = tmp_path / "out.csv"; make_trace(trace)
    audit = build(trace, out)
    assert audit["pipe_fields_exported_as_temperature_targets_not_power"] == ["PipeLow", "PipeGrow"]
    with out.open() as handle: rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == FIELDS
    data = rows[3]
    assert float(data["uBlScr"]) == .8 and float(data["uThScr"]) == .6
    assert float(data["uRoof"]) == .3
    assert float(data["mcExtAir"]) == 2.0
    assert float(data["qHpsProcessed"]) == 81
    assert float(data["qLedProcessed"]) == pytest.approx(61.52)
    assert float(data["pipeLowTarget"]) == 45 and float(data["pipeLowActive"]) == 1
    assert float(data["pipeGrowTarget"]) == 0 and float(data["pipeGrowActive"]) == 0


def test_export_fails_on_unknown_powered_led(tmp_path):
    trace = tmp_path / "trace.csv"; make_trace(trace, missing_led=True)
    with pytest.raises(ValueError, match="unknown while HPS is powered"):
        build(trace, tmp_path / "out.csv")
