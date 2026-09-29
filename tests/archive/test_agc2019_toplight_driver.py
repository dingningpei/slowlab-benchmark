import csv

import pytest

from scripts.agc.export_agc2019_toplight_driver import OUTPUT_FIELDS, build
from slowlab.archive.agc_lighting import LED_FIELDS


def write_trace(path, *, missing_on=False):
    fields = ("Time", "AssimLight", *LED_FIELDS)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"Time": 0, "AssimLight": 0, **{field: "" for field in LED_FIELDS}})
        writer.writerow({
            "Time": 300,
            "AssimLight": 100,
            **{field: "" if missing_on and field == "int_white_vip" else 1000 for field in LED_FIELDS},
        })


def test_driver_has_greenlight_metadata_and_keeps_two_overhead_sources(tmp_path):
    trace = tmp_path / "trace.csv"
    out = tmp_path / "driver.csv"
    write_trace(trace)
    audit = build(trace, out)
    assert not audit["greenlight_interlighting_used"]
    with out.open() as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == OUTPUT_FIELDS
    assert rows[1]["qLedProcessed"] == "W m**-2"
    assert float(rows[2]["uHpsObserved"]) == 0
    assert float(rows[3]["ledParPhotonFlux"]) == 97
    assert float(rows[3]["ledFarRedPhotonFlux"]) == 12


def test_driver_refuses_unknown_led_during_hps_operation(tmp_path):
    trace = tmp_path / "trace.csv"
    write_trace(trace, missing_on=True)
    with pytest.raises(ValueError, match="row 3.*unknown while HPS is powered"):
        build(trace, tmp_path / "driver.csv")
