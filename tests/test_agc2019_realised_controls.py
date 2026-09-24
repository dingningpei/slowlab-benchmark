import csv
import hashlib
import json
from datetime import date

import pytest

from scripts.export_agc2019_realised_controls import (
    OBSERVED_ACTUATORS, PROCESSED_RATES, REALISED_SETPOINTS, build,
)


def test_control_export_keeps_unknown_led_and_rejects_clock_mismatch(tmp_path):
    source = tmp_path / "GreenhouseClimate.csv"
    weather = tmp_path / "Weather.csv"
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("%time", *OBSERVED_ACTUATORS,
                                                   *PROCESSED_RATES, *REALISED_SETPOINTS))
        writer.writeheader()
        for index in range(289):
            row = {field: 0 for field in (*OBSERVED_ACTUATORS, *PROCESSED_RATES,
                                          *REALISED_SETPOINTS)}
            row.update({"%time": f"{43815 + index / 288:.8f}",
                        "int_white_vip": "NaN" if index == 1 else 0})
            writer.writerow(row)
    with weather.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("%time",))
        for index in range(289):
            writer.writerow((f"{43815 + index / 288:.8f}",))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "calibration_start": "2019-12-16", "holdout_start": "2020-04-01",
        "weather_sha256": digest(weather),
        "compartments": {"Reference": {"source_sha256": {
            "GreenhouseClimate.csv": digest(source)}}},
    }))
    out = tmp_path / "trace.csv"
    audit = build(source, weather, manifest, "Reference", date(2019, 12, 16), out)
    assert audit["missing_or_nan_by_field"]["int_white_vip"] == 1
    with out.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["int_white_vip"] == ""
    weather.write_text("%time\n43815\n")
    manifest.write_text(json.dumps({
        "calibration_start": "2019-12-16", "holdout_start": "2020-04-01",
        "weather_sha256": digest(weather),
        "compartments": {"Reference": {"source_sha256": {
            "GreenhouseClimate.csv": digest(source)}}},
    }))
    with pytest.raises(ValueError, match="not exactly aligned"):
        build(source, weather, manifest, "Reference", date(2019, 12, 16), out)
