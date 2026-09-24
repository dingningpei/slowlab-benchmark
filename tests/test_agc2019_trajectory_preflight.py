import json
from datetime import date

import pytest

from scripts.preflight_agc2019_trajectory import build, counts


def test_preflight_counts_only_finite_values_inside_day(tmp_path):
    path = tmp_path / "observations.csv"
    path.write_text(
        "timestamp,temperature,co2\n"
        "2020-03-14T23:55:00,19,500\n"
        "2020-03-15T00:00:00,20,NaN\n"
        "2020-03-15T00:05:00,21,600\n"
        "2020-03-16T00:00:00,22,700\n"
    )
    result = counts(path, date(2020, 3, 15), date(2020, 3, 16),
                    "timestamp", ("temperature", "co2"), excel_time=False)
    assert result["rows"] == 2
    assert result["valid"] == {"temperature": 2, "co2": 1}


def test_preflight_rejects_holdout_before_reading_source(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"calibration_start": "2019-12-16",
                                    "holdout_start": "2020-04-01"}))
    with pytest.raises(ValueError, match="frozen calibration period"):
        build(tmp_path, tmp_path, manifest, "Reference", date(2020, 4, 1))
