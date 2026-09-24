import csv
from datetime import date

import pytest

from slowlab.agc_greenlight_weather import (
    WeatherAssumptions, make_weather_rows, write_greenlight_weather,
)


def weather_file(path, pyrgeo=-80, include_endpoint=True):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("%time", "Tout", "Rhout", "Windsp", "Pyrgeo", "Iglob"))
        for index in range(289 if include_endpoint else 288):
            writer.writerow((f"{43815 + index / 288:.8f}", 10, 80, 2,
                             pyrgeo, 100 if 72 <= index < 216 else 0))


def test_negative_net_pyrgeo_requires_explicit_body_temperature_proxy(tmp_path):
    path = tmp_path / "Weather.csv"
    weather_file(path)
    assumptions = WeatherAssumptions(410, 10, 0, 0)
    rows = make_weather_rows(path, date(2019, 12, 16), assumptions)
    assert len(rows) == 289
    assert rows[0][0] == pytest.approx(0)
    assert rows[-1][0] == pytest.approx(86400)
    assert -25 < rows[0][5] < 10  # Physically plausible sky temperature.
    assert rows[216][8] > 0  # Only radiation already observed is integrated.
    assert rows[-1][8] == 0  # Next day's integral resets.
    output = tmp_path / "greenlight.csv"
    write_greenlight_weather(output, rows)
    with output.open() as handle:
        assert next(csv.reader(handle))[0:3] == ["Time", "tOut", "vpOut"]


def test_missing_day_endpoint_and_nonphysical_longwave_fail_closed(tmp_path):
    path = tmp_path / "Weather.csv"
    weather_file(path, include_endpoint=False)
    with pytest.raises(ValueError, match="both midnight endpoints"):
        make_weather_rows(path, date(2019, 12, 16), WeatherAssumptions(410, 10, 0, 0))
    weather_file(path, pyrgeo=-1000)
    with pytest.raises(ValueError, match="nonpositive sky irradiance"):
        make_weather_rows(path, date(2019, 12, 16), WeatherAssumptions(410, 10, 0, 0))
