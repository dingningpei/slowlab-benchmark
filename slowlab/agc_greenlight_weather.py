"""Convert one AGC source-clock weather day to GreenLight input format.

The output is a forcing-interface diagnostic. Unmeasured outdoor CO2, deep-soil
temperature and elevation must be supplied explicitly and are not validation
facts about the AGC facility.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .greenhouse_data import excel_datetime


FIELDS = (
    "Time", "tOut", "vpOut", "co2Out", "wind", "tSky", "tSoOut",
    "iGlob", "dayRadSum", "isDay", "isDaySmooth", "hElevation",
)
UNITS = (
    "s", "°C", "Pa", "mg m**-3", "m s**-1", "°C", "°C",
    "W m**-2", "MJ m**-2", "-", "-", "m above sea level",
)
DESCRIPTIONS = (
    "Time since source-clock midnight", "Observed outdoor air temperature",
    "Outdoor vapor pressure derived from observed RH and temperature",
    "Outdoor CO2 density from explicit assumed ppm", "Observed wind speed",
    "Sky temperature under net-Pyrgeo and assumed sensor-body temperature",
    "Explicit assumed deep-soil temperature", "Observed global solar radiation",
    "Observed-so-far global radiation integral; not a future daily total",
    "Day indicator from observed solar radiation", "Same sampled indicator",
    "Explicit assumed site elevation",
)


@dataclass(frozen=True)
class WeatherAssumptions:
    outdoor_co2_ppm: float
    deep_soil_temperature_c: float
    elevation_m: float
    pyrgeometer_body_minus_air_c: float

    def validate(self) -> None:
        limits = ((self.outdoor_co2_ppm, 250, 1000),
                  (self.deep_soil_temperature_c, -20, 40),
                  (self.elevation_m, -500, 9000),
                  (self.pyrgeometer_body_minus_air_c, -15, 15))
        if any(not math.isfinite(v) or not lo <= v <= hi for v, lo, hi in limits):
            raise ValueError("explicit GreenLight weather assumptions outside plausible ranges")


def _finite(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"missing or invalid AGC weather field {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite AGC weather field {field}")
    return value


def make_weather_rows(source: Path, day: date,
                      assumptions: WeatherAssumptions) -> list[tuple[float, ...]]:
    assumptions.validate()
    end = day + timedelta(days=1)
    selected: list[tuple[float, dict[str, str]]] = []
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            serial = _finite(row, "%time")
            stamp = excel_datetime(serial).date()
            if stamp == day or (stamp == end and abs(serial - round(serial)) < 1e-8):
                selected.append((serial, row))
    if len(selected) < 2:
        raise ValueError("AGC day has insufficient weather observations")
    source_midnight = float((day - date(1899, 12, 30)).days)
    times = [(serial - source_midnight) * 86400 for serial, _ in selected]
    if abs(times[0]) > 1 or abs(times[-1] - 86400) > 1:
        raise ValueError("AGC weather day must include both midnight endpoints")
    if any(not 0 < b - a <= 360 for a, b in zip(times, times[1:])):
        raise ValueError("AGC weather timestamps duplicate, regress, or have >6-minute gap")
    result = []
    accumulated_mj = 0.0
    previous_time = times[0]
    previous_iglob = 0.0
    for index, ((_, row), seconds) in enumerate(zip(selected, times)):
        t_out = _finite(row, "Tout")
        rh = _finite(row, "Rhout")
        wind = _finite(row, "Windsp")
        pyrgeo = _finite(row, "Pyrgeo")
        iglob = _finite(row, "Iglob")
        if not (0 <= rh <= 100 and wind >= 0 and iglob >= 0):
            raise ValueError("out-of-range AGC weather observation")
        if index:
            accumulated_mj += previous_iglob * (seconds - previous_time) * 1e-6
        if seconds >= 86400 - 1:
            accumulated_mj = 0.0  # Next source-clock day starts here.
        vp_out = rh / 100 * 610.78 * math.exp(17.2694 * t_out / (t_out + 238.3))
        co2_out = (101325 * 1e-6 * assumptions.outdoor_co2_ppm * 44.01e-3
                   / (8.314 * (t_out + 273.15))) * 1e6
        # AGC labels Pyrgeo "heat emission" and its negative values rule out
        # interpreting it as absolute downwelling irradiance. Under the explicit
        # net-flux hypothesis, the sensor-body emission is added back. AGC does
        # not publish body temperature, so this is a sensitivity assumption.
        body_k = t_out + assumptions.pyrgeometer_body_minus_air_c + 273.15
        downward_longwave = pyrgeo + 5.6697e-8 * body_k**4
        if downward_longwave <= 0:
            raise ValueError("net pyrgeometer hypothesis gave nonpositive sky irradiance")
        sky_c = (downward_longwave / 5.6697e-8) ** 0.25 - 273.15
        is_day = float(iglob > 0)
        result.append((seconds, t_out, vp_out, co2_out, wind, sky_c,
                       assumptions.deep_soil_temperature_c, iglob, accumulated_mj,
                       is_day, is_day, assumptions.elevation_m))
        previous_time, previous_iglob = seconds, iglob
    return result


def write_greenlight_weather(path: Path, rows: list[tuple[float, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerow(DESCRIPTIONS)
        writer.writerow(UNITS)
        writer.writerows(rows)
