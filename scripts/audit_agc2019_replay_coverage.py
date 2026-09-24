#!/usr/bin/env python3
"""Inventory calibration-day replay inputs without looking at model errors."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path


WEATHER = ("Tout", "Rhout", "Windsp", "Pyrgeo", "Iglob")
ACTUATORS = ("VentLee", "Ventwind", "EnScr", "BlackScr", "AssimLight",
             "PipeLow", "PipeGrow")
LED = ("int_blue_vip", "int_red_vip", "int_farred_vip", "int_white_vip")
LED_PROCESSED_W_M2 = {"int_blue_vip": 7.27, "int_red_vip": 25.3,
                      "int_farred_vip": 6.23, "int_white_vip": 22.72}
OBSERVATIONS = ("Tair", "Rhair", "CO2air")
EXCEL_EPOCH = date(1899, 12, 30)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def number(row: dict[str, str], field: str) -> float | None:
    try:
        value = float(row[field])
    except (KeyError, ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def calibration_rows(path: Path, start: date, stop: date) -> dict[date, list[tuple[float, dict[str, str]]]]:
    start_serial = (start - EXCEL_EPOCH).days
    stop_serial = (stop - EXCEL_EPOCH).days
    result = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            serial = number(row, "%time")
            if serial is None:
                raise ValueError(f"invalid timestamp in {path}")
            if serial >= stop_serial:
                break  # Never inspect holdout row values.
            if serial >= start_serial:
                result[EXCEL_EPOCH + timedelta(days=math.floor(serial))].append((serial, row))
    return result


def classify(weather: list[tuple[float, dict[str, str]]],
             climate: list[tuple[float, dict[str, str]]]) -> dict:
    weather_times = [stamp for stamp, _ in weather]
    climate_times = [stamp for stamp, _ in climate]
    aligned = (len(weather_times) == len(climate_times) == 289
               and weather_times == climate_times
               and all(0 < b - a <= 360 / 86400 + 1e-8
                       for a, b in zip(weather_times, weather_times[1:])))
    weather_bad = 0
    weather_any_bad = False
    weather_bad_fields = {field: 0 for field in WEATHER}
    for row_index, (_, row) in enumerate(weather):
        values = {field: number(row, field) for field in WEATHER}
        invalid = {field for field, value in values.items()
                   if value is None or (field == "Rhout" and not 0 <= value <= 100)
                   or (field in ("Windsp", "Iglob") and value < 0)}
        weather_any_bad |= bool(invalid)
        if row_index < 288:
            weather_bad += bool(invalid)
            for field in invalid:
                weather_bad_fields[field] += 1
    actuator_bad = 0
    actuator_any_bad = False
    actuator_bad_fields = {field: 0 for field in ACTUATORS}
    co2_bad = 0
    co2_any_bad = False
    led_ambiguous = 0
    led_any_ambiguous = False
    led_unknown_run = 0
    led_unknown_max_run = 0
    led_unknown_runs = 0
    led_known_energy_kwh_m2 = 0.0
    led_unknown_upper_energy_kwh_m2 = 0.0
    led_ambiguous_fields = {field: 0 for field in LED}
    led_missing_while_hps_off = {field: 0 for field in LED}
    climate_observation_bad = 0
    observation_bad_fields = {field: 0 for field in OBSERVATIONS}
    for row_index, (_, row) in enumerate(climate):
        core_row = row_index < 288
        actuators = {field: number(row, field) for field in ACTUATORS}
        invalid = {field for field, value in actuators.items()
                   if value is None or
                   (field in ("VentLee", "Ventwind", "EnScr", "BlackScr", "AssimLight")
                    and not 0 <= value <= 100) or
                   (field in ("PipeLow", "PipeGrow") and value < 0)}
        actuator_any_bad |= bool(invalid)
        if core_row:
            actuator_bad += bool(invalid)
            for field in invalid:
                actuator_bad_fields[field] += 1
        co2 = number(row, "co2_dos")
        bad_co2 = co2 is None or co2 < 0
        co2_any_bad |= bad_co2
        if core_row:
            co2_bad += bad_co2
        hps = number(row, "AssimLight")
        led_values = {field: number(row, field) for field in LED}
        unknown = {field for field, value in led_values.items()
                   if value is None or not 0 <= value <= 1000}
        if hps is not None and hps > 0:
            led_any_ambiguous |= bool(unknown)
            if core_row:
                led_ambiguous += bool(unknown)
                for field in unknown:
                    led_ambiguous_fields[field] += 1
                for field, value in led_values.items():
                    if field in unknown:
                        led_unknown_upper_energy_kwh_m2 += LED_PROCESSED_W_M2[field] / 12000
                    else:
                        led_known_energy_kwh_m2 += LED_PROCESSED_W_M2[field] * value / 1000 / 12000
        elif hps == 0 and core_row:
            for field in unknown:
                led_missing_while_hps_off[field] += 1
        if core_row and hps is not None and hps > 0 and unknown:
            if led_unknown_run == 0:
                led_unknown_runs += 1
            led_unknown_run += 1
            led_unknown_max_run = max(led_unknown_max_run, led_unknown_run)
        else:
            led_unknown_run = 0
        outcomes = {field: number(row, field) for field in OBSERVATIONS}
        invalid = {field for field, value in outcomes.items()
                   if value is None or (field == "Rhair" and not 0 <= value <= 100)
                   or (field == "CO2air" and value < 0)}
        if core_row:
            climate_observation_bad += bool(invalid)
            for field in invalid:
                observation_bad_fields[field] += 1
    eligible = (aligned and not weather_any_bad and not actuator_any_bad
                and not co2_any_bad and not led_any_ambiguous)
    return {
        "weather_rows": len(weather), "climate_rows": len(climate),
        "exact_288_grid_alignment": aligned,
        "weather_invalid_rows": weather_bad,
        "weather_invalid_by_field": weather_bad_fields,
        "actuator_invalid_rows": actuator_bad,
        "actuator_invalid_by_field": actuator_bad_fields,
        "co2_dose_invalid_rows": co2_bad,
        "led_unknown_while_hps_on_rows": led_ambiguous,
        "led_unknown_while_hps_on_runs": led_unknown_runs,
        "led_unknown_while_hps_on_max_consecutive_rows": led_unknown_max_run,
        "led_known_processing_energy_kwh_m2": led_known_energy_kwh_m2,
        "led_unknown_processing_energy_upper_kwh_m2": led_unknown_upper_energy_kwh_m2,
        "led_unknown_while_hps_on_by_field": led_ambiguous_fields,
        "led_missing_while_hps_off_by_field": led_missing_while_hps_off,
        "climate_observation_invalid_rows_not_used_for_input_eligibility": climate_observation_bad,
        "climate_observation_invalid_by_field_not_used_for_input_eligibility": observation_bad_fields,
        "input_eligible": eligible,
    }


def audit(source: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    start = date.fromisoformat(manifest["calibration_start"])
    stop = date.fromisoformat(manifest["holdout_start"])
    weather_path = source / "Weather" / "Weather.csv"
    if digest(weather_path) != manifest["weather_sha256"]:
        raise ValueError("weather hash differs from frozen manifest")
    weather = calibration_rows(weather_path, start, stop)
    days = [start + timedelta(days=index) for index in range((stop - start).days)]
    details = {}
    summary = {}
    for compartment, entry in manifest["compartments"].items():
        path = source / compartment / "GreenhouseClimate.csv"
        if digest(path) != entry["source_sha256"]["GreenhouseClimate.csv"]:
            raise ValueError(f"climate hash differs from frozen manifest: {compartment}")
        climate = calibration_rows(path, start, stop)
        details[compartment] = {}
        for day in days:
            # GreenLight interpolation needs the next-midnight endpoint. The
            # final calibration day remains ineligible because its endpoint is
            # the first holdout record, whose values must not be inspected.
            next_day = day + timedelta(days=1)
            weather_day = list(weather.get(day, []))
            climate_day = list(climate.get(day, []))
            if next_day < stop:
                if weather.get(next_day):
                    weather_day.append(weather[next_day][0])
                if climate.get(next_day):
                    climate_day.append(climate[next_day][0])
            details[compartment][day.isoformat()] = classify(weather_day, climate_day)
        summary[compartment] = {
            "calibration_days": len(days),
            "input_eligible_days": sum(item["input_eligible"] for item in details[compartment].values()),
            "input_eligible_dates": [day for day, item in details[compartment].items()
                                     if item["input_eligible"]],
            "days_with_unknown_led_while_hps_on": sum(
                item["led_unknown_while_hps_on_rows"] > 0
                for item in details[compartment].values()),
            "days_with_at_least_one_hour_continuous_unknown_led": sum(
                item["led_unknown_while_hps_on_max_consecutive_rows"] >= 12
                for item in details[compartment].values()),
            "longest_continuous_unknown_led_rows": max(
                item["led_unknown_while_hps_on_max_consecutive_rows"]
                for item in details[compartment].values()),
            "led_known_processing_energy_kwh_m2": sum(
                item["led_known_processing_energy_kwh_m2"]
                for item in details[compartment].values()),
            "led_unknown_processing_energy_upper_kwh_m2": sum(
                item["led_unknown_processing_energy_upper_kwh_m2"]
                for item in details[compartment].values()),
            "days_with_incomplete_grid_or_weather": sum(
                not item["exact_288_grid_alignment"] or item["weather_invalid_rows"] > 0
                for item in details[compartment].values()),
        }
    return {
        "status": "calibration_input_inventory_not_model_validation",
        "source_archive_md5": manifest["source_archive_md5"],
        "manifest_sha256": digest(manifest_path),
        "calibration_start": start.isoformat(), "holdout_start": stop.isoformat(),
        "eligibility_rule": "289 exactly aligned 5-minute weather/control endpoints from day midnight through next midnight; all required weather, observed actuator, processed CO2 dose, and LED VIP values while HPS is on must be present and in basic ranges. The last calibration day is ineligible because its next-midnight endpoint belongs to holdout and is not inspected. Climate outcome values and model errors are never used for input eligibility.",
        "led_off_semantics": "Missing LED VIP while HPS is off is tracked as structural/non-required under the documented interlock; it is not imputed as a measured zero.",
        "led_energy_caveat": "Energy bounds use official ReadMe's processed W/m2 coefficients and five-minute sampling, not an independent meter or photon/heat measurement. Unknown VIP values are bounded 0..1000 only while HPS is on. Entirely missing source rows are not included.",
        "summary": summary, "daily": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
