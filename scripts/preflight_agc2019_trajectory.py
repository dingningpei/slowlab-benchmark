#!/usr/bin/env python3
"""Audit one calibration-only AGC trajectory before physical-model replay.

This is an input/identifiability gate, not a simulator or a validation score.
It deliberately emits no model error when no AGC-driven prediction exists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowlab.greenhouse_data import excel_datetime


COMPARTMENTS = frozenset({
    "AICU", "Automatoes", "Digilog", "IUACAAS", "Reference", "TheAutomators"
})
CLIMATE_FIELDS = ("air_temperature_c", "relative_humidity_pct", "co2_ppm")
WEATHER_FIELDS = ("Iglob", "Tout", "Rhout", "Windsp", "Pyrgeo")
ACTION_FIELDS = (
    "t_heat_sp", "t_heat_vip", "t_vent_sp", "t_ventlee_vip",
    "t_ventwind_vip", "co2_sp", "co2_vip", "scr_enrg_sp",
    "scr_enrg_vip", "scr_blck_sp", "scr_blck_vip",
    "int_white_sp", "int_white_vip", "water_sup_intervals_sp_min",
    "water_sup_intervals_vip_min",
)
RESOURCE_FIELDS = ("Heat_cons", "ElecHigh", "ElecLow", "CO2_cons", "Irr", "Drain")
HARVEST_FIELDS = ("ProdA", "ProdB")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def number(value: str | None) -> float | None:
    try:
        result = float(value) if value is not None else math.nan
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def counts(path: Path, start: date, end: date, time_field: str,
           fields: tuple[str, ...], *, excel_time: bool) -> dict:
    present = Counter()
    rows = 0
    first = last = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set((time_field, *fields)) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} missing fields {sorted(missing)}")
        for row in reader:
            raw = number(row[time_field]) if excel_time else row[time_field]
            if raw is None or not raw:
                continue
            stamp = excel_datetime(raw).date() if excel_time else date.fromisoformat(raw[:10])
            if not start <= stamp < end:
                continue
            rows += 1
            first = stamp.isoformat() if first is None else first
            last = stamp.isoformat()
            for field in fields:
                present[field] += number(row[field]) is not None
    return {"rows": rows, "first_date": first, "last_date": last,
            "valid": {field: present[field] for field in fields}}


def excel_serials(path: Path, start: date, end: date) -> list[float]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        serials = []
        for row in csv.DictReader(handle):
            value = number(row.get("%time"))
            if value is not None and start <= excel_datetime(value).date() < end:
                serials.append(value)
    return serials


def build(source: Path, clean: Path, manifest_path: Path,
          compartment: str, day: date) -> dict:
    if compartment not in COMPARTMENTS:
        raise ValueError(f"unknown compartment: {compartment}")
    manifest = json.loads(manifest_path.read_text())
    if not date.fromisoformat(manifest["calibration_start"]) <= day < date.fromisoformat(manifest["holdout_start"]):
        raise ValueError("preflight day must be in frozen calibration period")
    end = day + timedelta(days=1)
    source_hashes = manifest["compartments"][compartment]["source_sha256"]
    clean_files = manifest["compartments"][compartment]["prepared"]
    for name, expected in source_hashes.items():
        if digest(source / compartment / name) != expected:
            raise ValueError(f"source hash mismatch: {name}")
    for name, info in clean_files.items():
        if digest(clean / compartment / name) != info["sha256"]:
            raise ValueError(f"clean hash mismatch: {name}")
    if digest(source / "Weather" / "Weather.csv") != manifest["weather_sha256"]:
        raise ValueError("weather hash mismatch")

    weather = counts(source / "Weather" / "Weather.csv", day, end, "%time", WEATHER_FIELDS, excel_time=True)
    climate = counts(clean / compartment / "climate_observations.csv", day, end,
                     "timestamp", CLIMATE_FIELDS, excel_time=False)
    actions = counts(source / compartment / "GreenhouseClimate.csv", day, end,
                     "%time", ACTION_FIELDS, excel_time=True)
    resources = counts(source / compartment / "Resources.csv", day, end,
                       "%Time ", RESOURCE_FIELDS, excel_time=True)
    harvest = counts(source / compartment / "Production.csv", day, end,
                     "%time", HARVEST_FIELDS, excel_time=True)
    pumps = counts(clean / compartment / "pump_minutes_events.csv", day, end,
                   "timestamp", ("pump_minutes",), excel_time=False)
    weather_times = excel_serials(source / "Weather" / "Weather.csv", day, end)
    climate_times = excel_serials(source / compartment / "GreenhouseClimate.csv", day, end)
    exact_clock_match = weather_times == climate_times and len(set(weather_times)) == len(weather_times)
    result = {
        "status": "input_preflight_only",
        "compartment": compartment, "source_clock_day": day.isoformat(),
        "manifest_sha256": digest(manifest_path),
        "coverage": {"weather": weather, "climate": climate, "actions": actions,
                     "resources": resources, "harvest": harvest, "pump_events": pumps},
        "source_clock_alignment": {
            "weather_and_compartment_excel_serials_identical": exact_clock_match,
            "weather_unique_serials": len(set(weather_times)),
            "compartment_unique_serials": len(set(climate_times)),
            "note": "Same source serials do not establish UTC or GreenLight clock mapping.",
        },
        "missing_action_samples": {
            field: actions["rows"] - actions["valid"][field] for field in ACTION_FIELDS
        },
        "prediction_errors": {field: None for field in (
            "temperature", "relative_humidity", "co2", "resources", "harvest")},
        "replay_gates": {
            "weather_clock": "unresolved: source timestamps are timezone-naive and DST mapping is not frozen",
            "facility": "unresolved: GreenLight Katzin-2021 geometry/equipment are not AGC compartment parameters",
            "actuation": "unresolved: historical realised settings do not yet map to GreenLight physical actuator inputs",
            "resources": "unresolved: independent pump-flow/area calibration and energy meter mapping",
            "harvest": "unresolved: crop initial state, fruit dry matter and harvest-date observation mapping",
        },
        "interpretation": "Input availability only; no AGC-driven GreenLight run or validation error.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--compartment", default="Reference")
    parser.add_argument("--day", type=date.fromisoformat, default=date(2020, 3, 18))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source, args.clean, args.manifest, args.compartment, args.day)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
