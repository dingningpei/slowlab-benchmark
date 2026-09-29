#!/usr/bin/env python3
"""Export AGC realised-control trace without inventing GreenLight actuators."""
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowlab.archive.greenhouse_data import excel_datetime  # noqa: E402


OBSERVED_ACTUATORS = (
    "VentLee", "Ventwind", "EnScr", "BlackScr", "AssimLight",
    "PipeLow", "PipeGrow",
)
PROCESSED_RATES = ("co2_dos",)
REALISED_SETPOINTS = (
    "t_heat_vip", "t_ventlee_vip", "t_ventwind_vip", "co2_vip",
    "scr_enrg_vip", "scr_blck_vip", "assim_vip",
    "int_blue_vip", "int_red_vip", "int_farred_vip", "int_white_vip",
    "water_sup_intervals_vip_min",
)
FIELDS = ("Time", "source_excel_time", *OBSERVED_ACTUATORS,
          *PROCESSED_RATES, *REALISED_SETPOINTS)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def parsed(value: str | None) -> float | None:
    try:
        result = float(value) if value is not None else math.nan
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def source_day_rows(path: Path, day: date) -> list[dict[str, str]]:
    end = day + timedelta(days=1)
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = set(("%time", *OBSERVED_ACTUATORS, *PROCESSED_RATES,
                        *REALISED_SETPOINTS))
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"AGC control file lacks {sorted(required - set(reader.fieldnames or ()))}")
        for row in reader:
            serial = parsed(row["%time"])
            if serial is None:
                continue
            stamp = excel_datetime(serial).date()
            if stamp == day or (stamp == end and abs(serial - round(serial)) < 1e-8):
                rows.append(row)
    return rows


def build(source: Path, weather: Path, manifest_path: Path,
          compartment: str, day: date, out: Path, *, period: str = "calibration") -> dict:
    manifest = json.loads(manifest_path.read_text())
    calibration_start=date.fromisoformat(manifest["calibration_start"]); holdout_start=date.fromisoformat(manifest["holdout_start"])
    if period == "calibration":
        allowed = calibration_start <= day < holdout_start
    elif period == "holdout":
        source_end=date.fromisoformat(manifest["root_observation_end_exclusive_source_clock"][:10])
        allowed = holdout_start <= day < source_end
    else:
        allowed = False
    if not allowed:
        raise ValueError(f"control export day outside explicit {period} period")
    if compartment not in manifest["compartments"]:
        raise ValueError("unknown AGC compartment")
    if digest(source) != manifest["compartments"][compartment]["source_sha256"]["GreenhouseClimate.csv"]:
        raise ValueError("AGC control source hash mismatch")
    if digest(weather) != manifest["weather_sha256"]:
        raise ValueError("AGC weather source hash mismatch")
    rows = source_day_rows(source, day)
    weather_rows = source_day_rows_weather(weather, day)
    serials = [float(row["%time"]) for row in rows]
    if serials != weather_rows or len(serials) != 289 or len(set(serials)) != len(serials):
        raise ValueError("control/weather source timestamps are not exactly aligned")
    source_midnight = float((day - date(1899, 12, 30)).days)
    missing = Counter()
    written = []
    for row in rows:
        serial = float(row["%time"])
        values = {"Time": (serial - source_midnight) * 86400,
                  "source_excel_time": serial}
        for field in (*OBSERVED_ACTUATORS, *PROCESSED_RATES, *REALISED_SETPOINTS):
            value = parsed(row[field])
            if value is None:
                missing[field] += 1
            values[field] = "" if value is None else value
        written.append(values)
    if any(written[0][field] == "" for field in OBSERVED_ACTUATORS):
        raise ValueError("initial observed actuator state is unknown")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(written)
    return {
        "status": "realised_control_trace_only",
        "compartment": compartment, "source_clock_day": day.isoformat(),
        "rows": len(written), "first_seconds": written[0]["Time"],
        "last_seconds": written[-1]["Time"],
        "source_sha256": digest(source), "weather_sha256": digest(weather),
        "export_sha256": digest(out),
        "missing_or_nan_by_field": {field: missing[field] for field in (*OBSERVED_ACTUATORS, *PROCESSED_RATES, *REALISED_SETPOINTS)},
        "semantics": "Observed equipment states, processed CO2 dosing, and realised VIP setpoints are retained separately; blank values remain unknown. No GreenLight actuation mapping is asserted.",
    }


def source_day_rows_weather(path: Path, day: date) -> list[float]:
    end = day + timedelta(days=1)
    selected = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            serial = parsed(row.get("%time"))
            if serial is None:
                continue
            stamp = excel_datetime(serial).date()
            if stamp == day or (stamp == end and abs(serial - round(serial)) < 1e-8):
                selected.append(serial)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--weather", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--compartment", default="Reference")
    parser.add_argument("--day", type=date.fromisoformat, default=date(2020, 3, 18))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    audit = build(args.source, args.weather, args.manifest,
                  args.compartment, args.day, args.out)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
