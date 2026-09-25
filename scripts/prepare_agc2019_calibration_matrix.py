#!/usr/bin/env python3
"""Prepare every input-complete AGC calibration compartment-day.

Selection uses the frozen input-coverage audit only.  No climate outcome or
model error is read while constructing this matrix.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_agc2019_greenlight_controls import build as build_driver  # noqa: E402
from scripts.export_agc2019_realised_controls import build as build_trace  # noqa: E402
from slowlab.agc_greenlight_weather import (  # noqa: E402
    WeatherAssumptions, make_weather_rows, write_greenlight_weather,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_identities(coverage: dict) -> list[tuple[str, date]]:
    identities = []
    for compartment, summary in sorted(coverage["summary"].items()):
        declared = summary["input_eligible_dates"]
        actual = [day for day, record in coverage["daily"][compartment].items()
                  if record["input_eligible"]]
        if declared != actual:
            raise ValueError(f"coverage summary mismatch: {compartment}")
        identities.extend((compartment, date.fromisoformat(day)) for day in declared)
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate calibration identity")
    return identities



def consecutive_sequences(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["compartment"], []).append(record)
    sequences = []
    for compartment, items in sorted(grouped.items()):
        items.sort(key=lambda item: item["day"])
        current = []
        for item in items:
            if current and date.fromisoformat(item["day"]) != date.fromisoformat(current[-1]["day"]) + timedelta(days=1):
                sequences.append({"compartment": compartment, "days": current})
                current = []
            current.append(item)
        if current:
            sequences.append({"compartment": compartment, "days": current})
    for index, sequence in enumerate(sequences):
        sequence["sequence_id"] = f"{sequence['compartment']}:{sequence['days'][0]['day']}:{len(sequence['days'])}d"
        sequence["day_count"] = len(sequence["days"])
    return sequences


def prepare(source: Path, manifest_path: Path, coverage_path: Path, out: Path,
            assumptions: WeatherAssumptions, *, period: str = "calibration") -> dict:
    manifest = json.loads(manifest_path.read_text())
    coverage = json.loads(coverage_path.read_text())
    if coverage["holdout_start"] != manifest["holdout_start"]:
        raise ValueError("manifest/coverage split mismatch")
    weather_source = source / "Weather" / "Weather.csv"
    if digest(weather_source) != manifest["weather_sha256"]:
        raise ValueError("weather source hash mismatch")
    identities = selected_identities(coverage)
    out.mkdir(parents=True, exist_ok=True)
    weather_files = {}
    records = []
    for compartment, day in identities:
        day_text = day.isoformat()
        if day_text not in weather_files:
            path = out / "weather" / f"{day_text}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            rows = make_weather_rows(weather_source, day, assumptions)
            write_greenlight_weather(path, rows)
            weather_files[day_text] = {"path": str(path), "sha256": digest(path), "rows": len(rows)}
        directory = out / "controls" / compartment
        directory.mkdir(parents=True, exist_ok=True)
        trace = directory / f"{day_text}_trace.csv"
        driver = directory / f"{day_text}_greenlight.csv"
        trace_audit = build_trace(source / compartment / "GreenhouseClimate.csv",
                                  weather_source, manifest_path, compartment, day, trace, period=period)
        driver_audit = build_driver(trace, driver)
        records.append({
            "compartment": compartment, "day": day_text,
            "weather": weather_files[day_text],
            "trace": {"path": str(trace), "sha256": trace_audit["export_sha256"]},
            "driver": {"path": str(driver), "sha256": driver_audit["output_sha256"]},
        })
    sequences = consecutive_sequences(records)
    return {
        "status": "calibration_inputs_only_no_outcomes_read",
        "identities": records,
        "continuous_sequences": sequences,
        "sequence_count": len(sequences),
        "identity_count": len(records),
        "distinct_weather_days": len(weather_files),
        "holdout_start_not_read": manifest["holdout_start"],
        "source_manifest_sha256": digest(manifest_path),
        "coverage_audit_sha256": digest(coverage_path),
        "weather_assumptions": assumptions.__dict__,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--outdoor-co2-ppm", type=float, default=410)
    parser.add_argument("--deep-soil-temperature-c", type=float, default=10)
    parser.add_argument("--elevation-m", type=float, default=0)
    parser.add_argument("--pyrgeometer-body-minus-air-c", type=float, default=0)
    args = parser.parse_args()
    assumptions = WeatherAssumptions(args.outdoor_co2_ppm, args.deep_soil_temperature_c,
                                     args.elevation_m, args.pyrgeometer_body_minus_air_c)
    result = prepare(args.source, args.manifest, args.coverage, args.out_dir, assumptions)
    args.matrix.parent.mkdir(parents=True, exist_ok=True)
    args.matrix.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"identity_count": result["identity_count"],
                      "sequence_count": result["sequence_count"],
                      "distinct_weather_days": result["distinct_weather_days"]}, indent=2))


if __name__ == "__main__": main()
