#!/usr/bin/env python3
"""Freeze source hashes and non-outcome coverage for AGC trajectory replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


COMPARTMENTS = (
    "AICU", "Automatoes", "Digilog", "IUACAAS", "Reference", "TheAutomators"
)
CALIBRATION_START = "2019-12-16"
HOLDOUT_START = "2020-04-01"
# The last valid raw serial converts to 09:35:00.384; the next five-minute
# sample marks an unambiguous exclusive boundary without rounding away data.
ROOT_OBSERVATION_END_EXCLUSIVE = "2020-05-26T09:40:00"
SOURCE_FILES = (
    "GreenhouseClimate.csv", "GrodanSens.csv", "Resources.csv", "Production.csv",
    "CropParameters.csv",
)
CLEAN_FILES = (
    "root_zone.csv", "climate_observations.csv", "irrigation_events.csv",
    "pump_minutes_events.csv",
)
OFFICIAL_ARCHIVE_MD5 = "2a0c7f3332881caef54ca8f4dc60c9a3"
OBSERVATION_FIELDS = {
    "root_zone.csv": (
        "wc_slab1_pct", "wc_slab2_pct", "ec_slab1_ds_m", "ec_slab2_ds_m"
    ),
    "climate_observations.csv": (
        "air_temperature_c", "relative_humidity_pct", "co2_ppm"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()  # Official release checksum; not a security primitive.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coverage(path: Path) -> dict:
    fields = OBSERVATION_FIELDS.get(path.name, ())
    periods = {
        period: {"rows": 0, "valid_observations": {name: 0 for name in fields}}
        for period in ("calibration", "holdout", "after_root_sensor_end")
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            timestamp = row["timestamp"]
            if timestamp[:10] < CALIBRATION_START:
                raise ValueError(f"pre-campaign timestamp in {path}: {timestamp}")
            if timestamp[:10] < HOLDOUT_START:
                period = "calibration"
            elif timestamp < ROOT_OBSERVATION_END_EXCLUSIVE:
                period = "holdout"
            else:
                period = "after_root_sensor_end"
            periods[period]["rows"] += 1
            for name in fields:
                periods[period]["valid_observations"][name] += bool(row[name])
    return periods


def build(source: Path, clean: Path, archive: Path) -> dict:
    archive_md5 = md5(archive)
    if archive_md5 != OFFICIAL_ARCHIVE_MD5:
        raise ValueError(f"official archive MD5 mismatch: {archive_md5}")
    compartments = {}
    for name in COMPARTMENTS:
        raw = {file: sha256(source / name / file) for file in SOURCE_FILES}
        prepared = {
            file: {"sha256": sha256(clean / name / file), "coverage": coverage(clean / name / file)}
            for file in CLEAN_FILES
        }
        compartments[name] = {"source_sha256": raw, "prepared": prepared}
    return {
        "dataset": "Autonomous Greenhouse Challenge, Second Edition (2019)",
        "official_version": 2,
        "source_archive_md5": archive_md5,
        "calibration_start": CALIBRATION_START,
        "holdout_start": HOLDOUT_START,
        "root_observation_end_exclusive_source_clock": ROOT_OBSERVATION_END_EXCLUSIVE,
        "source_clock_timezone": None,
        "water_area_hypotheses_m2": (62.5, 76.8, 96.0),
        "warning": "Coverage and hashes only; no holdout outcomes or fitted parameters.",
        "compartments": compartments,
        "weather_sha256": sha256(source / "Weather" / "Weather.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--clean", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.source, args.clean, args.archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        name: {
            file: {period: values["rows"] for period, values in details["coverage"].items()}
            for file, details in compartment["prepared"].items()
        }
        for name, compartment in result["compartments"].items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
