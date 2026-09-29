#!/usr/bin/env python3
"""Audit AGC crop observations available before a calibration replay starts."""
from __future__ import annotations

import argparse
import csv
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path


EXCEL_EPOCH = date(1899, 12, 30)
FIELDS = ("Stem_elong", "Stem_thick", "Cum_trusses", "stem_dens", "plant_dens")


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


def latest_at_or_before(path: Path, day: date) -> dict:
    target = (day - EXCEL_EPOCH).days
    latest = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            row = {key.strip(): value for key, value in raw.items() if key is not None}
            stamp = parsed(row.get("%Time"))
            if stamp is None:
                raise ValueError(f"invalid CropParameters timestamp in {path}")
            if stamp <= target and (latest is None or stamp > latest[0]):
                latest = (stamp, row)
    if latest is None:
        raise ValueError(f"no crop observation available before {day} in {path}")
    stamp, row = latest
    return {
        "source_excel_day": stamp,
        "source_date": (EXCEL_EPOCH + timedelta(days=stamp)).isoformat(),
        "age_days_at_replay": target - stamp,
        "observed": {field: parsed(row.get(field)) for field in FIELDS},
    }


def audit(source: Path, manifest_path: Path, day: date) -> dict:
    manifest = json.loads(manifest_path.read_text())
    start = date.fromisoformat(manifest["calibration_start"])
    stop = date.fromisoformat(manifest["holdout_start"])
    if not start <= day < stop:
        raise ValueError("crop-state audit is restricted to the frozen calibration period")
    compartments = {}
    for name, entry in manifest["compartments"].items():
        path = source / name / "CropParameters.csv"
        if digest(path) != entry["source_sha256"]["CropParameters.csv"]:
            raise ValueError(f"CropParameters hash differs from frozen manifest: {name}")
        compartments[name] = latest_at_or_before(path, day)
    return {
        "status": "observed_crop_context_not_greenlight_state_identification",
        "replay_day": day.isoformat(),
        "manifest_sha256": digest(manifest_path),
        "compartments": compartments,
        "unobserved_greenlight_states": [
            "cBuf", "cLeaf", "cStem", "cFruit", "tCanSum", "tCan24",
            "canopy temperature and leaf area index",
        ],
        "warning": "Stem growth, thickness, truss count and density show crop maturity but do not uniquely identify GreenLight biomass/carbohydrate states or LAI.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--day", required=True, type=date.fromisoformat)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.source, args.manifest, args.day)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
