#!/usr/bin/env python3
"""Audit calibration-period CO2 dosing against AGC2's published supply limit.

This is a capacity plausibility check, not an independent flow-meter validation.
It never inspects the frozen holdout period or silently clips source values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowlab.archive.greenhouse_data import COMPARTMENTS, excel_datetime  # noqa: E402

CAP_KG_M2_H = 0.015  # Hemming et al. 2020: 15 g m^-2 h^-1.
FLOOR_AREA_M2 = 96
PAPER_CROP_AREA_M2 = 76.8
README_GROWING_AREA_M2 = 62.5
CAP_FLOOR_IF_PAPER_CROP_BASIS = CAP_KG_M2_H * PAPER_CROP_AREA_M2 / FLOOR_AREA_M2
CAP_FLOOR_IF_README_GROWING_BASIS = CAP_KG_M2_H * README_GROWING_AREA_M2 / FLOOR_AREA_M2


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def audit_file(path: Path, start: date, end: date) -> dict:
    samples = 0
    valid = 0
    missing = 0
    negative = 0
    above = []
    above_paper_crop_basis = 0
    above_readme_growing_basis = 0
    max_valid_rate = 0.0
    daily = defaultdict(lambda: {"samples": 0, "above_cap": 0,
                                  "max_kg_m2_h": 0.0,
                                  "excess_kg_m2": 0.0})
    previous = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            stamp = excel_datetime(float(row["%time"]))
            if not start <= stamp.date() < end:
                continue
            samples += 1
            raw = row.get("co2_dos", "")
            try:
                value = float(raw)
            except (TypeError, ValueError):
                missing += 1
                continue
            if not math.isfinite(value):
                missing += 1
                continue
            if value < 0:
                negative += 1
                continue
            valid += 1
            max_valid_rate = max(max_valid_rate, value)
            above_paper_crop_basis += value > CAP_FLOOR_IF_PAPER_CROP_BASIS + 1e-12
            above_readme_growing_basis += value > CAP_FLOOR_IF_README_GROWING_BASIS + 1e-12
            day = stamp.date().isoformat()
            daily[day]["samples"] += 1
            daily[day]["max_kg_m2_h"] = max(daily[day]["max_kg_m2_h"], value)
            if value > CAP_KG_M2_H + 1e-12:
                daily[day]["above_cap"] += 1
                # Five-minute samples are rates, not instantaneous pulses.
                daily[day]["excess_kg_m2"] += (value - CAP_KG_M2_H) / 12
                above.append({"source_time": stamp.isoformat(sep=" "),
                              "co2_dos_kg_m2_h": value,
                              "multiple_of_published_capacity": value / CAP_KG_M2_H})
            previous = stamp
    return {
        "samples": samples, "valid_nonnegative_samples": valid,
        "missing_or_nonfinite_samples": missing, "negative_samples": negative,
        "above_capacity_samples": len(above),
        "max_valid_rate_kg_m2_h": max_valid_rate,
        "above_floor_equivalent_if_paper_crop_basis": above_paper_crop_basis,
        "above_floor_equivalent_if_readme_growing_basis": above_readme_growing_basis,
        "above_capacity_fraction_of_valid": len(above) / valid if valid else None,
        "days_with_above_capacity": {day: data for day, data in daily.items()
                                     if data["above_cap"]},
        "above_capacity_rows": above,
        "last_calibration_timestamp": previous.isoformat(sep=" ") if previous else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    start = date.fromisoformat(manifest["calibration_start"])
    end = date.fromisoformat(manifest["holdout_start"])
    compartments = {}
    for name in COMPARTMENTS:
        path = args.source / name / "GreenhouseClimate.csv"
        expected = manifest["compartments"][name]["source_sha256"]["GreenhouseClimate.csv"]
        if digest(path) != expected:
            raise ValueError(f"source hash mismatch: {name}")
        compartments[name] = audit_file(path, start, end)
    result = {
        "status": "calibration_only_capacity_plausibility_audit",
        "unit_hypothesis": "kg/m2/hour, based on same-source daily CO2 ledger",
        "published_capacity_kg_m2_h": CAP_KG_M2_H,
        "unverified_area_denominator_hypotheses": {
            "floor_equivalent_if_paper_crop_area_76_8_m2": CAP_FLOOR_IF_PAPER_CROP_BASIS,
            "floor_equivalent_if_readme_growing_area_62_5_m2": CAP_FLOOR_IF_README_GROWING_BASIS,
            "note": "Neither source proves which area denominator the published cap or processed co2_dos uses. These are comparisons only, not conversions applied to data.",
        },
        "period_start_inclusive": start.isoformat(),
        "period_end_exclusive": end.isoformat(),
        "compartments": compartments,
        "warning": "Exceedances are preserved; source ledger is not independent meter validation.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {"above_capacity_samples": data["above_capacity_samples"],
                             "days_with_above_capacity": len(data["days_with_above_capacity"])}
                      for name, data in compartments.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
