#!/usr/bin/env python3
"""Check AGC co2_dos unit hypotheses against same-source daily accounting.

This establishes internal consistency only: Resources CO2_cons is computed from
co2_dos and is not an independent meter validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowlab.archive.greenhouse_data import COMPARTMENTS, excel_datetime  # noqa: E402


def audit_compartment(root: Path, name: str) -> dict:
    daily_rate = defaultdict(float)
    samples = Counter()
    with (root / name / "GreenhouseClimate.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            day = excel_datetime(float(row["%time"])).date()
            if day.isoformat() >= "2020-04-01":
                continue
            try:
                rate = float(row["co2_dos"])
            except ValueError:
                continue
            if math.isfinite(rate) and rate >= 0:
                daily_rate[day] += rate / 12  # 5-minute samples of an hourly rate.
                samples[day] += 1
    observed = {}
    with (root / name / "Resources.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            day = excel_datetime(float(row["%Time "])).date()
            if day.isoformat() >= "2020-04-01" or samples[day] < 280:
                continue
            try:
                value = float(row["CO2_cons"])
            except ValueError:
                continue
            if math.isfinite(value) and value > 0:
                observed[day] = value
    errors_as_kg_m2_h = [abs(daily_rate[day] - value) / value
                          for day, value in observed.items()]
    errors_as_kg_ha_h = [abs(daily_rate[day] / 10000 - value) / value
                          for day, value in observed.items()]
    if not observed:
        raise ValueError(f"no complete calibration days for {name}")
    return {
        "days": len(observed),
        "median_relative_error_if_rate_kg_m2_h": statistics.median(errors_as_kg_m2_h),
        "median_relative_error_if_rate_kg_ha_h": statistics.median(errors_as_kg_ha_h),
        "mean_absolute_error_if_rate_kg_m2_h_kg_m2_day": statistics.mean(
            abs(daily_rate[day] - value) for day, value in observed.items()),
        "warning": "CO2_cons is computed from co2_dos; this is an internal unit check, not independent flux validation.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = {
        "period": "calibration_only_before_2020-04-01",
        "hypotheses": "ReadMe labels co2_dos kg/ha/hour but same-source CO2_cons is kg/m2/day",
        "compartments": {name: audit_compartment(args.source, name) for name in COMPARTMENTS},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
