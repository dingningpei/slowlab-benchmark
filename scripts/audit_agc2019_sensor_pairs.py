#!/usr/bin/env python3
"""Summarize paired slab-sensor disagreement in the calibration period only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


COMPARTMENTS = (
    "AICU", "Automatoes", "Digilog", "IUACAAS", "Reference", "TheAutomators"
)
CALIBRATION_END_EXCLUSIVE = "2020-04-01"
FIELDS = {
    "wc_pct": ("wc_slab1_pct", "wc_slab2_pct"),
    "ec_ds_m": ("ec_slab1_ds_m", "ec_slab2_ds_m"),
}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])


def audit(source: Path) -> dict:
    results = {}
    for compartment in COMPARTMENTS:
        differences = {key: [] for key in FIELDS}
        signed_differences = {key: [] for key in FIELDS}
        monthly_signed = {key: {} for key in FIELDS}
        total_calibration_rows = 0
        path = source / compartment / "root_zone.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["timestamp"][:10] >= CALIBRATION_END_EXCLUSIVE:
                    continue
                total_calibration_rows += 1
                for key, (left, right) in FIELDS.items():
                    if not row[left] or not row[right]:
                        continue
                    a, b = float(row[left]), float(row[right])
                    if math.isfinite(a) and math.isfinite(b):
                        differences[key].append(abs(a - b))
                        signed_differences[key].append(a - b)
                        month = row["timestamp"][:7]
                        monthly_signed[key].setdefault(month, []).append(a - b)
        results[compartment] = {"calibration_rows": total_calibration_rows}
        for key, values in differences.items():
            results[compartment][key] = {
                "paired_rows": len(values),
                "median_sensor1_minus_sensor2": (
                    statistics.median(signed_differences[key]) if values else None
                ),
                "median_absolute_pair_difference": statistics.median(values) if values else None,
                "p90_absolute_pair_difference": percentile(values, 0.9) if values else None,
                "monthly_median_sensor1_minus_sensor2": {
                    month: statistics.median(month_values)
                    for month, month_values in sorted(monthly_signed[key].items())
                },
            }
    return {
        "source": str(source.resolve()),
        "calibration_end_exclusive": CALIBRATION_END_EXCLUSIVE,
        "interpretation": "Pair differences include spatial heterogeneity; they are not pure sensor error.",
        "compartments": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["compartments"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
