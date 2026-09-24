#!/usr/bin/env python3
"""Summarize paired slab-sensor disagreement in the calibration period only."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import deque
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


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean, right_mean = statistics.mean(left), statistics.mean(right)
    covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_variance = sum((a - left_mean) ** 2 for a in left)
    right_variance = sum((b - right_mean) ** 2 for b in right)
    if left_variance <= 0 or right_variance <= 0:
        return None
    return covariance / math.sqrt(left_variance * right_variance)


def audit(source: Path) -> dict:
    results = {}
    for compartment in COMPARTMENTS:
        differences = {key: [] for key in FIELDS}
        signed_differences = {key: [] for key in FIELDS}
        monthly_signed = {key: {} for key in FIELDS}
        paired_changes = {key: ([], []) for key in FIELDS}
        hourly_changes = {key: ([], []) for key in FIELDS}
        total_calibration_rows = 0
        previous_row: dict[str, str] | None = None
        previous_twelve: deque[dict[str, str]] = deque(maxlen=12)
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
                        if (previous_row is not None and previous_row[left]
                                and previous_row[right]
                                and 0 < float(row["excel_time"])
                                - float(previous_row["excel_time"]) < 6 / 1440):
                            prior_a, prior_b = (
                                float(previous_row[left]), float(previous_row[right])
                            )
                            if math.isfinite(prior_a) and math.isfinite(prior_b):
                                paired_changes[key][0].append(a - prior_a)
                                paired_changes[key][1].append(b - prior_b)
                        if len(previous_twelve) == 12:
                            prior_hour = previous_twelve[0]
                            if (prior_hour[left] and prior_hour[right]
                                    and abs(float(row["excel_time"])
                                    - float(prior_hour["excel_time"]) - 1 / 24) < 0.001):
                                prior_a, prior_b = (
                                    float(prior_hour[left]), float(prior_hour[right])
                                )
                                if math.isfinite(prior_a) and math.isfinite(prior_b):
                                    hourly_changes[key][0].append(a - prior_a)
                                    hourly_changes[key][1].append(b - prior_b)
                previous_row = row
                previous_twelve.append(row)
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
                "adjacent_five_minute_change_correlation": correlation(
                    *paired_changes[key]
                ),
                "one_hour_change_correlation": correlation(*hourly_changes[key]),
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
