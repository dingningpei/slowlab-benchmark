#!/usr/bin/env python3
"""Test whether a fixed inter-sensor offset transfers within calibration data.

Fit through February, diagnose on March; April/May holdout is never read.
The exercise tests an observation shortcut, not a greenhouse simulator.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


COMPARTMENTS = (
    "AICU", "Automatoes", "Digilog", "IUACAAS", "Reference", "TheAutomators"
)
FIT_END_EXCLUSIVE = "2020-03-01"
DIAGNOSTIC_END_EXCLUSIVE = "2020-04-01"
FIELDS = {
    "wc_pct": ("wc_slab1_pct", "wc_slab2_pct"),
    "ec_ds_m": ("ec_slab1_ds_m", "ec_slab2_ds_m"),
}


def audit(source: Path) -> dict:
    results = {}
    for name in COMPARTMENTS:
        fit = {key: [] for key in FIELDS}
        diagnostic = {key: [] for key in FIELDS}
        path = source / name / "root_zone.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                date = row["timestamp"][:10]
                if date >= DIAGNOSTIC_END_EXCLUSIVE:
                    break
                destination = fit if date < FIT_END_EXCLUSIVE else diagnostic
                for key, (left, right) in FIELDS.items():
                    if row[left] and row[right]:
                        destination[key].append(float(row[left]) - float(row[right]))
        results[name] = {}
        for key in FIELDS:
            training = fit[key]
            later = diagnostic[key]
            if not training or not later:
                raise ValueError(f"insufficient paired {key} data for {name}")
            offset = statistics.median(training)
            results[name][key] = {
                "fit_pairs": len(training),
                "march_pairs": len(later),
                "fit_median_sensor1_minus_sensor2": offset,
                "march_median_sensor1_minus_sensor2": statistics.median(later),
                "march_fixed_offset_mae": statistics.mean(abs(delta - offset) for delta in later),
                "march_no_offset_mae": statistics.mean(abs(delta) for delta in later),
            }
    return {
        "source": str(source.resolve()),
        "fit_end_exclusive": FIT_END_EXCLUSIVE,
        "diagnostic_end_exclusive": DIAGNOSTIC_END_EXCLUSIVE,
        "warning": "Within-calibration diagnostic only; does not use April/May holdout.",
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
