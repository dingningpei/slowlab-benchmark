#!/usr/bin/env python3
"""Audit the *processed* AGC litres-per-pump-minute conversion on training data.

The ratio is a source-data transformation, not an independent flow-meter
measurement. Counter resets and corrections are excluded from regression.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


COMPARTMENTS = (
    "AICU", "Automatoes", "Digilog", "IUACAAS", "Reference", "TheAutomators"
)
EXCEL_EPOCH = datetime(1899, 12, 30)
CALIBRATION_END_EXCLUSIVE = datetime(2020, 4, 1)


def finite(row: dict[str, str], field: str) -> float | None:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def stats(pairs: list[tuple[float, float]]) -> dict:
    if not pairs:
        return {"matched_increments": 0, "litres_per_m2_per_pump_minute": None}
    pump_minutes = sum(minutes for minutes, _ in pairs)
    litres = sum(volume for _, volume in pairs)
    return {
        "matched_increments": len(pairs),
        "matched_pump_minutes": pump_minutes,
        "matched_reported_litres_per_m2": litres,
        "litres_per_m2_per_pump_minute": litres / pump_minutes,
    }


def audit(source: Path) -> dict:
    results = {}
    for compartment in COMPARTMENTS:
        previous: tuple[float, float] | None = None
        paired: list[tuple[float, float]] = []
        monthly: dict[str, list[tuple[float, float]]] = defaultdict(list)
        unmatched_positive_volume = 0
        unmatched_positive_minutes = 0
        path = source / compartment / "GreenhouseClimate.csv"
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                row = {key.strip(): value for key, value in raw.items()}
                serial = finite(row, "%time")
                if serial is None:
                    previous = None
                    continue
                clock = EXCEL_EPOCH + timedelta(days=serial)
                if clock >= CALIBRATION_END_EXCLUSIVE:
                    break
                minutes = finite(row, "water_sup")
                volume = finite(row, "Cum_irr")
                if minutes is None or volume is None:
                    previous = None
                    continue
                if previous is not None:
                    delta_minutes = minutes - previous[0]
                    delta_volume = volume - previous[1]
                    if 0 < delta_minutes <= 5 and 0 < delta_volume <= 5:
                        paired.append((delta_minutes, delta_volume))
                        monthly[clock.strftime("%Y-%m")].append(
                            (delta_minutes, delta_volume)
                        )
                    elif delta_volume > 0 and delta_minutes <= 0:
                        unmatched_positive_volume += 1
                    elif delta_minutes > 0 and delta_volume <= 0:
                        unmatched_positive_minutes += 1
                previous = minutes, volume
        results[compartment] = {
            **stats(paired),
            "monthly": {month: stats(values) for month, values in sorted(monthly.items())},
            "positive_volume_without_positive_pump_minutes": unmatched_positive_volume,
            "positive_pump_minutes_without_positive_volume": unmatched_positive_minutes,
        }
    return {
        "source": str(source.resolve()),
        "calibration_end_exclusive": CALIBRATION_END_EXCLUSIVE.isoformat(),
        "interpretation": "Processed volume per raw pump minute; not an independent physical calibration.",
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
    print(json.dumps({
        name: {
            "matched_increments": item["matched_increments"],
            "litres_per_m2_per_pump_minute": item["litres_per_m2_per_pump_minute"],
        }
        for name, item in result["compartments"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
