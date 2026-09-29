#!/usr/bin/env python3
"""Audit source-clock solar timing and the 2020 daylight-saving boundary.

This is evidence about the source convention, not a UTC conversion.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


EXCEL_EPOCH = datetime(1899, 12, 30)
DST_DATE = datetime(2020, 3, 29).date()


def audit(path: Path) -> dict:
    days: dict[str, list[tuple[float, float | None]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            clock = EXCEL_EPOCH + timedelta(days=float(row["%time"]))
            value = float(row["Iglob"])
            irradiance = value if math.isfinite(value) else None
            days[clock.date().isoformat()].append(
                (clock.hour + clock.minute / 60 + clock.second / 3600, irradiance)
            )

    profiles = {}
    for date, samples in days.items():
        finite = [(hour, value) for hour, value in samples if value is not None]
        if not finite:
            continue
        peak = max(value for _, value in finite)
        above = [hour for hour, value in finite if value > max(20, 0.2 * peak)]
        if not above:
            continue
        profiles[date] = {
            "solar_window_midpoint_clock_hour": (min(above) + max(above)) / 2,
            "first_above_threshold_clock_hour": min(above),
            "last_above_threshold_clock_hour": max(above),
            "peak_W_m2": peak,
        }

    def median_month(month: int) -> float:
        values = [
            item["solar_window_midpoint_clock_hour"]
            for date, item in profiles.items()
            if int(date[5:7]) == month and item["peak_W_m2"] >= 250
        ]
        return statistics.median(values)

    dst_samples = days[DST_DATE.isoformat()]
    return {
        "source": str(path.resolve()),
        "method": "midpoint of Iglob > max(20 W/m2, 20% of daily peak); weather-dependent diagnostic only",
        "monthly_median_clock_hour": {
            str(month): median_month(month) for month in (1, 2, 3, 4, 5)
        },
        "transition_days": {
            date: profiles[date]
            for date in ("2020-03-28", "2020-03-29", "2020-03-30", "2020-03-31")
        },
        "march_29_02h_nan_count": sum(
            2 <= hour < 3 and value is None for hour, value in dst_samples
        ),
        "march_29_02h_finite_count": sum(
            2 <= hour < 3 and value is not None for hour, value in dst_samples
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
