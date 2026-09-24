#!/usr/bin/env python3
"""Reproduce AGC2's calibration-period processed lighting electricity ledger.

The daily ledger is computed from the same process-computer statuses and VIPs,
so agreement checks units and handling of blanks, not an independent meter.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowlab.greenhouse_data import COMPARTMENTS, excel_datetime  # noqa: E402

HPS_W_M2 = 81.0
LED_W_M2 = {"int_blue_vip": 7.27, "int_red_vip": 25.3,
             "int_farred_vip": 6.23, "int_white_vip": 22.72}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def finite(value: str | None) -> float | None:
    try:
        result = float(value) if value is not None else math.nan
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def audit(climate: Path, resources: Path, start: date, end: date) -> dict:
    days = defaultdict(lambda: {"high": 0.0, "low": 0.0, "rows": 0,
                                "led_blank_hps_on_samples": 0,
                                "lamp_par_zero_in_those_samples": 0,
                                "bad_hps": 0})
    with climate.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            stamp = excel_datetime(float(row["%time"]))
            if not start <= stamp.date() < end:
                continue
            data = days[stamp.date()]
            data["rows"] += 1
            hps = finite(row.get("AssimLight"))
            if hps not in (0, 100):
                data["bad_hps"] += 1
                continue
            power = HPS_W_M2 * hps / 100
            led_missing_while_hps_on = False
            for field, watts in LED_W_M2.items():
                fraction = finite(row.get(field))
                if fraction is None:
                    if hps == 0:  # Hardware interlock: LED cannot be powered alone.
                        fraction = 0
                    else:
                        led_missing_while_hps_on = True
                        continue
                if not 0 <= fraction <= 1000:
                    led_missing_while_hps_on = True
                    continue
                power += watts * fraction / 1000
            if led_missing_while_hps_on:
                data["led_blank_hps_on_samples"] += 1
                data["lamp_par_zero_in_those_samples"] += finite(row.get("Tot_PAR_Lamps")) == 0
            bucket = "high" if 7 <= stamp.hour < 23 else "low"
            data[bucket] += power / 12 / 1000  # 5-minute W/m2 -> kWh/m2.
    observed = {}
    with resources.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            day = excel_datetime(float(row["%Time "])).date()
            if start <= day < end:
                high = finite(row.get("ElecHigh"))
                low = finite(row.get("ElecLow"))
                if high is not None and low is not None:
                    observed[day] = (high, low)
    errors = []
    eligible = 0
    for day, data in days.items():
        if data["rows"] != 288 or data["bad_hps"] or data["led_blank_hps_on_samples"] or day not in observed:
            continue
        eligible += 1
        errors.extend((abs(data["high"] - observed[day][0]),
                       abs(data["low"] - observed[day][1])))
    return {
        "calibration_days_seen": len(days),
        "eligible_complete_days_without_led_blank_while_hps_on": eligible,
        "excluded_days_with_led_blank_while_hps_on": sum(bool(x["led_blank_hps_on_samples"]) for x in days.values()),
        "led_blank_while_hps_on_samples": sum(x["led_blank_hps_on_samples"] for x in days.values()),
        "processed_lamp_par_zero_in_those_samples": sum(x["lamp_par_zero_in_those_samples"] for x in days.values()),
        "median_absolute_daily_bucket_error_kwh_m2": statistics.median(errors) if errors else None,
        "max_absolute_daily_bucket_error_kwh_m2": max(errors) if errors else None,
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
        climate = args.source / name / "GreenhouseClimate.csv"
        resources = args.source / name / "Resources.csv"
        hashes = manifest["compartments"][name]["source_sha256"]
        if digest(climate) != hashes["GreenhouseClimate.csv"] or digest(resources) != hashes["Resources.csv"]:
            raise ValueError(f"source hash mismatch: {name}")
        compartments[name] = audit(climate, resources, start, end)
    result = {
        "status": "calibration_only_same_source_lighting_accounting_check",
        "period_start_inclusive": start.isoformat(),
        "period_end_exclusive": end.isoformat(),
        "coefficients_w_m2": {"HPS": HPS_W_M2, **LED_W_M2},
        "compartments": compartments,
        "warning": "ElecHigh and ElecLow are calculated from these same process-computer fields, not independently metered electricity. Missing LED VIPs while HPS is on exclude a day.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(compartments, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
