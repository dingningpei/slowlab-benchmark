#!/usr/bin/env python3
"""Compare requested and realised irrigation-interval fields on training data.

Same-row agreement is descriptive, not a measured actuator latency.
"""
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
CALIBRATION_END_EXCLUSIVE_EXCEL = 43922.0  # 2020-04-01
REQUESTED = "water_sup_intervals_sp_min"
REALISED = "water_sup_intervals_vip_min"


def number(raw: str) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def audit(source: Path) -> dict:
    compartments = {}
    for name in COMPARTMENTS:
        counts = {
            "rows": 0,
            "requested_present": 0,
            "realised_present": 0,
            "both_present": 0,
            "same_row_equal": 0,
            "requested_changes": 0,
            "change_same_row_equal": 0,
            "change_matched_within_60min_before_next_request": 0,
            "change_superseded_before_match": 0,
            "change_unmatched_at_60min": 0,
        }
        previous_request: float | None = None
        pending: tuple[float, float] | None = None
        matched_lags_minutes: list[float] = []
        with (source / name / "GreenhouseClimate.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            for row in csv.DictReader(handle):
                if float(row["%time"]) >= CALIBRATION_END_EXCLUSIVE_EXCEL:
                    break
                serial = float(row["%time"])
                counts["rows"] += 1
                requested = number(row[REQUESTED])
                realised = number(row[REALISED])
                counts["requested_present"] += requested is not None
                counts["realised_present"] += realised is not None
                if requested is not None and realised is not None:
                    counts["both_present"] += 1
                    counts["same_row_equal"] += abs(requested - realised) < 1e-6
                if requested is not None:
                    changed = previous_request is None or abs(requested - previous_request) > 1e-6
                    if changed:
                        if pending is not None:
                            counts["change_superseded_before_match"] += 1
                        counts["requested_changes"] += 1
                        counts["change_same_row_equal"] += (
                            realised is not None and abs(requested - realised) < 1e-6
                        )
                        pending = (serial, requested)
                    previous_request = requested
                if pending is not None:
                    elapsed_minutes = (serial - pending[0]) * 1440
                    if elapsed_minutes > 60 + 1e-6:
                        counts["change_unmatched_at_60min"] += 1
                        pending = None
                    elif realised is not None and abs(realised - pending[1]) < 1e-6:
                        counts["change_matched_within_60min_before_next_request"] += 1
                        matched_lags_minutes.append(elapsed_minutes)
                        pending = None
        if pending is not None:
            counts["change_censored_at_calibration_end"] = 1
        else:
            counts["change_censored_at_calibration_end"] = 0
        counts["matched_lag_minutes_median"] = (
            statistics.median(matched_lags_minutes) if matched_lags_minutes else None
        )
        counts["matched_lag_minutes_p90"] = (
            sorted(matched_lags_minutes)[math.ceil(0.9 * len(matched_lags_minutes)) - 1]
            if matched_lags_minutes else None
        )
        compartments[name] = counts
    return {
        "source": str(source.resolve()),
        "calibration_end_exclusive_excel": CALIBRATION_END_EXCLUSIVE_EXCEL,
        "requested_field": REQUESTED,
        "realised_field": REALISED,
        "interpretation": "Realised VIP is the replay input; first matching sample is an upper bound on observable settling, not a causal actuator latency.",
        "compartments": compartments,
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
