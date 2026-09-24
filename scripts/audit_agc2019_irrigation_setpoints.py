#!/usr/bin/env python3
"""Compare requested and realised irrigation-interval fields on training data.

Same-row agreement is descriptive, not a measured actuator latency.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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
        }
        previous_request: float | None = None
        with (source / name / "GreenhouseClimate.csv").open(
            newline="", encoding="utf-8-sig"
        ) as handle:
            for row in csv.DictReader(handle):
                if float(row["%time"]) >= CALIBRATION_END_EXCLUSIVE_EXCEL:
                    break
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
                        counts["requested_changes"] += 1
                        counts["change_same_row_equal"] += (
                            realised is not None and abs(requested - realised) < 1e-6
                        )
                    previous_request = requested
        compartments[name] = counts
    return {
        "source": str(source.resolve()),
        "calibration_end_exclusive_excel": CALIBRATION_END_EXCLUSIVE_EXCEL,
        "requested_field": REQUESTED,
        "realised_field": REALISED,
        "interpretation": "Realised VIP is the replay input; equality does not identify exact execution delay.",
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
