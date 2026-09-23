#!/usr/bin/env python3
"""Prepare external AGC 2019 data without copying raw files into the repository."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowlab.greenhouse_data import prepare_agc2019


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True,
                        help="AutonomousGreenhouseChallenge2019 directory")
    parser.add_argument("--out", type=Path, required=True,
                        help="external output directory; do not place raw data in Git")
    args = parser.parse_args()
    audit = prepare_agc2019(args.source, args.out)
    summary = {
        name: {
            "events": details["irrigation"]["positive_events"],
            "counter_resets": details["irrigation"]["counter_resets"],
            "rejected_root_values": details["root_zone"]["rejected_values"],
            "rejected_climate_values": details["climate"]["rejected_values"],
            "mean_daily_irrigation_difference_l_m2": details[
                "daily_irrigation_mean_abs_difference_l_m2"
            ],
        }
        for name, details in audit["compartments"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
