#!/usr/bin/env python3
"""Profile official AGC 2019 columns before freezing field-specific repairs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


COMPARTMENTS = (
    "AICU", "Automatoes", "Digilog", "IUACAAS", "Reference", "TheAutomators"
)
FILES = (
    "GreenhouseClimate.csv", "GrodanSens.csv", "Resources.csv",
    "Production.csv", "TomQuality.csv", "LabAnalysis.csv",
)
CAMPAIGN_EXCEL_DAYS = (43815, 43980)


def profile_file(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = [name.strip() for name in (reader.fieldnames or [])]
        stats = {
            field: {
                "missing": 0, "non_numeric": 0, "zero": 0, "negative": 0,
                "minimum": None, "maximum": None,
            }
            for field in fields
        }
        rows = 0
        outside_campaign = 0
        duplicate_timestamps = 0
        previous_timestamp: float | None = None
        for raw in reader:
            rows += 1
            row = {name.strip(): value for name, value in raw.items() if name is not None}
            for field in fields:
                value = str(row.get(field, "")).strip()
                if not value or value.lower() in {"nan", "na", "none", "null"}:
                    stats[field]["missing"] += 1
                    continue
                try:
                    number = float(value)
                except ValueError:
                    stats[field]["non_numeric"] += 1
                    continue
                if not math.isfinite(number):
                    stats[field]["missing"] += 1
                    continue
                stats[field]["zero"] += number == 0
                stats[field]["negative"] += number < 0
                old_min = stats[field]["minimum"]
                old_max = stats[field]["maximum"]
                stats[field]["minimum"] = number if old_min is None else min(old_min, number)
                stats[field]["maximum"] = number if old_max is None else max(old_max, number)
            time_value = row.get("%time", row.get("%Time", "")).strip()
            try:
                timestamp = float(time_value)
            except ValueError:
                continue
            if math.isfinite(timestamp):
                outside_campaign += not CAMPAIGN_EXCEL_DAYS[0] <= timestamp < CAMPAIGN_EXCEL_DAYS[1] + 1
                duplicate_timestamps += previous_timestamp == timestamp
                previous_timestamp = timestamp
    return {
        "rows": rows,
        "outside_campaign_timestamp_rows": outside_campaign,
        "adjacent_duplicate_timestamp_rows": duplicate_timestamps,
        "fields": stats,
    }


def audit(source: Path) -> dict:
    result = {"source": str(source.resolve()), "compartments": {}}
    for compartment in COMPARTMENTS:
        result["compartments"][compartment] = {}
        for name in FILES:
            result["compartments"][compartment][name] = profile_file(source / compartment / name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        compartment: {
            name: {
                "rows": table["rows"],
                "outside_campaign_timestamp_rows": table["outside_campaign_timestamp_rows"],
            }
            for name, table in files.items()
        }
        for compartment, files in result["compartments"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
