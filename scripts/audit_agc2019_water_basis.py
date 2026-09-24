#!/usr/bin/env python3
"""Compare AGC source water/yield accounting with published water efficiency.

Area scenarios are explicit hypotheses, not fitted choices. This script reads
the official archive and writes a diagnostic report; it does not alter data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PRODUCTION_AREA_M2 = 62.5  # Official AGC ReadMe, Production field definitions.
AREA_HYPOTHESES_M2 = (62.5, 76.8, 96.0)
# Hemming et al. (2020), Sensors 20(22):6430, Table 2.
PUBLISHED_WATER_L_KG = {
    "Automatoes": 25.0,
    "AICU": 25.2,
    "TheAutomators": 25.9,
    "IUACAAS": 26.9,
    "Digilog": 27.9,
    "Reference": 27.4,
}


def column_sum(path: Path, column: str) -> float:
    total = 0.0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            raw = row[column].strip()
            if not raw or raw.lower() in {"nan", "na", "none", "null"}:
                continue
            value = float(raw)
            if not math.isfinite(value):
                continue
            total += value
    return total


def audit(root: Path) -> dict:
    compartments = {}
    for name, published in PUBLISHED_WATER_L_KG.items():
        resources = root / name / "Resources.csv"
        production = root / name / "Production.csv"
        irrigation = column_sum(resources, "Irr")
        drain = column_sum(resources, "Drain")
        yield_per_production_m2 = (
            column_sum(production, "ProdA") + column_sum(production, "ProdB")
        )
        if yield_per_production_m2 <= 0:
            raise ValueError(f"nonpositive production for {name}")
        raw_ratio = irrigation / yield_per_production_m2
        compartments[name] = {
            "irrigation_reported_L_m2_sum": irrigation,
            "drain_reported_L_m2_sum": drain,
            "production_reported_kg_m2_sum": yield_per_production_m2,
            "published_water_L_kg": published,
            "naive_same_area_irrigation_L_kg": raw_ratio,
            "naive_same_area_net_irrigation_L_kg":
                (irrigation - drain) / yield_per_production_m2,
            "irrigation_L_kg_if_reported_area_m2": {
                str(area): raw_ratio * area / PRODUCTION_AREA_M2
                for area in AREA_HYPOTHESES_M2
            },
            "published_over_naive_ratio": published / raw_ratio,
        }
    return {
        "source": str(root.resolve()),
        "production_area_m2": PRODUCTION_AREA_M2,
        "irrigation_area_hypotheses_m2": AREA_HYPOTHESES_M2,
        "method": "sum Resources Irr, Drain and Production ProdA+ProdB over all source rows",
        "warning": "Published Table 2 water-efficiency accounting is not reproduced; do not select an area by fit.",
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
    print(json.dumps({
        name: {
            "same_area_L_kg": round(item["naive_same_area_irrigation_L_kg"], 2),
            "published_L_kg": item["published_water_L_kg"],
            "ratio": round(item["published_over_naive_ratio"], 3),
        }
        for name, item in result["compartments"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
