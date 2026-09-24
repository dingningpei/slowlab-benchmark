#!/usr/bin/env python3
"""Export one official AGC weather day with explicit unmeasured forcings."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowlab.agc_greenlight_weather import (  # noqa: E402
    WeatherAssumptions, make_weather_rows, write_greenlight_weather,
)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--day", type=date.fromisoformat, default=date(2020, 3, 18))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--outdoor-co2-ppm", required=True, type=float)
    parser.add_argument("--deep-soil-temperature-c", required=True, type=float)
    parser.add_argument("--elevation-m", required=True, type=float)
    parser.add_argument("--pyrgeometer-body-minus-air-c", required=True, type=float)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not date.fromisoformat(manifest["calibration_start"]) <= args.day < date.fromisoformat(manifest["holdout_start"]):
        raise ValueError("weather export restricted to frozen calibration period")
    if sha256(args.source) != manifest["weather_sha256"]:
        raise ValueError("official weather hash does not match frozen manifest")
    assumptions = WeatherAssumptions(args.outdoor_co2_ppm,
                                     args.deep_soil_temperature_c,
                                     args.elevation_m,
                                     args.pyrgeometer_body_minus_air_c)
    rows = make_weather_rows(args.source, args.day, assumptions)
    write_greenlight_weather(args.out, rows)
    audit = {
        "status": "weather_input_diagnostic_only",
        "source_sha256": sha256(args.source),
        "manifest_sha256": sha256(args.manifest),
        "export_sha256": sha256(args.out),
        "source_clock_day": args.day.isoformat(),
        "rows": len(rows),
        "first_seconds": rows[0][0],
        "last_seconds": rows[-1][0],
        "explicit_unmeasured_assumptions": {
            "outdoor_co2_ppm": assumptions.outdoor_co2_ppm,
            "deep_soil_temperature_c": assumptions.deep_soil_temperature_c,
            "elevation_m": assumptions.elevation_m,
            "pyrgeometer_body_minus_air_c": assumptions.pyrgeometer_body_minus_air_c,
        },
        "pyrgeo_interpretation": "Hypothesis: net longwave flux; source sensor body temperature absent. Sensitivity required.",
        "warning": "No AGC facility or realised controls are connected; this is not trajectory validation.",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
