#!/usr/bin/env python3
"""Apply the published AGC crop-to-floor area basis to light drivers."""
from __future__ import annotations
import argparse, json
from pathlib import Path

FLOOR_AREA_M2 = 96.0
CROP_GROWING_AREA_M2 = 76.8
AREA_SCALE = round(CROP_GROWING_AREA_M2 / FLOOR_AREA_M2, 12)


def build(base: Path, out: Path) -> dict:
    data = json.loads(base.read_text())
    data["Info"].update({
        "About": "AGC overhead-light thermal equations with source-grounded crop-area to floor-area normalization",
        "Status": "calibration replay; fixed area conversion, not fitted to climate error",
        "Area normalization": "Published crop-growing area 76.8 m2 divided by compartment floor area 96 m2 = 0.8. GreenLight energy balances are per floor area.",
    })
    data["Uncalibrated AGC overhead LED parameter seeds"]["agcGrowingToFloorArea"] = {
        "unit": "-", "type": "const", "definition": repr(AREA_SCALE),
        "description": "Published 76.8 m2 crop-growing area divided by 96 m2 floor area; fixed source conversion",
    }
    for section in data.values():
        if not isinstance(section, dict):
            continue
        for name, spec in section.items():
            if not isinstance(spec, dict) or "definition" not in spec:
                continue
            definition = spec["definition"].replace(
                "qLedProcessed", "(agcGrowingToFloorArea*qLedProcessed)"
            )
            if name == "qLampIn" and definition == "qHpsProcessed":
                definition = "agcGrowingToFloorArea*qHpsProcessed"
            spec["definition"] = definition
    lighting = data["AGC observed lighting inputs and shortwave exchanges"]
    lighting["rParGhLed"]["definition"] = "agcGrowingToFloorArea * ledParPhotonFlux / zetaLedPar"
    lighting["parLedCan"]["definition"] = "agcGrowingToFloorArea * ledParPhotonFlux * (rParLedCan / max(1e-9, rParGhLed))"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    build(args.base, args.out)


if __name__ == "__main__":
    main()
