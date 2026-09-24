#!/usr/bin/env python3
"""Compare loaded GreenLight defaults with documented AGC2 facility quantities.

This reports mismatches and unmapped quantities. It does not construct or
validate an AGC facility model, and it never runs the ODE solver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


PARAMETERS = (
    "aFlr", "aCov", "aRoof", "aSide", "hAir", "hGh", "hVent",
    "cDgh", "cLeakage", "tauRfPar", "tauThScrPar", "tauBlScrPar",
    "useBlScr", "thetaLampMax", "thetaIntLampMax", "pBoil",
    "pBoilGro", "phiExtCo2", "lPipe", "lGroPipe",
)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def number(definition: str) -> float | None:
    """Evaluate only literal decimal or simple pi-free division, not model code."""
    try:
        value = float(definition)
    except ValueError:
        parts = definition.split("/")
        if len(parts) != 2:
            return None
        try:
            denominator = float(parts[1])
            value = float(parts[0]) / denominator
        except (ValueError, ZeroDivisionError):
            return None
    return value if math.isfinite(value) else None


def build(model: Path, weather: Path) -> dict:
    from greenlight import GreenLight  # Optional, separately installed dependency.

    sim = GreenLight(
        base_path=str(model.parent),
        input_prompt=[str(model), str(weather)],
        output_path=str(weather.parent / "agc2019_parameter_audit_unused.csv"),
    )
    sim.load()
    defaults = {}
    for key in PARAMETERS:
        expression = sim.consts.get(key)
        defaults[key] = {"definition": expression,
                         "numeric_value_if_literal": number(expression) if expression else None}
    return {
        "status": "default_parameter_gap_only_no_simulation",
        "model_sha256": sha256(model),
        "weather_sha256": sha256(weather),
        "loaded_defaults": defaults,
        "agc_published": {
            "aFlr_m2": 96,
            "aRoof_opening_max_m2": 0.3 * 96,
            "cover_PAR_transmission_used_in_processed_Tot_PAR": 0.5,
            "energy_screen_PAR_transmission_used_in_processed_Tot_PAR": 0.75,
            "blackout_screen_PAR_transmission_used_in_processed_Tot_PAR": 0.02,
            "hps_lamps_rated_electrical_w_per_m2_floor": 6 * 1000 / 96,
            "hps_w_per_m2_used_in_processed_electricity_ledger": 81.0,
            "led_blue_w_per_m2_used_in_processed_electricity_ledger": 7.27,
            "led_red_w_per_m2_used_in_processed_electricity_ledger": 25.3,
            "led_far_red_w_per_m2_used_in_processed_electricity_ledger": 6.23,
            "led_white_w_per_m2_used_in_processed_electricity_ledger": 22.72,
            "hps_capacity_umol_m2_s": 100,
            "led_capacity_umol_m2_s_including_far_red": 109,
            "rail_pipe_peak_w_per_m2": 180,
            "grow_pipe_peak_w_per_m2": 30,
            "co2_nominal_supply_max_mg_s_if_floor_denominator": 15 * 96 * 1000 / 3600,
        },
        "unmapped_or_unmeasured": [
            "cover and exposed side-wall surface areas",
            "below-screen air and whole-greenhouse heights/volumes",
            "roof-vent opening geometry and anti-thrips net resistance",
            "separate leeward/windward vent fractions to GreenLight uRoof mapping",
            "screen infrared/convective coefficients and time-varying closure semantics",
            "heating pipe physical capacity versus boiler power and zero-valued pipe-temperature readings",
            "HPS/LED electrical-to-radiative conversion and LED realised photon output",
            "CO2 dosing area denominator and above-nominal processed-rate observations",
        ],
        "interpretation": (
            "Published PAR coefficients are processing assumptions, not independent optical measurements. "
            "The default parameter vector is not a validated AGC facility override."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--weather", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.model, args.weather)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"],
                      "loaded_defaults": {key: value["numeric_value_if_literal"]
                                          for key, value in result["loaded_defaults"].items()}},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
