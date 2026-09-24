#!/usr/bin/env python3
"""Export source-grounded AGC climate drivers for GreenLight calibration.

Positive pipe temperatures are exported as tracking targets, while zero is exported
as an off/status flag.  Neither value is treated as measured heating power.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowlab.agc_lighting import LED_FIELDS, agc_toplight_flux  # noqa: E402


FIELDS = ("Time", "uBlScr", "uThScr", "uRoof", "mcExtAir",
          "qHpsProcessed", "qLedProcessed", "ledParPhotonFlux",
          "ledFarRedPhotonFlux", "pipeLowTarget", "pipeLowActive",
          "pipeGrowTarget", "pipeGrowActive")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"missing or invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"missing or invalid {field}")
    return value


def build(trace: Path, out: Path) -> dict:
    written = []
    with trace.open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            time = number(row, "Time")
            bounded = {field: number(row, field) for field in
                       ("BlackScr", "EnScr", "VentLee", "Ventwind", "AssimLight")}
            if any(not 0 <= value <= 100 for value in bounded.values()):
                raise ValueError(f"out-of-range control on row {row_number}")
            pipe_low = number(row, "PipeLow")
            pipe_grow = number(row, "PipeGrow")
            if min(pipe_low, pipe_grow) < 0:
                raise ValueError(f"negative pipe code on row {row_number}")
            dose = number(row, "co2_dos")
            if dose < 0:
                raise ValueError(f"negative CO2 dose on row {row_number}")
            led = {field: None if row.get(field, "") == "" else number(row, field)
                   for field in LED_FIELDS}
            try:
                light = agc_toplight_flux(bounded["AssimLight"], led)
            except ValueError as exc:
                raise ValueError(f"invalid lighting on row {row_number}: {exc}") from exc
            written.append({
                "Time": time,
                "uBlScr": bounded["BlackScr"] / 100,
                "uThScr": bounded["EnScr"] / 100,
                "uRoof": (bounded["VentLee"] + bounded["Ventwind"]) / 200,
                "mcExtAir": dose * 1e6 / 3600,
                "qHpsProcessed": light.processed_hps_power_w_m2,
                "qLedProcessed": light.processed_led_power_w_m2,
                "ledParPhotonFlux": light.led_par_photon_flux_umol_m2_s,
                "ledFarRedPhotonFlux": light.led_far_red_photon_flux_umol_m2_s,
                "pipeLowTarget": pipe_low if pipe_low > 0 else 0.0,
                "pipeLowActive": float(pipe_low > 0),
                "pipeGrowTarget": pipe_grow if pipe_grow > 0 else 0.0,
                "pipeGrowActive": float(pipe_grow > 0),
            })
    if len(written) < 2 or any(b["Time"] <= a["Time"] for a, b in zip(written, written[1:])):
        raise ValueError("control trace needs strictly increasing time rows")
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptions = dict(zip(FIELDS, (
        "Time since day start", "Observed blackout-screen closure proxy",
        "Observed energy-screen closure proxy", "Mean two-window opening proxy",
        "Processed CO2 dose under calibration accounting unit hypothesis",
        "Official-ledger HPS electrical input", "Official-ledger ELIXIA electrical input",
        "ELIXIA PAR photon flux", "ELIXIA far-red photon flux",
        "Observed positive rail-pipe temperature target", "Rail-pipe active flag",
        "Observed positive grow-pipe temperature target", "Grow-pipe active flag")))
    units = dict(zip(FIELDS, ("s", "-", "-", "-", "mg m**-2 s**-1",
                              "W m**-2", "W m**-2", "umol m**-2 s**-1",
                              "umol m**-2 s**-1", "°C", "-", "°C", "-")))
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerow(descriptions); writer.writerow(units); writer.writerows(written)
    return {
        "status": "calibration_driver_with_explicit_proxies",
        "rows": len(written), "trace_sha256": digest(trace), "output_sha256": digest(out),
        "pipe_fields_exported_as_temperature_targets_not_power": ["PipeLow", "PipeGrow"],
        "caveats": [
            "uRoof is the arithmetic mean opening fraction, not a measured air-exchange rate",
            "CO2 conversion follows the calibration accounting hypothesis and is not independent flow metering",
            "screen positions are closure proxies; product heat-transfer parameters remain to be calibrated",
            "lighting power coefficients are same-source processing values, not independent electricity metering",
            "positive pipe values are temperature targets and zero values are off/status flags; neither is heat power"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--trace", required=True, type=Path); parser.add_argument("--out", required=True, type=Path); parser.add_argument("--audit", required=True, type=Path); args = parser.parse_args()
    result = build(args.trace, args.out); args.audit.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
