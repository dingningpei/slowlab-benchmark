#!/usr/bin/env python3
"""Build a two-source overhead-light driver from an audited AGC trace.

The output is intentionally upstream of GreenLight's thermal equations.  It
contains observed-command photon and electricity-accounting quantities but no
invented fixture temperature, radiative split, or interlighting geometry.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowlab.archive.agc_lighting import LED_FIELDS, agc_toplight_flux  # noqa: E402


OUTPUT_FIELDS = (
    "Time",
    "uHpsObserved",
    "hpsPhotonFlux",
    "ledParPhotonFlux",
    "ledFarRedPhotonFlux",
    "qHpsProcessed",
    "qLedProcessed",
    "qHpsNameplateFloorBasis",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(trace: Path, out: Path) -> dict:
    rows_out = []
    with trace.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Time", "AssimLight", *LED_FIELDS}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"control trace lacks {sorted(required - set(reader.fieldnames or ()))}")
        for row_number, row in enumerate(reader, start=2):
            led = {
                field: None if row[field] == "" else float(row[field])
                for field in LED_FIELDS
            }
            try:
                flux = agc_toplight_flux(float(row["AssimLight"]), led)
                time = float(row["Time"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid toplight input on row {row_number}: {exc}") from exc
            rows_out.append({
                "Time": time,
                "uHpsObserved": flux.hps_fraction,
                "hpsPhotonFlux": flux.hps_photon_flux_umol_m2_s,
                "ledParPhotonFlux": flux.led_par_photon_flux_umol_m2_s,
                "ledFarRedPhotonFlux": flux.led_far_red_photon_flux_umol_m2_s,
                "qHpsProcessed": flux.processed_hps_power_w_m2,
                "qLedProcessed": flux.processed_led_power_w_m2,
                "qHpsNameplateFloorBasis": flux.hps_nameplate_power_w_m2_floor,
            })
    if len(rows_out) < 2:
        raise ValueError("toplight driver needs at least two time rows")
    times = [row["Time"] for row in rows_out]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("toplight driver times must be strictly increasing")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerow({
            "Time": "Time",
            "uHpsObserved": "Observed HPS command fraction",
            "hpsPhotonFlux": "HPS capacity-scaled photon flux",
            "ledParPhotonFlux": "ELIXIA PAR photon flux",
            "ledFarRedPhotonFlux": "ELIXIA far-red photon flux",
            "qHpsProcessed": "Official-ledger HPS electrical input",
            "qLedProcessed": "Official-ledger ELIXIA electrical input",
            "qHpsNameplateFloorBasis": "HPS nameplate input on 96 m2 floor basis",
        })
        writer.writerow({
            "Time": "s",
            "uHpsObserved": "-",
            "hpsPhotonFlux": "umol m**-2 s**-1",
            "ledParPhotonFlux": "umol m**-2 s**-1",
            "ledFarRedPhotonFlux": "umol m**-2 s**-1",
            "qHpsProcessed": "W m**-2",
            "qLedProcessed": "W m**-2",
            "qHpsNameplateFloorBasis": "W m**-2",
        })
        writer.writerows(rows_out)
    return {
        "status": "two_overhead_source_driver_prethermal_calibration",
        "rows": len(rows_out),
        "trace_sha256": digest(trace),
        "output_sha256": digest(out),
        "greenlight_interlighting_used": False,
        "warning": (
            "Processed electrical coefficients are same-source accounting, not independent metering. "
            "No electrical-to-radiative or fixture-heat partition is asserted."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    result = build(args.trace, args.out)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
