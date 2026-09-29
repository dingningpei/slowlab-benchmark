#!/usr/bin/env python3
"""Audit calibration-only LED input excitation and seed energy feasibility."""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowlab.archive.agc_lighting import LED_FIELDS, agc_toplight_flux  # noqa: E402
from slowlab.archive.greenhouse_data import excel_datetime  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(samples: list[tuple[float, dict[str, float]]], zeta: float, eta_nir: float,
              eta_cool: float) -> dict:
    if zeta <= 0 or min(eta_nir, eta_cool) < 0 or eta_nir + eta_cool > 1:
        raise ValueError("invalid LED energy parameters")
    on = [(q, channels) for q, channels in samples if q > 0]
    if not on:
        raise ValueError("no powered LED samples")
    matrix = np.asarray([[channels[field] for field in LED_FIELDS] for _, channels in on])
    powers = np.asarray([q for q, _ in on])
    residuals = []
    for q, channels in on:
        photons = sum(capacity * channels[field] for field, capacity in {
            "int_blue_vip": 11.0, "int_red_vip": 49.0,
            "int_farred_vip": 0.0, "int_white_vip": 37.0,
        }.items())
        residuals.append(q - photons / zeta - eta_nir * q - eta_cool * q)
    centered = matrix - matrix.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    nonzero = singular[singular > max(centered.shape) * np.finfo(float).eps * singular[0]] if singular[0] else np.array([])
    return {
        "powered_samples": len(on),
        "unique_channel_vectors": int(len(np.unique(matrix, axis=0))),
        "channel_matrix_rank": int(np.linalg.matrix_rank(matrix)),
        "centered_channel_matrix_rank": int(np.linalg.matrix_rank(centered)),
        "centered_condition_number_nonzero_subspace": float(nonzero[0] / nonzero[-1]) if len(nonzero) else None,
        "channel_min_fraction": dict(zip(LED_FIELDS, matrix.min(axis=0).tolist())),
        "channel_max_fraction": dict(zip(LED_FIELDS, matrix.max(axis=0).tolist())),
        "channel_std_fraction": dict(zip(LED_FIELDS, matrix.std(axis=0).tolist())),
        "processed_led_power_w_m2": {"min": float(powers.min()), "max": float(powers.max()), "std": float(powers.std())},
        "seed_fixture_heat_residual_w_m2": {"min": float(min(residuals)), "max": float(max(residuals))},
        "seed_energy_partition_feasible": min(residuals) >= -1e-9,
    }


def audit(source: Path, manifest_path: Path, coverage_path: Path,
          zeta: float, eta_nir: float, eta_cool: float) -> dict:
    manifest = json.loads(manifest_path.read_text())
    coverage = json.loads(coverage_path.read_text())
    holdout = date.fromisoformat(manifest["holdout_start"])
    samples = []
    selected = {}
    for compartment, summary in coverage["summary"].items():
        eligible = set(summary["input_eligible_dates"])
        selected[compartment] = sorted(eligible)
        path = source / compartment / "GreenhouseClimate.csv"
        expected = manifest["compartments"][compartment]["source_sha256"]["GreenhouseClimate.csv"]
        if digest(path) != expected:
            raise ValueError(f"source hash mismatch: {compartment}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                stamp = excel_datetime(float(row["%time"]))
                if stamp.date() >= holdout:
                    break
                if stamp.date().isoformat() not in eligible:
                    continue
                led = {field: float(row[field]) for field in LED_FIELDS}
                flux = agc_toplight_flux(float(row["AssimLight"]), led)
                samples.append((flux.processed_led_power_w_m2,
                                dict(flux.led_fraction_by_channel)))
    result = summarize(samples, zeta, eta_nir, eta_cool)
    result.update({
        "status": "calibration_input_excitation_and_seed_energy_check_only",
        "selected_compartment_days": selected,
        "holdout_start_not_read": holdout.isoformat(),
        "parameters": {"zeta_led_par_umol_j": zeta, "eta_led_nir": eta_nir, "eta_led_cool": eta_cool},
        "interpretation": "Full input rank is necessary but does not identify fixture heat capacity, emissivity, area, or heat-transfer coefficients from climate observations.",
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--zeta", type=float, default=5.4)
    parser.add_argument("--eta-nir", type=float, default=0.02)
    parser.add_argument("--eta-cool", type=float, default=0.0)
    args = parser.parse_args()
    result = audit(args.source, args.manifest, args.coverage, args.zeta, args.eta_nir, args.eta_cool)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
