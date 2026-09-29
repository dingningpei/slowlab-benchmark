#!/usr/bin/env python3
"""Diagnostic AGC climate comparison; a complete solve is not validation."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path


CALIBRATION_START = date(2019, 12, 16)
CALIBRATION_END = date(2020, 3, 31)
FIELDS = (
    ("tAir", "air_temperature_c"),
    ("rhIn", "relative_humidity_pct"),
    ("co2InPpm", "co2_ppm"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interpolate(times: list[float], values: list[float], point: float) -> float:
    index = bisect_left(times, point)
    if index < len(times) and times[index] == point:
        return values[index]
    if index == 0 or index == len(times):
        raise ValueError("simulation sample lies outside observed climate times")
    fraction = (point - times[index - 1]) / (times[index] - times[index - 1])
    return values[index - 1] + fraction * (values[index] - values[index - 1])


def error_metrics(model: list[float], actual: list[float]) -> dict[str, float]:
    if len(model) != len(actual) or not model:
        raise ValueError("metric series must have equal non-zero length")
    errors = [m - o for m, o in zip(model, actual)]
    deltas = [(model[i] - model[i-1]) - (actual[i] - actual[i-1])
              for i in range(1, len(model))]
    return {
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "bias": sum(errors) / len(errors),
        "hourly_change_rmse": (math.sqrt(sum(error * error for error in deltas) / len(deltas))
                               if deltas else math.nan),
    }


def audit(simulation: Path, observations: Path, day: date, solver_audit: Path,
          method: str | None = None) -> dict:
    if not CALIBRATION_START <= day <= CALIBRATION_END:
        raise ValueError("only frozen calibration dates may be inspected")

    solver = json.loads(solver_audit.read_text())
    if method is not None:
        solver = solver[method]
    solver_complete = bool(solver.get("solver_success", solver.get("success", False)))
    solver_end = float(solver.get("end_seconds", solver.get("end", math.nan)))
    solver_complete = solver_complete and abs(solver_end - 86400.0) <= 1e-6

    with simulation.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        next(reader)  # GreenLight description row.
        next(reader)  # GreenLight units row.
        simulated = list(reader)
    if not simulated:
        raise ValueError("empty GreenLight trajectory")
    sim_times = [float(row["Time"]) for row in simulated]
    if any(not math.isfinite(t) or t < 0 or t >= 86400 for t in sim_times):
        raise ValueError("invalid GreenLight output time")
    if any(b <= a for a, b in zip(sim_times, sim_times[1:])):
        raise ValueError("GreenLight output times are not strictly increasing")
    if (abs(sim_times[0]) > 1e-6 or sim_times[-1] < 82800 - 1e-6
            or any(b - a > 3600 + 1e-6 for a, b in zip(sim_times, sim_times[1:]))):
        raise ValueError("GreenLight output does not cover the day at hourly resolution")

    start = datetime.combine(day, datetime.min.time())
    with observations.open(newline="", encoding="utf-8-sig") as handle:
        observed = [row for row in csv.DictReader(handle)
                    if datetime.fromisoformat(row["timestamp"]).date() == day]
    if len(observed) < 2:
        raise ValueError("not enough observed climate records for day")
    obs_times = [(datetime.fromisoformat(row["timestamp"]) - start).total_seconds()
                 for row in observed]
    if any(b <= a for a, b in zip(obs_times, obs_times[1:])):
        raise ValueError("observed climate times are not strictly increasing")

    metrics = {}
    physical_violations = {}
    for sim_name, obs_name in FIELDS:
        model = [float(row[sim_name]) for row in simulated]
        actual = [interpolate(obs_times, [float(row[obs_name]) for row in observed], t)
                  for t in sim_times]
        if any(not math.isfinite(value) for value in model + actual):
            raise ValueError(f"non-finite value in {sim_name} comparison")
        metrics[sim_name] = {
            **error_metrics(model, actual),
            "initial_model": model[0],
            "initial_observed": actual[0],
            "model_min": min(model),
            "model_max": max(model),
        }
        if sim_name == "tAir":
            physical_violations[sim_name] = sum(value <= -273.15 for value in model)
        elif sim_name == "rhIn":
            physical_violations[sim_name] = sum(value < -1e-3 or value > 100.001 for value in model)
        else:
            physical_violations[sim_name] = sum(value < 0 for value in model)

    return {
        "day": day.isoformat(),
        "status": "diagnostic_only_not_physical_validation",
        "solver_complete": solver_complete,
        "solver_end_seconds": solver_end,
        "physical_violations": physical_violations,
        "physical_check_scope": "sampled CSV rows only; solver internal steps are not checked",
        "samples_compared": len(simulated),
        "metrics": metrics,
        "sha256": {"simulation": sha256(simulation),
                   "observations": sha256(observations),
                   "solver_audit": sha256(solver_audit)},
        "caveat": "Initial state and facility/action mapping are not independently calibrated; metrics are diagnostics only.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--day", required=True, type=date.fromisoformat)
    parser.add_argument("--solver-audit", required=True, type=Path)
    parser.add_argument("--method")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.simulation, args.observations, args.day,
                   args.solver_audit, args.method)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
