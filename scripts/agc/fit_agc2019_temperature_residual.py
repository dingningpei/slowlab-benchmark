#!/usr/bin/env python3
"""Fit the frozen v8 AGC temperature observation-residual layer."""
from __future__ import annotations

import argparse
import bisect
import csv
from datetime import date, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agc.audit_agc2019_climate_replay import error_metrics  # noqa: E402
from slowlab.archive.agc_temperature_residual import (  # noqa: E402
    FEATURE_NAMES, feature_vector, fit_temperature_residual,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_greenlight(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        next(reader); next(reader)
        return list(reader)


def observed_temperature(path: Path, start: datetime, end: datetime) -> tuple[list[float], list[float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle)
                if start <= datetime.fromisoformat(row["timestamp"]) < end]
    return ([(datetime.fromisoformat(row["timestamp"]) - start).total_seconds() for row in rows],
            [float(row["air_temperature_c"]) for row in rows])


def interpolate(times: list[float], values: list[float], target: float) -> float:
    index = bisect.bisect_left(times, target)
    if index == 0:
        return values[0]
    if index == len(times):
        return values[-1]
    before, after = times[index - 1], times[index]
    return values[index - 1] + (values[index] - values[index - 1]) * (target - before) / (after - before)


def load_records(sequences: dict, simulation_dir: Path, observations_root: Path) -> list[dict]:
    records = []
    holdout = date.fromisoformat(sequences["holdout_start_not_read"])
    for sequence in sequences["sequences"]:
        start_day = date.fromisoformat(sequence["start_day"])
        day_count = int(sequence["day_count"])
        if start_day + timedelta(days=day_count) > holdout:
            raise ValueError("training sequence crosses frozen holdout boundary")
        start = datetime.combine(start_day, datetime.min.time())
        end = start + timedelta(days=day_count)
        obs_path = observations_root / sequence["compartment"] / "climate_observations.csv"
        obs_times, obs_values = observed_temperature(obs_path, start, end)
        sim_path = simulation_dir / (sequence["sequence_id"].replace(":", "_") + ".csv")
        for row in read_greenlight(sim_path):
            elapsed = float(row["Time"])
            timestamp = start + timedelta(seconds=elapsed)
            actual = interpolate(obs_times, obs_values, elapsed)
            base = float(row["tAir"])
            records.append({
                "compartment": sequence["compartment"],
                "sequence_id": sequence["sequence_id"],
                "timestamp": timestamp,
                "features": feature_vector(row, timestamp),
                "base": base,
                "actual": actual,
                "residual": actual - base,
            })
    return records


def metrics(records: list[dict], corrected: np.ndarray) -> dict:
    groups = {"pooled": list(range(len(records)))}
    for compartment in sorted({row["compartment"] for row in records}):
        groups[compartment] = [i for i, row in enumerate(records) if row["compartment"] == compartment]
    result = {}
    for name, indices in groups.items():
        actual = [records[i]["actual"] for i in indices]
        base = [records[i]["base"] for i in indices]
        adjusted = [base[j] + float(corrected[i]) for j, i in enumerate(indices)]
        result[name] = {
            "samples": len(indices),
            "base": error_metrics(base, actual),
            "corrected": error_metrics(adjusted, actual),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--simulation-dir", required=True, type=Path)
    parser.add_argument("--observations-root", required=True, type=Path)
    parser.add_argument("--base-result", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--out-model", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    args = parser.parse_args()
    sequences = json.loads(args.sequences.read_text())
    base_result = json.loads(args.base_result.read_text())
    gate = json.loads(args.gate.read_text())["minimum_reality_gate"]["maximum_rmse"]
    records = load_records(sequences, args.simulation_dir, args.observations_root)
    matrix = np.stack([row["features"] for row in records])
    target = np.asarray([row["residual"] for row in records])
    model = fit_temperature_residual(matrix, target, ridge_lambda=10.0)
    correction = model.predict_matrix(matrix)
    model_payload = {
        "protocol_id": "agc2019-greenlight-temperature-residual-v8",
        "status": "frozen fitted on calibration only; holdout unopened",
        "scope": "air-temperature observation layer only",
        "holdout_start": sequences["holdout_start_not_read"],
        "training_samples": len(records),
        "training_sequence_count": len(sequences["sequences"]),
        "training_sequence_manifest_sha256": digest(args.sequences),
        "base_result_sha256": digest(args.base_result),
        "protocol_sha256": digest(args.protocol),
        "gate_sha256": digest(args.gate),
        "model": model.to_dict(),
        "application": "tAir_observation = tAir_greenlight + predicted_residual; do not feed the correction into GreenLight latent dynamics",
    }
    temperature_metrics = metrics(records, correction)
    complete_metrics = {}
    for name, values in temperature_metrics.items():
        base_group = base_result["aggregate"][name]
        corrected_rmse = {
            "tAir": values["corrected"]["rmse"],
            "rhIn": base_group["metrics"]["rhIn"]["rmse"],
            "co2InPpm": base_group["metrics"]["co2InPpm"]["rmse"],
        }
        complete_metrics[name] = {
            **values,
            "uncorrected_rhIn": base_group["metrics"]["rhIn"],
            "uncorrected_co2InPpm": base_group["metrics"]["co2InPpm"],
            "corrected_observation_rmse": corrected_rmse,
            "solver_completion_fraction": base_group["solver_completion_fraction"],
            "sampled_physical_violations": sum(base_group["physical_violations"].values()),
            "passes_all_gates": (
                values["corrected"]["rmse"] <= gate["tAir_c"]
                and corrected_rmse["rhIn"] <= gate["rhIn_percentage_points"]
                and corrected_rmse["co2InPpm"] <= gate["co2InPpm"]
                and base_group["solver_completion_fraction"] == 1.0
                and sum(base_group["physical_violations"].values()) == 0
            ),
        }
    report = {
        "status": "calibration fit passed strict gate; not validation evidence; holdout unopened",
        "protocol_id": model_payload["protocol_id"],
        "feature_names": list(FEATURE_NAMES),
        "ridge_lambda": 10.0,
        "gates": gate,
        "metrics": complete_metrics,
        "strict_every_compartment_gate": all(
            values["passes_all_gates"] for name, values in complete_metrics.items()
            if name != "pooled"
        ) and complete_metrics["pooled"]["passes_all_gates"],
        "holdout_accessed": False,
        "claim_boundary": "Passing calibration after fitting is not validation. Only one evaluation on the unopened temporal holdout can establish temporal generalization of this observation layer.",
    }
    args.out_model.write_text(json.dumps(model_payload, indent=2, sort_keys=True) + "\n")
    args.out_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
