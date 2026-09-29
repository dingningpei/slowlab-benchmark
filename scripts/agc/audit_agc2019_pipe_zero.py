#!/usr/bin/env python3
"""Audit whether zero-valued AGC pipe readings behave like physical temperatures.

Calibration-only, source-hash-checked evidence for actuator interpretation.
The audit does not infer actual heating power or pipe cooldown dynamics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowlab.archive.greenhouse_data import COMPARTMENTS, excel_datetime  # noqa: E402

PIPES = ("PipeLow", "PipeGrow")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def parsed(value: str | None) -> float | None:
    try:
        result = float(value) if value is not None else math.nan
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def audit(path: Path, start: date, end: date) -> dict:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"%time", "Tair", *PIPES}
        if not required <= set(reader.fieldnames or ()):  # Fail if schema changes.
            raise ValueError(f"missing fields in {path}")
        for row in reader:
            serial = parsed(row["%time"])
            if serial is None or not start <= excel_datetime(serial).date() < end:
                continue
            rows.append((serial, parsed(row["Tair"]), *(parsed(row[key]) for key in PIPES)))
    result = {"samples": len(rows)}
    for index, pipe in enumerate(PIPES, start=2):
        valid = [row[index] for row in rows if row[index] is not None]
        jumps = []
        stable_air = 0
        examples = []
        for before, after in zip(rows, rows[1:]):
            a, b = before[index], after[index]
            dt = (after[0] - before[0]) * 86400
            if a is None or b is None or not 0 < dt <= 360 or (a == 0) == (b == 0):
                continue
            jumps.append(abs(b - a))
            if before[1] is not None and after[1] is not None and abs(after[1] - before[1]) <= 1:
                stable_air += 1
            if len(examples) < 3:
                examples.append({
                    "source_time_before": excel_datetime(before[0]).isoformat(sep=" "),
                    "pipe_before_c_or_zero_code": a, "pipe_after_c_or_zero_code": b,
                    "air_before_c": before[1], "air_after_c": after[1],
                })
        result[pipe] = {
            "valid_samples": len(valid),
            "zero_samples": sum(value == 0 for value in valid),
            "zero_fraction": sum(value == 0 for value in valid) / len(valid) if valid else None,
            "zero_boundary_transitions_within_six_minutes": len(jumps),
            "median_apparent_pipe_temperature_jump_c": statistics.median(jumps) if jumps else None,
            "fraction_with_air_temperature_change_at_most_one_c": stable_air / len(jumps) if jumps else None,
            "transition_examples": examples,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    start = date.fromisoformat(manifest["calibration_start"])
    end = date.fromisoformat(manifest["holdout_start"])
    compartments = {}
    for name in COMPARTMENTS:
        path = args.source / name / "GreenhouseClimate.csv"
        if digest(path) != manifest["compartments"][name]["source_sha256"]["GreenhouseClimate.csv"]:
            raise ValueError(f"source hash mismatch: {name}")
        compartments[name] = audit(path, start, end)
    result = {
        "status": "calibration_only_pipe_zero_semantics_audit",
        "period_start_inclusive": start.isoformat(),
        "period_end_exclusive": end.isoformat(),
        "compartments": compartments,
        "interpretation": "Abrupt 0/nonzero switches with stable air support an off/status code; exact device semantics still require confirmation. Do not treat zeros as measured pipe temperatures.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({name: {pipe: {key: data[pipe][key] for key in (
        "zero_fraction", "zero_boundary_transitions_within_six_minutes",
        "median_apparent_pipe_temperature_jump_c",
        "fraction_with_air_temperature_change_at_most_one_c")}
        for pipe in PIPES} for name, data in compartments.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
