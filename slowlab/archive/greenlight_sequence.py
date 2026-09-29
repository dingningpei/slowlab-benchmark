"""Compose consecutive GreenLight input CSVs without resetting model state."""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Sequence


def concatenate_greenlight_inputs(paths: Sequence[Path], out: Path,
                                  period_seconds: float = 86400.0) -> int:
    if not paths:
        raise ValueError("at least one input file is required")
    header = descriptions = units = None
    combined = []
    expected = 0
    for day_index, path in enumerate(paths):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            current_header = tuple(reader.fieldnames or ())
            current_descriptions = next(reader, None)
            current_units = next(reader, None)
            rows = list(reader)
        if current_descriptions is None or current_units is None or len(rows) < 2:
            raise ValueError(f"invalid GreenLight input: {path}")
        if header is None:
            header, descriptions, units = current_header, current_descriptions, current_units
        elif (current_header != header or current_descriptions != descriptions or current_units != units):
            raise ValueError("GreenLight input metadata changed across sequence")
        times = [float(row["Time"]) for row in rows]
        if abs(times[0]) > 1e-6 or abs(times[-1] - period_seconds) > 1e-6:
            raise ValueError("each GreenLight input must cover one complete period")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("input times must be strictly increasing")
        for row_index, row in enumerate(rows):
            if day_index and row_index == 0:
                continue
            row = dict(row)
            row["Time"] = repr(times[row_index] + day_index * period_seconds)
            combined.append(row)
        expected += len(rows) - (1 if day_index else 0)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader(); writer.writerow(descriptions); writer.writerow(units); writer.writerows(combined)
    if len(combined) != expected:
        raise AssertionError("unexpected sequence row count")
    return len(combined)
