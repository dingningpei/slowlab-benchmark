"""Narrow, unit-checked interfaces between GreenLight output and SlowLab."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class GreenLightSeries:
    name: str
    description: str
    unit: str
    time_seconds: tuple[float, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class CanopyWaterDemand:
    root_uptake_l_m2: float
    canopy_condensation_l_m2: float


def assert_greenlight_solution_complete(solution: object, expected_end_seconds: float) -> None:
    """Fail closed on GreenLight's partial solve_ivp result.

    GreenLight can write a CSV and return from ``run()`` after an ODE failure,
    so file existence or process exit status alone is not a completion check.
    """
    if not getattr(solution, "success", False):
        raise ValueError(f"GreenLight solver failed: {getattr(solution, 'message', 'unknown error')}")
    times = getattr(solution, "t", ())
    if len(times) < 2 or not math.isfinite(float(times[-1])):
        raise ValueError("GreenLight solver returned no finite endpoint")
    if abs(float(times[-1]) - expected_end_seconds) > 1e-6:
        raise ValueError(f"GreenLight stopped at {times[-1]}, expected {expected_end_seconds}")


def read_greenlight_series(path: Path, variable: str) -> GreenLightSeries:
    """Read a GreenLight CSV while respecting its description and unit rows."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        try:
            descriptions = next(reader)
            units = next(reader)
        except StopIteration as exc:
            raise ValueError("GreenLight output lacks metadata rows") from exc
        if variable not in (reader.fieldnames or ()):
            raise ValueError(f"GreenLight output has no variable {variable!r}")
        times: list[float] = []
        values: list[float] = []
        for row_number, row in enumerate(reader, start=4):
            try:
                time_value = float(row["Time"])
                value = float(row[variable])
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"invalid numeric data on GreenLight CSV row {row_number}") from exc
            if not math.isfinite(time_value) or not math.isfinite(value):
                raise ValueError(f"non-finite value on GreenLight CSV row {row_number}")
            times.append(time_value)
            values.append(value)
    if len(times) < 2:
        raise ValueError("GreenLight output needs at least two numeric rows")
    return GreenLightSeries(
        name=variable,
        description=str(descriptions[variable]),
        unit=str(units[variable]),
        time_seconds=tuple(times),
        values=tuple(values),
    )


def integrate_canopy_water_demand(
    time_seconds: Sequence[float] | Iterable[float],
    transpiration_kg_m2_s: Sequence[float] | Iterable[float],
    *,
    end_time_seconds: float | None = None,
) -> CanopyWaterDemand:
    """Integrate GreenLight's sampled canopy vapour flux.

    One kg of liquid water per square metre equals one litre per square metre.
    Positive ``mvCanAir`` becomes root-water demand. Negative flux is retained
    separately as canopy condensation and is never silently treated as uptake.
    GreenLight writes interval-start samples and its official example uses a
    left-rectangle sum. The final interval is therefore extended to the declared
    simulation end. A uniform final interval may be inferred for convenience.
    """
    times = tuple(float(value) for value in time_seconds)
    fluxes = tuple(float(value) for value in transpiration_kg_m2_s)
    if len(times) != len(fluxes) or len(times) < 2:
        raise ValueError("time and flux series must have equal length >= 2")
    intervals = []
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("GreenLight times must be finite and strictly increasing")
        intervals.append(dt)
    if end_time_seconds is None:
        first_dt = intervals[0]
        if any(abs(dt - first_dt) > max(1e-9, 1e-9 * first_dt) for dt in intervals):
            raise ValueError("end_time_seconds is required for non-uniform output")
        end_time_seconds = times[-1] + first_dt
    final_dt = float(end_time_seconds) - times[-1]
    if not math.isfinite(final_dt) or final_dt <= 0:
        raise ValueError("simulation end must be after the final sample")
    intervals.append(final_dt)

    uptake = 0.0
    condensation = 0.0
    for flux, dt in zip(fluxes, intervals):
        if not math.isfinite(flux):
            raise ValueError("GreenLight transpiration flux must be finite")
        area = flux * dt
        if area >= 0:
            uptake += area
        else:
            condensation -= area
    return CanopyWaterDemand(
        root_uptake_l_m2=uptake,
        canopy_condensation_l_m2=condensation,
    )


def greenlight_canopy_water_demand(
    path: Path, *, simulation_end_seconds: float | None = None
) -> CanopyWaterDemand:
    series = read_greenlight_series(path, "mvCanAir")
    expected = "kg m**-2 s**-1"
    if series.unit != expected:
        raise ValueError(f"mvCanAir unit changed: expected {expected!r}, got {series.unit!r}")
    return integrate_canopy_water_demand(
        series.time_seconds,
        series.values,
        end_time_seconds=simulation_end_seconds,
    )
