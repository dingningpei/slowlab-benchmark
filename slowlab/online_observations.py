"""Causal sensor records for the V2.2 executor (not an agent-owned object).

Only query payloads may cross the agent boundary. The executor owns the clock,
records samples once, and applies sensor noise before recording. Offline CSVs
are deliberately not accepted: relabelled timestamps cannot be repaired here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from array import array
from bisect import bisect_left, bisect_right
import math
from numbers import Real
from typing import Mapping


def _number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class SensorRecord:
    compartment: str
    variable: str
    unit: str
    measurement_time: float
    available_at: float
    value: float


class OnlineObservations:
    """Append-only executor store; times are nonnegative simulation seconds.

    `schema` is an explicit public sensor allowlist, mapping variable to unit.
    Queries use a historical knowledge cutoff, not interpolated sensor values.
    This store implements time gating, not Full/Endpoint permission policies.
    """

    def __init__(self, schema: Mapping[str, str]):
        if not schema or any(not isinstance(k, str) or not k or
                             not isinstance(v, str) or not v for k, v in schema.items()):
            raise ValueError("sensor schema must contain nonempty names and units")
        self._schema = dict(schema)
        self._clock = 0.0
        self._channel_records: dict[tuple[str, str], list[SensorRecord]] = {}
        self._channel_times: dict[tuple[str, str], list[float]] = {}

    @property
    def clock(self) -> float:
        return self._clock

    def advance_to(self, time_seconds: float) -> None:
        """Executor only: advance after the physical simulation has reached time."""
        time_seconds = _number(time_seconds, "clock")
        if time_seconds < self._clock:
            raise ValueError("clock cannot move backwards")
        self._clock = time_seconds

    def record(self, compartment: str, variable: str, *, measurement_time: float,
               available_at: float, value: float) -> None:
        """Executor only: ingest a completed measurement with scheduled delivery.

        Future delivery is permitted; future measurement or retroactive delivery
        is not. Delayed measurements can arrive out of measurement-time order.
        """
        if not isinstance(compartment, str) or not compartment:
            raise ValueError("compartment must be a nonempty string")
        if variable not in self._schema:
            raise ValueError("variable is not an allowed sensor")
        measured = _number(measurement_time, "measurement_time")
        available = _number(available_at, "available_at")
        numeric = _number(value, "value")
        if not 0 <= measured <= self._clock:
            raise ValueError("measurement must have already occurred")
        if available < self._clock or available < measured:
            raise ValueError("delivery cannot be backdated")
        channel = (compartment, variable)
        times = self._channel_times.setdefault(channel, [])
        records = self._channel_records.setdefault(channel, [])
        index = bisect_left(times, measured)
        if index < len(times) and times[index] == measured:
            raise ValueError("measurement identity is immutable")
        record = SensorRecord(compartment, variable, self._schema[variable],
                              measured, available, numeric)
        times.insert(index, measured)
        records.insert(index, record)

    def history(self, compartment: str, variable: str, *, as_of: float | None = None,
                start: float = 0.0, end: float | None = None) -> list[dict]:
        """Return fresh payloads retaining original measurement/delivery times."""
        if variable not in self._schema:
            raise ValueError("variable is not an allowed sensor")
        cutoff = self._clock if as_of is None else _number(as_of, "as_of")
        lo = _number(start, "start")
        hi = cutoff if end is None else _number(end, "end")
        if not 0 <= lo <= hi <= cutoff <= self._clock:
            raise ValueError("query window must be within the available clock")
        channel = (compartment, variable)
        times = self._channel_times.get(channel, ())
        records = self._channel_records.get(channel, ())
        left, right = bisect_left(times, lo), bisect_right(times, hi)
        return [asdict(r) for r in records[left:right] if r.available_at <= cutoff]

    def latest(self, compartment: str, variable: str, *, as_of: float | None = None) -> dict | None:
        if variable not in self._schema:
            raise ValueError("variable is not an allowed sensor")
        cutoff = self._clock if as_of is None else _number(as_of, "as_of")
        if not 0 <= cutoff <= self._clock:
            raise ValueError("query window must be within the available clock")
        channel = (compartment, variable)
        times = self._channel_times.get(channel, ())
        records = self._channel_records.get(channel, ())
        for index in range(bisect_right(times, cutoff)-1, -1, -1):
            record = records[index]
            if record.available_at <= cutoff:
                return asdict(record)
        return None


class PackedOnlineObservations(OnlineObservations):
    """Same causal API with per-channel float64 arrays for long campaigns.

    Each record keeps measurement, delivery, and value as three binary floats.
    Public payload dictionaries are materialized only when queried.
    """

    def __init__(self, schema: Mapping[str, str]):
        super().__init__(schema)
        self._channel_times: dict[tuple[str, str], array] = {}
        self._channel_available: dict[tuple[str, str], array] = {}
        self._channel_values: dict[tuple[str, str], array] = {}

    def record(self, compartment: str, variable: str, *, measurement_time: float,
               available_at: float, value: float) -> None:
        if not isinstance(compartment, str) or not compartment:
            raise ValueError("compartment must be a nonempty string")
        if variable not in self._schema:
            raise ValueError("variable is not an allowed sensor")
        measured = _number(measurement_time, "measurement_time")
        available = _number(available_at, "available_at")
        numeric = _number(value, "value")
        if not 0 <= measured <= self._clock:
            raise ValueError("measurement must have already occurred")
        if available < self._clock or available < measured:
            raise ValueError("delivery cannot be backdated")
        channel = (compartment, variable)
        times = self._channel_times.setdefault(channel, array('d'))
        available_times = self._channel_available.setdefault(channel, array('d'))
        values = self._channel_values.setdefault(channel, array('d'))
        index = bisect_left(times, measured)
        if index < len(times) and times[index] == measured:
            raise ValueError("measurement identity is immutable")
        times.insert(index, measured)
        available_times.insert(index, available)
        values.insert(index, numeric)

    def _payload(self, compartment: str, variable: str, index: int) -> dict:
        channel = (compartment, variable)
        return {'compartment': compartment, 'variable': variable,
                'unit': self._schema[variable],
                'measurement_time': self._channel_times[channel][index],
                'available_at': self._channel_available[channel][index],
                'value': self._channel_values[channel][index]}

    def history(self, compartment: str, variable: str, *, as_of: float | None = None,
                start: float = 0.0, end: float | None = None) -> list[dict]:
        if variable not in self._schema:
            raise ValueError("variable is not an allowed sensor")
        cutoff = self._clock if as_of is None else _number(as_of, "as_of")
        lo = _number(start, "start")
        hi = cutoff if end is None else _number(end, "end")
        if not 0 <= lo <= hi <= cutoff <= self._clock:
            raise ValueError("query window must be within the available clock")
        channel = (compartment, variable)
        times = self._channel_times.get(channel, ())
        available_times = self._channel_available.get(channel, ())
        left, right = bisect_left(times, lo), bisect_right(times, hi)
        return [self._payload(compartment, variable, index)
                for index in range(left, right) if available_times[index] <= cutoff]

    def latest(self, compartment: str, variable: str, *, as_of: float | None = None) -> dict | None:
        if variable not in self._schema:
            raise ValueError("variable is not an allowed sensor")
        cutoff = self._clock if as_of is None else _number(as_of, "as_of")
        if not 0 <= cutoff <= self._clock:
            raise ValueError("query window must be within the available clock")
        channel = (compartment, variable)
        times = self._channel_times.get(channel, ())
        available_times = self._channel_available.get(channel, ())
        for index in range(bisect_right(times, cutoff) - 1, -1, -1):
            if available_times[index] <= cutoff:
                return self._payload(compartment, variable, index)
        return None
