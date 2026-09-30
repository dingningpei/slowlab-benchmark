"""Declared virtual-sensor measurement noise, sampled once per reading.

The model is frozen in ``configs/sensor_noise_v0.json``. Every random draw is a
pure function of (noise seed, component, name, compartment, measurement time),
computed with a keyed hash, so a reading's error never depends on the true
state, on actions, on how often or in which order anything is queried, or on the
feedback condition. The same reading identity always yields the same observed
value; the observation stores then keep it immutable.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Mapping

_PERSON = b'slowlab-noise-v0'
_NORMAL = NormalDist()


def _merged_setting(config: dict, setting: str) -> dict:
    settings = config['settings']
    if setting not in settings:
        raise ValueError(f'unknown sensor-noise setting: {setting}')
    spec = settings[setting]
    if 'inherits' not in spec:
        return copy.deepcopy(spec)
    base = _merged_setting(config, spec['inherits'])
    for channel, fields in spec.get('overrides', {}).items():
        if channel not in base['channels']:
            raise ValueError(f'override for unknown channel: {channel}')
        base['channels'][channel].update(fields)
    return base


class SensorNoise:
    """Executor-private measurement model; never passed to an agent or tool."""

    def __init__(self, config: dict, seed: int, setting: str = 'main'):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 64:
            raise ValueError('noise seed must be an unsigned 64-bit integer')
        self.noise_id = config['noise_id']
        self.setting = setting
        self._spec = _merged_setting(config, setting)
        self._key = seed.to_bytes(8, 'big')
        self._group_of = {}
        for group, spec in self._spec['sensor_groups'].items():
            p = spec['dropout_probability']
            if not 0 <= p < 1:
                raise ValueError('dropout probability must lie in [0, 1)')
            for channel in spec['channels']:
                if channel in self._group_of:
                    raise ValueError(f'channel in two sensor groups: {channel}')
                if channel not in self._spec['channels']:
                    raise ValueError(f'grouped channel has no noise model: {channel}')
                self._group_of[channel] = group
        for channel, spec in self._spec['channels'].items():
            if spec['model'] not in ('additive', 'log_multiplicative'):
                raise ValueError(f'unknown noise model for {channel}')
            if channel not in self._group_of:
                raise ValueError(f'noisy channel lacks a sensor group: {channel}')
        self.channels = frozenset(self._spec['channels'])
        self.exact_channels = frozenset(self._spec['exact_channels'])

    @classmethod
    def from_file(cls, path: Path, seed: int, setting: str = 'main') -> 'SensorNoise':
        return cls(json.loads(Path(path).read_text()), seed, setting)

    # ── counter-based draws ────────────────────────────────────────────────
    def _uniform(self, component: str, name: str, compartment: str, time_seconds: int) -> float:
        message = f'{component}|{name}|{compartment}|{time_seconds}'.encode()
        digest = hashlib.blake2b(message, key=self._key, digest_size=16, person=_PERSON).digest()
        k = int.from_bytes(digest[:8], 'big') >> 11
        return (k + 0.5) / 2 ** 53

    def _normal(self, component: str, name: str, compartment: str, time_seconds: int) -> float:
        return _NORMAL.inv_cdf(self._uniform(component, name, compartment, time_seconds))

    @staticmethod
    def _time(measurement_time: float) -> int:
        if isinstance(measurement_time, bool) or not math.isfinite(measurement_time):
            raise ValueError('measurement time must be finite')
        t = int(round(measurement_time))
        if t < 0 or abs(t - measurement_time) > 1e-9:
            raise ValueError('measurement time must be a nonnegative whole second')
        return t

    # ── public API ─────────────────────────────────────────────────────────
    def dropped_groups(self, compartment: str, measurement_time: float) -> frozenset:
        t = self._time(measurement_time)
        if t == 0:
            return frozenset()
        return frozenset(group for group, spec in self._spec['sensor_groups'].items()
                         if self._uniform('dropout', group, compartment, t) < spec['dropout_probability'])

    def observed_value(self, channel: str, compartment: str, measurement_time: float,
                       truth: float) -> float:
        spec = self._spec['channels'][channel]
        t = self._time(measurement_time)
        if isinstance(truth, bool) or not isinstance(truth, (int, float)) or not math.isfinite(truth):
            raise ValueError(f'nonfinite true value for {channel}')
        if spec['model'] == 'additive':
            sd = math.hypot(spec['reading_sd_abs'], spec['reading_sd_rel'] * abs(truth))
            value = truth + sd * self._normal('reading', channel, compartment, t)
            if spec['sensor_bias_sd_abs']:
                value += spec['sensor_bias_sd_abs'] * self._normal('bias', channel, compartment, 0)
        else:
            day = t // 86400
            log_error = (spec['day_sd_log'] * self._normal('day', channel, compartment, day)
                         + spec['reading_sd_log'] * self._normal('reading', channel, compartment, t))
            value = truth * math.exp(log_error)
        low, high = spec['clip']
        if low is not None:
            value = max(low, value)
        if high is not None:
            value = min(high, value)
        value = round(value, spec['decimals'])
        return value + 0.0  # normalise -0.0

    def observe(self, compartment: str, measurement_time: float,
                truth: Mapping[str, float]) -> tuple[dict, tuple]:
        """Return (observed readings, dropped channels) for one endpoint.

        Channels without a noise model must be declared exact and pass through
        unchanged. Dropped channels are omitted from the observed mapping.
        """
        unknown = set(truth) - self.channels - self.exact_channels
        if unknown:
            raise ValueError('channel with no declared measurement model: ' + ', '.join(sorted(unknown)))
        dropped_groups = self.dropped_groups(compartment, measurement_time)
        observed, dropped = {}, []
        for channel, value in truth.items():
            if channel in self.exact_channels:
                observed[channel] = value
            elif self._group_of[channel] in dropped_groups:
                dropped.append(channel)
            else:
                observed[channel] = self.observed_value(channel, compartment, measurement_time, value)
        return observed, tuple(sorted(dropped))

    def describe(self) -> dict:
        """Non-secret identity of the model for results and settlements (no seed)."""
        return {'noise_id': self.noise_id, 'setting': self.setting}


def verify_trace_row(noise: SensorNoise, unit: str, end: float, row: dict,
                     previous_row: dict | None) -> None:
    """Re-derive one private trace row's public readings from its recorded truth.

    Checks: dropouts match the model; kept readings equal the model's observed
    value and were measured at ``end``; dropped readings are the held previous
    record; the controller at the step start read exactly the previous public
    climate-box reading (one physical sensor, two stores).
    """
    truth = row['sensor_truth']
    if set(truth) != noise.channels:
        raise AssertionError('trace truth does not cover every noisy channel')
    expected, dropped = noise.observe(unit, end, truth)
    if tuple(row['sensor_dropped']) != dropped:
        raise AssertionError(f'dropout mismatch for unit {unit} at {end}')
    public = row['public_endpoint']
    for channel in noise.channels:
        record = public[channel]
        if channel in dropped:
            if record['measurement_time'] >= end:
                raise AssertionError('dropped reading was recorded')
            if previous_row is not None and record != previous_row['public_endpoint'][channel]:
                raise AssertionError('dropped reading did not hold the previous record')
        else:
            if record['measurement_time'] != end or record['available_at'] != end:
                raise AssertionError('kept reading not released at the endpoint')
            if record['value'] != expected[channel]:
                raise AssertionError(f'observed value mismatch for {channel} at {end}')
    if previous_row is not None:
        for channel, record in row['controller_records'].items():
            if channel in public and record != previous_row['public_endpoint'][channel]:
                raise AssertionError('controller and public stores disagree on a shared sensor')
