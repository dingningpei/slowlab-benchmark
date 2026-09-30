"""Formal weather repair rule v0 (research plan decision, 2026-09-30).

In the channels the physical model uses (air temperature, dew point, wind
speed from the meteo file; shortwave and longwave down from the radiation
file), a record is invalid when it is non-finite, equals the fill value, or
lies outside the reader's physical support. A run of at most ``MAX_RUN``
consecutive invalid records is replaced by linear interpolation between the
nearest valid records on either side; every replacement is logged. A year is
excluded when any run is longer, when a run touches the end of the series
(no neighbour on one side), or when a month has more than
``MAX_REPAIRS_PER_MONTH`` repaired records over all channels.

The rule is applied by the audit, which records every replaced value; the
formal reader applies exactly those values.
"""
from __future__ import annotations

import numpy as np

REQUIRED = {'meteo': ('TA002', 'TD002', 'F010'), 'radiation': ('SWD', 'LWD')}
# (low, high, low_inclusive) matching the strict reader's value anchors.
SUPPORT = {'TA002': (230.0, 330.0, False), 'TD002': (220.0, 330.0, False), 'F010': (0.0, 70.0, True),
           'SWD': (-10.0, 1500.0, False), 'LWD': (100.0, 600.0, False)}
MAX_RUN = 6
MAX_REPAIRS_PER_MONTH = 12


def invalid_mask(channel: str, values: np.ndarray, fill: float | None = None) -> np.ndarray:
    low, high, inclusive = SUPPORT[channel]
    values = np.asarray(values, dtype=np.float64)
    bad = ~np.isfinite(values)
    if fill is not None:
        bad |= values == fill
    with np.errstate(invalid='ignore'):
        bad |= (values < low) if inclusive else (values <= low)
        bad |= values >= high
    return bad


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of True."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def repair_series(values: np.ndarray, mask: np.ndarray) -> tuple[list[tuple[int, float, float]], list[tuple[int, int, str]]]:
    """Return ([(index, original, repaired)], [(start, end, reason) for unrepairable runs])."""
    values = np.asarray(values, dtype=np.float64)
    repairs, problems = [], []
    for start, end in runs(mask):
        if end - start > MAX_RUN:
            problems.append((start, end, f'run of {end - start} invalid records'))
            continue
        if start == 0 or end == len(values):
            problems.append((start, end, 'invalid records at series edge'))
            continue
        left, right = values[start - 1], values[end]
        for i in range(start, end):
            w = (i - start + 1) / (end - start + 1)
            repairs.append((i, float(values[i]), float(left + w * (right - left))))
    return repairs, problems


def repair_months(months: list[tuple[str, dict]]) -> dict:
    """Apply rule v0 to consecutive months.

    ``months`` is an ordered list of (yyyymm, {role: {'arrays': {...}, 'fills': {...}}}).
    Returns {'repairs': [...], 'problems': [...], 'repairs_per_month': {...}}.
    """
    repairs, problems = [], []
    per_month = {ym: 0 for ym, _ in months}
    for role, channels in REQUIRED.items():
        for channel in channels:
            pieces, masks, owner = [], [], []
            for ym, data in months:
                v = np.asarray(data[role]['arrays'][channel], dtype=np.float64)
                pieces.append(v)
                masks.append(invalid_mask(channel, v, data[role]['fills'].get(channel)))
                owner.extend((ym, i) for i in range(len(v)))
            series, mask = np.concatenate(pieces), np.concatenate(masks)
            fixed, bad = repair_series(series, mask)
            for start, end, reason in bad:
                (ym0, i0), (ym1, i1) = owner[start], owner[end - 1]
                problems.append(f'{role}.{channel}: {reason} ({ym0}[{i0}] to {ym1}[{i1}])')
            for index, original, value in fixed:
                ym, local = owner[index]
                per_month[ym] += 1
                repairs.append({'month': ym, 'role': role, 'channel': channel, 'index': local,
                                'original': original if np.isfinite(original) else None, 'repaired': value})
    for ym, count in per_month.items():
        if count > MAX_REPAIRS_PER_MONTH:
            problems.append(f'{ym}: {count} repaired records exceed {MAX_REPAIRS_PER_MONTH}')
    return {'repairs': repairs, 'problems': problems, 'repairs_per_month': per_month}
