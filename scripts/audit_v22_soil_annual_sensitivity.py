#!/usr/bin/env python3
"""Audit paired 8/20 C annual deep-soil scenarios without claiming calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

RESOURCES = ('heat_kwh_m2', 'light_kwh_m2', 'co2_kg_m2',
             'harvest_kg_m2', 'transpiration_kg_m2')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arm(result_path: Path, progress_path: Path, soil_c: int) -> dict:
    result = json.loads(result_path.read_text())
    rows = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert result['status'] == 'passed_annual_runtime_pilot'
    assert result['pilot_days'] == result['completed_days'] == 365
    assert result['completed_steps'] == 365 * 288
    assert result['soil_boundary_c'] == soil_c
    assert result['origin_utc'] == '2016-12-31T23:00:00+00:00'
    assert result['elapsed_seconds'] < result['max_seconds']
    assert result['peak_rss_kib_linux'] * 1024 < result['max_rss_bytes']
    assert len(rows) == 365
    for day, row in enumerate(rows, 1):
        assert row['day'] == day and row['clock_seconds'] == day * 86400
        assert 0 < row['rss_bytes'] < result['max_rss_bytes']
        for prefix in ('air_c', 'rh_pct'):
            low, mid, high = (row[f'{prefix}_{suffix}'] for suffix in ('min', 'mean', 'max'))
            assert all(math.isfinite(value) for value in (low, mid, high))
            assert low <= mid <= high
        assert set(row['cumulative_per_m2']) == set(result['ledger']['per_m2'])
    for key in result['ledger']['per_m2']:
        assert math.isclose(rows[-1]['cumulative_per_m2'][key],
                            result['ledger']['per_m2'][key], abs_tol=1e-7)
    return {'result': result, 'rows': rows,
            'result_sha256': digest(result_path), 'progress_sha256': digest(progress_path)}


def audit(arm8: dict, arm20: dict) -> dict:
    left, right = arm8['result'], arm20['result']
    for field in ('pilot_days', 'schedule_days', 'origin_utc', 'native_rhs', 'array_output'):
        assert left[field] == right[field], field
    lrows, rrows = arm8['rows'], arm20['rows']
    for day in range(365):
        assert lrows[day]['day'] == rrows[day]['day']
        assert lrows[day]['phase'] == rrows[day]['phase']
    climate = {}
    for name in ('air_c_mean', 'rh_pct_mean'):
        differences = [a[name] - b[name] for a, b in zip(lrows, rrows)]
        climate[name] = {'annual_mean_8': sum(a[name] for a in lrows) / 365,
                         'annual_mean_20': sum(b[name] for b in rrows) / 365,
                         'mean_difference_8_minus_20': sum(differences) / 365,
                         'max_abs_daily_mean_difference': max(map(abs, differences))}
    resources = {}
    for key in RESOURCES:
        a = left['ledger']['per_m2'][key]
        b = right['ledger']['per_m2'][key]
        resources[key] = {'soil_8': a, 'soil_20': b,
                          'difference_8_minus_20': a - b,
                          'relative_difference_vs_20': (a - b) / b if b else None}
    return {'status': 'passed_paired_structural_audit',
            'scope': '2017 Cabauw weather, fixed single-compartment policy, uncalibrated 8/20 C deep-soil scenario; not real-greenhouse validation',
            'weather_year': 2017,
            'days': 365,
            'arm_hashes': {'8': {'result': arm8['result_sha256'], 'progress': arm8['progress_sha256']},
                           '20': {'result': arm20['result_sha256'], 'progress': arm20['progress_sha256']}},
            'climate': climate, 'resources': resources,
            'synthetic_margin_eur': {'soil_8': left['ledger']['synthetic_margin_eur'],
                                     'soil_20': right['ledger']['synthetic_margin_eur'],
                                     'difference_8_minus_20': left['ledger']['synthetic_margin_eur'] - right['ledger']['synthetic_margin_eur']},
            'max_rss_bytes': {'soil_8': left['peak_rss_kib_linux'] * 1024,
                              'soil_20': right['peak_rss_kib_linux'] * 1024}}


def main() -> None:
    parser = argparse.ArgumentParser()
    for soil in (8, 20):
        parser.add_argument(f'--result-{soil}', type=Path, required=True)
        parser.add_argument(f'--progress-{soil}', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit(load_arm(args.result_8, args.progress_8, 8),
                   load_arm(args.result_20, args.progress_20, 20))
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'climate': report['climate'],
                      'resources': report['resources']}))


if __name__ == '__main__':
    main()
