#!/usr/bin/env python3
"""Verify the pre-2020 weather lock against the immutable diagnostic report."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / 'configs/v22/weather_generator_2020_diagnostic_lock.json'
RESULT = ROOT / 'results/v22/weather_candidate_v2_2020_diagnostic.json'
OUT = ROOT / 'results/v22/weather_candidate_v2_2020_gate.json'


def main() -> None:
    lock = json.loads(LOCK.read_text())
    result = json.loads(RESULT.read_text())
    for name, expected in lock['file_sha256'].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'locked input hash changed: {name}')
    if result['comparison_year'] != lock['diagnostic_year'] or result['version'] != 'candidate-v2':
        raise ValueError('diagnostic identity differs from lock')
    if [row['development_seed'] for row in result['rows']] != lock['development_seeds']:
        raise ValueError('diagnostic seed order differs from lock')
    accepted = [row for row in result['rows'] if row['nonreplay'] is True]
    if result['support_passes'] != sum(row['physical_support'] for row in result['rows']):
        raise ValueError('physical support count mismatch')
    if result['nonreplay_passes'] != len(accepted) or not accepted:
        raise ValueError('non-replay count mismatch')
    t = lock['thresholds']
    values = {
        'physical_support_count': result['support_passes'],
        'all_source_nonreplay_count': result['nonreplay_passes'],
        'max_abs_monthly_mean_training_sd': max(row['max_abs_monthly_standardized_mean_delta_vs_reference'] for row in accepted),
        'daily_variance_ratio_min': min(x for row in accepted for x in row['daily_variance_ratio_vs_reference']),
        'daily_variance_ratio_max': max(x for row in accepted for x in row['daily_variance_ratio_vs_reference']),
        'ten_minute_p99_ratio_min': min(x for row in accepted for x in row['ten_minute_p99_ratio_vs_reference']),
        'ten_minute_p99_ratio_max': max(x for row in accepted for x in row['ten_minute_p99_ratio_vs_reference']),
        'max_abs_daily_cross_channel_correlation_delta': max(row['max_abs_daily_cross_channel_correlation_delta_vs_reference'] for row in accepted),
    }
    checks = {
        'physical_support': values['physical_support_count'] >= t['physical_support_count_min'],
        'nonreplay': values['all_source_nonreplay_count'] >= t['all_source_nonreplay_count_min'],
        'monthly_mean': values['max_abs_monthly_mean_training_sd'] <= t['among_accepted_max_abs_monthly_mean_training_sd'],
        'daily_variance': t['among_accepted_daily_variance_ratio_range'][0] <= values['daily_variance_ratio_min']
                          and values['daily_variance_ratio_max'] <= t['among_accepted_daily_variance_ratio_range'][1],
        'ten_minute_p99': t['among_accepted_ten_minute_p99_ratio_range'][0] <= values['ten_minute_p99_ratio_min']
                           and values['ten_minute_p99_ratio_max'] <= t['among_accepted_ten_minute_p99_ratio_range'][1],
        'daily_correlation': values['max_abs_daily_cross_channel_correlation_delta'] <= t['among_accepted_max_abs_daily_cross_channel_correlation_delta'],
    }
    report = {'status': 'passed' if all(checks.values()) else 'failed_cross_year_weather_gate',
              'scope': '2020 weather-only diagnostic; no greenhouse trajectory or agent outcome validation',
              'lock_sha256': hashlib.sha256(LOCK.read_bytes()).hexdigest(),
              'diagnostic_sha256': hashlib.sha256(RESULT.read_bytes()).hexdigest(),
              'values': values, 'checks': checks}
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'checks': checks}))


if __name__ == '__main__':
    main()
