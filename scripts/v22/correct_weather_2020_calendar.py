#!/usr/bin/env python3
"""Erratum: recompute 2020 monthly diagnostics using leap-year month boundaries."""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from slowlab.archive.private_weather import CandidateWeatherGenerator, CHANNELS, ROWS_PER_DAY, read_source_year


def month_means(matrix, year, np):
    days = [calendar.monthrange(year, month)[1] for month in range(1, 13)]
    if year == 2020:
        days[-1] -= 1
    assert sum(days) == 365
    daily = matrix[:365 * ROWS_PER_DAY].reshape(365, ROWS_PER_DAY, 5).mean(axis=1)
    ends = np.cumsum(days)
    starts = np.r_[0, ends[:-1]]
    return np.asarray([daily[a:b].mean(axis=0) for a, b in zip(starts, ends)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    lock = json.loads((ROOT / 'configs/v22/weather_generator_2020_diagnostic_lock.json').read_text())
    old = json.loads((ROOT / 'results/v22/weather_candidate_v2_2020_diagnostic.json').read_text())
    for name in ('slowlab/archive/private_weather.py', 'configs/v22/weather_generator_protocol_v2.json'):
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != lock['file_sha256'][name]:
            raise ValueError('generator differs from locked v2')
    plan = ROOT / 'configs/v22/weather_gapfilled_plan.json'
    sources = {year: read_source_year(args.cache, plan, year) for year in (2017, 2018, 2020)}
    generator = CandidateWeatherGenerator({year: sources[year] for year in (2017, 2018)})
    reference = month_means(sources[2020], 2020, np)
    rows = []
    for previous in old['rows']:
        if previous['nonreplay'] is not True:
            continue
        seed = previous['development_seed']
        generated = generator.sample_candidate(seed)
        monthly = np.abs((month_means(generated, 2017, np) - reference) / generator.scale)
        location = np.unravel_index(np.argmax(monthly), monthly.shape)
        rows.append({'development_seed': seed,
                     'corrected_max_abs_monthly_training_sd': float(monthly[location]),
                     'max_month': int(location[0] + 1),
                     'max_channel': CHANNELS[location[1]],
                     'previous_misgrouped_max': previous['max_abs_monthly_standardized_mean_delta_vs_reference']})
    result = {'status': 'calendar_erratum_recomputed',
              'scope': 'v2 2020 monthly diagnostic only; old report preserved, 2020 no longer untouched',
              'reason': '2020 previous monthly comparison used non-leap month ends despite including Feb 29',
              'reference_month_days': [calendar.monthrange(2020, m)[1] - (m == 12) for m in range(1, 13)],
              'generated_month_days': [calendar.monthrange(2017, m)[1] for m in range(1, 13)],
              'rows': rows,
              'corrected_max': max(row['corrected_max_abs_monthly_training_sd'] for row in rows),
              'locked_monthly_limit': lock['thresholds']['among_accepted_max_abs_monthly_mean_training_sd'],
              'unchanged_daily_variance_gate': 'failed in original report; independent of monthly partition'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'corrected_max': result['corrected_max'],
                      'monthly_gate_passed': result['corrected_max'] <= result['locked_monthly_limit']}))


if __name__ == '__main__':
    main()
