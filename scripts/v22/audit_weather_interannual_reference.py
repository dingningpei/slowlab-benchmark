#!/usr/bin/env python3
"""Describe observed 2017-2020 interannual weather spread; no generator fitting."""
from __future__ import annotations

import argparse
import calendar
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from slowlab.archive.private_weather import CHANNELS, ROWS_PER_DAY, read_source_year


def features(path, np, year):
    daily = path.reshape(365, ROWS_PER_DAY, len(CHANNELS)).mean(axis=1)
    month_days = [calendar.monthrange(year, month)[1] for month in range(1, 13)]
    if year == 2020:
        month_days[-1] -= 1  # Dec 31 is excluded from the fixed 365-day window.
    month_ends = np.cumsum(month_days)
    month_starts = np.r_[0, month_ends[:-1]]
    return {'monthly_means': np.array([daily[a:b].mean(axis=0) for a, b in zip(month_starts, month_ends)]),
            'daily_variance': daily.var(axis=0),
            'ten_minute_p99': np.quantile(path, .99, axis=0),
            'daily_correlation': np.corrcoef(daily, rowvar=False)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    plan = ROOT / 'configs/v22/weather_gapfilled_plan.json'
    years = {year: read_source_year(args.cache, plan, year)[:365 * ROWS_PER_DAY]
             for year in (2017, 2018, 2019, 2020)}
    scale = np.std(np.vstack((years[2017], years[2018])), axis=0, ddof=1)
    summary = {year: features(path, np, year) for year, path in years.items()}
    pairs = []
    for a, b in itertools.combinations(years, 2):
        fa, fb = summary[a], summary[b]
        monthly = np.abs((fa['monthly_means'] - fb['monthly_means']) / scale)
        variance_ratio = fa['daily_variance'] / fb['daily_variance']
        p99_ratio = fa['ten_minute_p99'] / fb['ten_minute_p99']
        pairs.append({'years': [a, b],
                      'max_abs_monthly_mean_training_sd': float(monthly.max()),
                      'max_abs_monthly_by_channel': monthly.max(axis=0).tolist(),
                      'daily_variance_ratio_by_channel': variance_ratio.tolist(),
                      'ten_minute_p99_ratio_by_channel': p99_ratio.tolist(),
                      'max_abs_daily_correlation_delta': float(np.max(np.abs(fa['daily_correlation'] - fb['daily_correlation']))),
                      'daily_correlation_delta_matrix': (fa['daily_correlation'] - fb['daily_correlation']).tolist()})
    report = {'scope': 'descriptive 2017-2020 same-station interannual reference; 2020 already opened in v2 development, not an untouched holdout',
              'channels': list(CHANNELS), 'year_rule': 'first 365 local-standard-time days; 2020 includes Feb 29, excludes Dec 31',
              'normalization_fit_years': [2017, 2018], 'pair_count': len(pairs), 'pairs': pairs,
              'limitation': 'Four annual observations from one station do not identify a general weather distribution or independently validate a site generator.'}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'pair_count': len(pairs), 'max_natural_monthly_sd': max(p['max_abs_monthly_mean_training_sd'] for p in pairs)}))


if __name__ == '__main__':
    main()
