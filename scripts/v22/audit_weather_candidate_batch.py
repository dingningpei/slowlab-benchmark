#!/usr/bin/env python3
"""Predeclared 12-seed development audit; no crop outcomes or formal site identities."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from slowlab.archive.private_weather import CandidateWeatherGenerator, ROWS_PER_DAY, read_source_year
from slowlab.archive.weather_similarity import audit_nonreplay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--comparison-year', type=int, choices=(2019, 2020), default=2019)
    args = parser.parse_args()
    import numpy as np
    started = time.monotonic()
    plan = ROOT / 'configs/v22/weather_gapfilled_plan.json'
    sources = {year: read_source_year(args.cache, plan, year)
               for year in (2017, 2018, 2019, 2020)}
    generator = CandidateWeatherGenerator({year: sources[year] for year in (2017, 2018)})
    held = sources[args.comparison_year][:365 * ROWS_PER_DAY]
    held_daily = held.reshape(365, ROWS_PER_DAY, 5).mean(axis=1)
    ends = np.cumsum([31,28,31,30,31,30,31,31,30,31,30,31])
    starts = np.r_[0, ends[:-1]]
    rows = []
    for offset in range(12):
        seed = 20260928 + offset
        entry = {'development_seed': seed}
        try:
            path = generator.sample_candidate(seed)
        except ValueError as exc:
            entry.update({'physical_support': False, 'nonreplay': None,
                          'support_error': str(exc)})
            rows.append(entry)
            continue
        entry['physical_support'] = True
        checked = audit_nonreplay(path, sources, generator.scale, threshold=0.08)
        entry['nonreplay'] = checked['passed']
        entry['minimum_standardized_rmse'] = {
            window: record['minimum_standardized_rmse']
            for window, record in checked['by_window'].items()}
        if checked['passed']:
            daily = path.reshape(365, ROWS_PER_DAY, 5).mean(axis=1)
            entry['daily_variance_ratio_vs_reference'] = (daily.var(axis=0) / held_daily.var(axis=0)).tolist()
            entry['ten_minute_p99_ratio_vs_reference'] = (np.quantile(path, .99, axis=0) / np.quantile(held, .99, axis=0)).tolist()
            monthly = [((daily[a:b].mean(axis=0) - held_daily[a:b].mean(axis=0)) / generator.scale)
                       for a,b in zip(starts, ends)]
            entry['max_abs_monthly_standardized_mean_delta_vs_reference'] = float(np.max(np.abs(monthly)))
            correlations = np.corrcoef(daily, rowvar=False)
            held_correlations = np.corrcoef(held_daily, rowvar=False)
            entry['max_abs_daily_cross_channel_correlation_delta_vs_reference'] = float(
                np.max(np.abs(correlations - held_correlations)))
        rows.append(entry)
    result = {'scope': 'weather-only development audit; no formal private sites or crop outcomes',
              'version': 'candidate-v2', 'comparison_year': args.comparison_year,
              'versioned_rule': 'first 365 local-standard-time days; 2020 retains Feb 29 and excludes Dec 31',
              'fixed_seed_start': 20260928,
              'fixed_seed_count': 12, 'support_passes': sum(row['physical_support'] for row in rows),
              'nonreplay_passes': sum(row['nonreplay'] is True for row in rows),
              'rows': rows, 'elapsed_seconds': time.monotonic() - started}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('support_passes', 'nonreplay_passes', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
