#!/usr/bin/env python3
"""Bounded weather-only leave-one-year-out development pilot for v3."""
from __future__ import annotations

import argparse
import calendar
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.v22_private_weather import CHANNELS, ROWS_PER_DAY, read_source_year
from slowlab.v22_private_weather_v3 import BlockWeatherGeneratorV3
from slowlab.v22_weather_similarity import audit_nonreplay


def calendar_path(path, year, np):
    if year == 2020:
        path = np.concatenate((path[:59 * ROWS_PER_DAY], path[60 * ROWS_PER_DAY:]))
    if len(path) != 365 * ROWS_PER_DAY:
        raise ValueError('expected 365 calendar-aligned days')
    return path


def features(path, scale, np):
    daily = path.reshape(365, ROWS_PER_DAY, len(CHANNELS)).mean(axis=1)
    ends = np.cumsum([31,28,31,30,31,30,31,31,30,31,30,31])
    starts = np.r_[0, ends[:-1]]
    monthly = np.array([daily[a:b].mean(axis=0) for a,b in zip(starts,ends)])
    variance = daily.var(axis=0)
    p99 = np.quantile(path, .99, axis=0)
    correlation = np.corrcoef(daily, rowvar=False)
    upper = correlation[np.triu_indices(len(CHANNELS), 1)]
    vector = np.r_[((monthly / scale).ravel() / math.sqrt(60)),
                   (np.log(np.maximum(variance, 1e-12)) / math.sqrt(5)),
                   ((p99 / scale) / math.sqrt(5)), upper / math.sqrt(10)]
    return {'vector': vector, 'monthly': monthly, 'daily_variance': variance,
            'p99': p99, 'correlation': correlation}


def energy_score(rows, truth, np):
    rows = np.asarray(rows)
    first = np.linalg.norm(rows - truth, axis=1).mean()
    second = np.linalg.norm(rows[:,None,:] - rows[None,:,:], axis=2).mean() / 2
    return float(first - second)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidates-per-fold', type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.candidates_per_fold <= 32:
        raise ValueError('bounded development pilot requires 1-32 candidates per fold')
    import numpy as np
    started = time.monotonic()
    plan = ROOT / 'configs/v22_weather_gapfilled_plan.json'
    years = {year: read_source_year(args.cache, plan, year)
             for year in (2017,2018,2019,2020)}
    folds = []
    for heldout in years:
        training = {year:path for year,path in years.items() if year != heldout}
        generator = BlockWeatherGeneratorV3(training)
        truth = features(calendar_path(years[heldout], heldout, np), generator.scale, np)
        baseline = [features(calendar_path(path, year, np), generator.scale, np)
                    for year,path in training.items()]
        rows = []
        generated = []
        for offset in range(args.candidates_per_fold):
            seed = 20261000 + heldout * 100 + offset
            entry = {'development_seed':seed}
            try:
                path = generator.sample_candidate(seed)
            except ValueError as exc:
                entry.update(physical_support=False, nonreplay=None,
                             support_error=str(exc))
                rows.append(entry)
                continue
            entry['physical_support'] = True
            check = audit_nonreplay(path, training, generator.scale, threshold=.08)
            entry['nonreplay'] = check['passed']
            entry['minimum_6h_distance'] = check['by_window']['6h']['minimum_standardized_rmse']
            if check['passed']:
                stats = features(path, generator.scale, np)
                entry['wind_daily_variance_ratio_vs_heldout'] = float(
                    stats['daily_variance'][2] / truth['daily_variance'][2])
                generated.append(stats)
            rows.append(entry)
        fold = {'heldout_year':heldout, 'fit_years':list(training), 'rows':rows,
                'physical_support_count':sum(row['physical_support'] for row in rows),
                'nonreplay_count':len(generated)}
        if generated:
            matrices = np.stack([row['monthly'] for row in generated])
            low, high = np.quantile(matrices, [.05,.95], axis=0)
            fold['monthly_5_95_coverage'] = float(np.mean((truth['monthly'] >= low) & (truth['monthly'] <= high)))
            fold['weather_energy_score'] = energy_score([row['vector'] for row in generated], truth['vector'], np)
            fold['whole_year_resample_baseline_energy_score'] = energy_score(
                [row['vector'] for row in baseline], truth['vector'], np)
            fold['generated_wind_daily_variance_ratio_range'] = [
                min(row['wind_daily_variance_ratio_vs_heldout'] for row in rows if row['nonreplay']),
                max(row['wind_daily_variance_ratio_vs_heldout'] for row in rows if row['nonreplay'])]
        folds.append(fold)
    report = {'scope':'2017-2020 same-station weather-only v3 development pilot; no formal sites, holdout acquisition or crop outcomes',
              'candidate_version':'block-weather-v3-development',
              'candidates_per_fold':args.candidates_per_fold,
              'vector_score':'monthly means/channel train SD/sqrt(60); log daily variance/sqrt(5); 10-minute p99/channel train SD/sqrt(5); 10 daily cross-channel correlations/sqrt(10)',
              'leap_rule':'remove Feb 29 only for calendar-aligned development fit and scoring; original 2020 diagnostic retained',
              'folds':folds,'elapsed_seconds':time.monotonic()-started}
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'folds':len(folds),'accepted':[f['nonreplay_count'] for f in folds],
                      'elapsed_seconds':report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
