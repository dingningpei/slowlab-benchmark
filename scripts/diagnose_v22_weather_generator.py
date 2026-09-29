#!/usr/bin/env python3
"""Bounded development-only real-source check; never creates formal site seeds."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.v22_private_weather import CandidateWeatherGenerator, read_source_year, ROWS_PER_DAY
from slowlab.v22_weather_similarity import audit_nonreplay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--dev-seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    plan = ROOT / 'configs/v22_weather_gapfilled_plan.json'
    sources = {year: read_source_year(args.cache, plan, year)
               for year in (2017, 2018, 2019, 2020)}
    generator = CandidateWeatherGenerator({year: sources[year] for year in (2017, 2018)})
    attempts = []
    chosen = None
    candidate = None
    similarity = None
    for attempt in range(20):
        try:
            proposed = generator.sample_candidate(args.dev_seed + attempt)
        except ValueError as exc:
            attempts.append({'attempt': attempt, 'support': 'rejected', 'reason': str(exc)})
            continue
        checked = audit_nonreplay(proposed, sources, generator.scale, threshold=0.08)
        attempts.append({'attempt': attempt, 'support': 'accepted',
                         'nonreplay_passed': checked['passed'],
                         'nearest_24h': checked['by_window']['24h']['minimum_standardized_rmse'],
                         'nearest_6h': checked['by_window']['6h']['minimum_standardized_rmse']})
        if checked['passed']:
            chosen, candidate, similarity = attempt, proposed, checked
            break
    if candidate is None:
        report = {'status': 'all_candidate_paths_rejected',
                  'scope': 'development weather diagnostic only; no formal sites',
                  'support_attempts': attempts, 'elapsed_seconds': time.monotonic()-started}
        args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'status': report['status'], 'attempts': len(attempts),
                          'last_reason': attempts[-1]}))
        return
    import numpy as np
    generated_daily = candidate.reshape(365, ROWS_PER_DAY, 5).mean(axis=1)
    held_daily = sources[2019].reshape(365, ROWS_PER_DAY, 5).mean(axis=1)
    monthly_ends = np.cumsum([31,28,31,30,31,30,31,31,30,31,30,31])
    monthly_starts = np.r_[0, monthly_ends[:-1]]
    monthly_delta = [((generated_daily[a:b].mean(axis=0) - held_daily[a:b].mean(axis=0))
                      / generator.scale).tolist() for a,b in zip(monthly_starts,monthly_ends)]
    daily_variance_ratio = (generated_daily.var(axis=0) / held_daily.var(axis=0)).tolist()
    ten_minute_p99_ratio = (np.quantile(candidate, 0.99, axis=0)
                            / np.quantile(sources[2019], 0.99, axis=0)).tolist()
    report = {'status': 'candidate_development_diagnostic',
              'scope': '2017-18 fit; 2019 descriptive selection check; 2020 only fixed duplicate filter; no crop outcomes, no private formal seeds',
              'source_rows': {str(year):len(matrix) for year,matrix in sources.items()},
              'candidate_rows':len(candidate), 'support_attempts':attempts,
              'accepted_attempt':chosen, 'similarity':similarity,
              'monthly_standardized_mean_delta_vs_2019':monthly_delta,
              'daily_variance_ratio_vs_2019':daily_variance_ratio,
              'ten_minute_p99_ratio_vs_2019':ten_minute_p99_ratio,
              'elapsed_seconds':time.monotonic()-started}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'similarity_passed':similarity['passed'],
                      'accepted_attempt':chosen,'elapsed_seconds':report['elapsed_seconds']}))


if __name__ == '__main__':
    main()
