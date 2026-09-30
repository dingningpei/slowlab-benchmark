#!/usr/bin/env python3
"""Evaluation Monte Carlo error from the development-site study.

For every (site, policy) the score varies across evaluation weather years; for
every site and policy pair the paired difference varies too. Pooling the
within-site year variance (degrees of freedom: years - 1 per group) gives the
standard deviation of one evaluation year; the standard error with K years is
that divided by sqrt(K). Methods on a site share evaluation years, so the
paired-difference error is the one that matters for method comparisons.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def pooled_sd(groups):
    ss = sum(float(np.sum((np.asarray(g) - np.mean(g)) ** 2)) for g in groups if len(g) > 1)
    df = sum(len(g) - 1 for g in groups if len(g) > 1)
    return math.sqrt(ss / df) if df else None, df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    scores, failed, planting = {}, [], defaultdict(list)
    for path in sorted(args.out_dir.glob('site*_*.json')):
        m = re.fullmatch(r'site(\d+)_(\w+)_(\d{4})', path.stem)
        record = json.loads(path.read_text())
        if record.get('status') != 'completed':
            failed.append({'job': path.stem, 'error': record.get('error')})
            continue
        site, policy, year = int(m.group(1)), m.group(2), int(m.group(3))
        scores[(site, policy, year)] = record['score_eur_m2']
        for p in record['plantings']:
            planting[(site, policy, p['planting_day'])].append(p['mean_margin_eur_m2'])
    sites = sorted({k[0] for k in scores})
    policies = sorted({k[1] for k in scores})
    years = sorted({k[2] for k in scores})
    single = [[scores[(s, p, y)] for y in years if (s, p, y) in scores] for s in sites for p in policies]
    sd_single, df_single = pooled_sd(single)
    pairs = {}
    for p, q in itertools.combinations(policies, 2):
        groups = [[scores[(s, p, y)] - scores[(s, q, y)] for y in years if (s, p, y) in scores and (s, q, y) in scores]
                  for s in sites]
        sd, df = pooled_sd(groups)
        mean_abs = float(np.mean([abs(np.mean(g)) for g in groups if g]))
        pairs[f'{p}-{q}'] = {'sd_one_year': sd, 'df': df, 'mean_abs_site_difference': mean_abs}
    sd_pair = math.sqrt(np.mean([v['sd_one_year'] ** 2 for v in pairs.values()])) if pairs else None
    by_planting = {str(day): pooled_sd([v for (s, p, d), v in planting.items() if d == day])[0]
                   for day in sorted({k[2] for k in planting})}
    report = {'purpose': 'evaluation Monte Carlo error on development sites (decision 2026-09-30)',
              'sites': len(sites), 'policies': policies, 'years': years, 'completed': len(scores), 'failed': failed,
              'score_sd_one_year': sd_single, 'score_sd_df': df_single,
              'paired_difference_sd_one_year': sd_pair, 'pairs': pairs,
              'planting_mean_sd_one_year': by_planting,
              'standard_error_by_k': {str(k): {'score': sd_single / math.sqrt(k) if sd_single else None,
                                               'paired_difference': sd_pair / math.sqrt(k) if sd_pair else None}
                                      for k in range(1, len(years))},
              'site_means': {f'site{s:02d}_{p}': float(np.mean([scores[(s, p, y)] for y in years if (s, p, y) in scores]))
                             for s in sites for p in policies}}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('completed', 'score_sd_one_year', 'paired_difference_sd_one_year',
                                             'standard_error_by_k')}, indent=1))


if __name__ == '__main__':
    main()
