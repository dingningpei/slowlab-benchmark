#!/usr/bin/env python3
"""Batch of predictor-data runs on development sites (indices kept apart from other studies).

Each site gets ``runs_per_site`` runs in distinct development years (drawn per
site), the first two with layout ``two_wave`` and the rest ``single``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEV_YEARS = (2014, 2017, 2018, 2019, 2020)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weather-cache', required=True)
    parser.add_argument('--first-site', type=int, default=100)
    parser.add_argument('--sites', type=int, required=True)
    parser.add_argument('--runs-per-site', type=int, default=3)
    parser.add_argument('--master-seed', type=int, default=20260930)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    jobs = []
    for site in range(args.first_site, args.first_site + args.sites):
        rng = np.random.default_rng([args.master_seed, site, 7])
        years = [int(y) for y in rng.choice(DEV_YEARS, size=args.runs_per_site, replace=False)]
        for k, year in enumerate(years):
            weather = ({'kind': 'development_expanded', 'cache': args.weather_cache} if year == 2014 else
                       {'cache': args.weather_cache, 'plan': 'configs/weather_gapfilled_plan.json'})
            jobs.append({'id': f'site{site:03d}_run{k}_{year}', 'year': year, 'weather': weather,
                         'layout': 'two_wave' if k < 2 else 'single', 'layout_seed': int(rng.integers(2 ** 62)),
                         'site': {'distribution': 'configs/site_distribution_v0.json',
                                  'master_seed': args.master_seed, 'site_index': site}})
    batch = {'out_dir': args.out_dir, 'script': 'scripts/generate_predictor_data.py',
             'purpose': 'process predictor training data, development sites',
             'common': {'backend': 'greenlight', 'contract': 'configs/task_contract_v7.json',
                        'sensor_noise': {'config': 'configs/sensor_noise_v0.json', 'setting': 'main'}},
             'jobs': jobs}
    args.batch.write_text(json.dumps(batch, indent=1))
    print(json.dumps({'jobs': len(jobs), 'sites': args.sites,
                      'layouts': {l: sum(j['layout'] == l for j in jobs) for l in ('two_wave', 'single')}}))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
