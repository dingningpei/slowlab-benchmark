#!/usr/bin/env python3
"""Batch for the evaluation Monte Carlo error study on development sites.

Development sites use a development master seed (not a formal seed). Each
(site, policy) is evaluated in every development weather year: 2014 via the
expanded development archive and 2017-2020 via the original archive. The
spread across years of a policy's score, and of the paired difference between
two policies, gives the standard error for K evaluation years.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV_YEARS = (2014, 2017, 2018, 2019, 2020)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weather-cache', required=True)
    parser.add_argument('--sites', type=int, required=True)
    parser.add_argument('--master-seed', type=int, default=20260930)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    examples = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
    from slowlab.scripted_model import FIRST_WAVE
    policies = {'a': examples['policy_a'], 'b': examples['policy_b'], 'c': FIRST_WAVE[1]}
    jobs = []
    for site in range(args.sites):
        for name, policy in policies.items():
            for year in DEV_YEARS:
                weather = ({'kind': 'development_expanded', 'cache': args.weather_cache} if year == 2014 else
                           {'cache': args.weather_cache, 'plan': 'configs/weather_gapfilled_plan.json'})
                jobs.append({'id': f'site{site:02d}_{name}_{year}', 'policy': policy, 'year': year, 'weather': weather,
                             'site': {'distribution': 'configs/site_distribution_v0.json',
                                      'master_seed': args.master_seed, 'site_index': site}})
    batch = {'out_dir': args.out_dir, 'purpose': 'evaluation Monte Carlo error study, development sites',
             'common': {'backend': 'greenlight', 'contract': 'configs/task_contract_v8.json',
                        'sensor_noise': {'config': 'configs/sensor_noise_v0.json', 'setting': 'main'}},
             'jobs': jobs}
    args.batch.write_text(json.dumps(batch, indent=1))
    print(json.dumps({'jobs': len(jobs), 'sites': args.sites, 'policies': list(policies), 'years': DEV_YEARS}))


if __name__ == '__main__':
    sys_path = str(ROOT)
    import sys
    sys.path.insert(0, sys_path)
    main()
