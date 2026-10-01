#!/usr/bin/env python3
"""Development variance study: BO campaigns with repeats on development sites, then their evaluation.

Writes two batches for scripts/run_evaluation_batch.py:
* campaigns: each development site x BO seed x feedback condition (staggered BO, which uses process information);
* evaluations: each campaign's recommendation on the site's three development evaluation years.
Development sites come from distribution v1 with the development master seed, so they are apart from every
formal site; weather comes from the development pool (2014, 2017-2020).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.private_runs import DEVELOPMENT_YEARS, weather_spec  # noqa: E402
from slowlab.site_distribution import assign_weather_years  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-cache', required=True)
    parser.add_argument('--sites', type=int, default=8)
    parser.add_argument('--seeds', type=int, default=2)
    parser.add_argument('--master-seed', type=int, default=20260930)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--backend', default='greenlight')
    args = parser.parse_args()
    dist_path = 'configs/site_distribution_v1.json'
    distribution = json.loads((ROOT / dist_path).read_text())
    contract = 'configs/task_contract_v8.json'
    campaigns, evaluations = [], []
    for index in range(args.sites):
        site = {'distribution': dist_path, 'master_seed': args.master_seed, 'site_index': index}
        years = assign_weather_years(distribution, args.master_seed, index, DEVELOPMENT_YEARS, 3)
        for seed in range(args.seeds):
            for feedback in ('full', 'endpoint'):
                job = f'site{index:02d}_seed{seed}_{feedback}'
                private = args.run_dir / 'private' / job
                campaigns.append({'id': job, 'method': 'bo', 'bo': {'schedule': 'staggered', 'seed': 1000 + seed},
                                  'site': site, 'year': years['campaign_year'], 'feedback': feedback,
                                  'private_dir': str(private)})
                for year in years['evaluation_years']:
                    evaluations.append({'id': f'{job}_eval{year}', 'site': site, 'year': year,
                                        'settlement': str(private / 'settlement.json'),
                                        'weather': weather_spec(year, args.dev_cache)})
    common = {'contract': contract, 'backend': args.backend, 'dev_cache': args.dev_cache}
    (args.run_dir).mkdir(parents=True, exist_ok=True)
    (args.run_dir / 'campaigns.json').write_text(json.dumps(
        {'out_dir': str(args.run_dir / 'campaigns'), 'script': 'scripts/run_campaign_job.py',
         'purpose': 'development variance study: staggered BO campaigns', 'common': common, 'jobs': campaigns}, indent=1))
    (args.run_dir / 'evaluations.json').write_text(json.dumps(
        {'out_dir': str(args.run_dir / 'evaluations'), 'purpose': 'development variance study: evaluations',
         'common': {'contract': contract, 'backend': args.backend,
                    **({'sensor_noise': {'config': 'configs/sensor_noise_v0.json', 'setting': 'main'}}
                       if args.backend == 'greenlight' else {})},
         'jobs': evaluations}, indent=1))
    print(json.dumps({'campaigns': len(campaigns), 'evaluations': len(evaluations)}))


if __name__ == '__main__':
    main()
