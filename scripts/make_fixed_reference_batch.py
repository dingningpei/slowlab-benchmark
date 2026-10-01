#!/usr/bin/env python3
"""Evaluation batch for selecting the fixed reference policy on development sites."""
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
    parser.add_argument('--master-seed', type=int, default=20260930)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--backend', default='greenlight')
    args = parser.parse_args()
    dist_path = 'configs/site_distribution_v1.json'
    distribution = json.loads((ROOT / dist_path).read_text())
    candidates = json.loads((ROOT / 'configs/fixed_reference_candidates_v0.json').read_text())['candidates']
    jobs = []
    for index in range(args.sites):
        site = {'distribution': dist_path, 'master_seed': args.master_seed, 'site_index': index}
        years = assign_weather_years(distribution, args.master_seed, index, DEVELOPMENT_YEARS, 3)['evaluation_years']
        for name, policy in candidates.items():
            for year in years:
                jobs.append({'id': f'site{index:02d}_{name}_{year}', 'site': site, 'year': year, 'policy': policy,
                             'weather': weather_spec(year, args.dev_cache)})
    batch = {'out_dir': args.out_dir, 'purpose': 'fixed-reference selection on development sites',
             'common': {'contract': 'configs/task_contract_v8.json', 'backend': args.backend,
                        **({'sensor_noise': {'config': 'configs/sensor_noise_v0.json', 'setting': 'main'}}
                           if args.backend == 'greenlight' else {})},
             'jobs': jobs}
    args.batch.write_text(json.dumps(batch, indent=1))
    print(json.dumps({'jobs': len(jobs)}))


if __name__ == '__main__':
    main()
