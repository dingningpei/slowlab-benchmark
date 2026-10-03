#!/usr/bin/env python3
"""Radius selection for the prior-informed local BO on development sites (decision 2026-10-03).

    make_prior_bo_selection.py campaigns   --run-dir DIR --dev-cache CACHE --config configs/prior_bo_v0.json
    make_prior_bo_selection.py evaluations --run-dir DIR --dev-cache CACHE

campaigns: each candidate radius x development site (distribution v1, development master seed, sites 0-7)
x method seed x feedback condition, on the site's development campaign year.
evaluations: built after the campaigns finish; each recommendation on the site's three development
evaluation years. A site's evaluation of a policy in a year is deterministic (the site fixes the
evaluation draws), so a policy recommended more than once on a site is evaluated once and shared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.private_runs import DEVELOPMENT_YEARS, weather_spec  # noqa: E402
from slowlab.site_distribution import assign_weather_years  # noqa: E402

DIST = 'configs/site_distribution_v1.json'
CONTRACT = 'configs/task_contract_v8.json'


def policy_key(policy):
    return hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('campaigns', 'evaluations'))
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--dev-cache', required=True)
    ap.add_argument('--config', default='configs/prior_bo_v0.json')
    ap.add_argument('--sites', type=int, default=8)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--master-seed', type=int, default=20260930)
    args = ap.parse_args()
    distribution = json.loads((ROOT / DIST).read_text())
    run_dir = args.run_dir.resolve()
    if args.stage == 'campaigns':
        radii = json.loads((ROOT / args.config).read_text())['candidate_radii']
        jobs = []
        for index in range(args.sites):
            site = {'distribution': DIST, 'master_seed': args.master_seed, 'site_index': index}
            years = assign_weather_years(distribution, args.master_seed, index, DEVELOPMENT_YEARS, 3)
            for radius in radii:
                for seed in range(args.seeds):
                    for feedback in ('full', 'endpoint'):
                        job = f'site{index:02d}_r{int(round(radius * 100)):02d}_seed{seed}_{feedback}'
                        jobs.append({'id': job, 'method': 'bo', 'site': site, 'year': years['campaign_year'],
                                     'feedback': feedback, 'private_dir': str(run_dir / 'private' / job),
                                     'bo': {'agent': 'prior_local', 'config': args.config, 'radius': radius,
                                            'seed': 3000 + seed},
                                     'evaluation_years': years['evaluation_years']})
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'campaigns.json').write_text(json.dumps(
            {'out_dir': str(run_dir / 'campaigns'), 'script': 'scripts/run_campaign_job.py',
             'purpose': 'prior BO radius selection on development sites: campaigns',
             'common': {'contract': CONTRACT, 'backend': 'greenlight', 'dev_cache': args.dev_cache}, 'jobs': jobs},
            indent=1))
        print(json.dumps({'campaigns': len(jobs)}))
        return
    batch = json.loads((run_dir / 'campaigns.json').read_text())
    jobs, uses, missing = [], {}, []
    for job in batch['jobs']:
        path = Path(batch['out_dir']) / f"{job['id']}.json"
        record = json.loads(path.read_text()) if path.exists() else {}
        if record.get('status') != 'completed':
            missing.append(job['id'])
            continue
        policy = record['bo']['recommendation']
        index = job['site']['site_index']
        for year in job['evaluation_years']:
            eval_id = f"site{index:02d}_{policy_key(policy)}_y{year}"
            uses.setdefault(eval_id, []).append(job['id'])
            if len(uses[eval_id]) == 1:
                jobs.append({'id': eval_id, 'site': job['site'], 'year': year, 'policy': policy,
                             'weather': weather_spec(year, args.dev_cache)})
    (run_dir / 'evaluations.json').write_text(json.dumps(
        {'out_dir': str(run_dir / 'evaluations'), 'purpose': 'prior BO radius selection: evaluations',
         'common': {'contract': CONTRACT, 'backend': 'greenlight',
                    'sensor_noise': {'config': 'configs/sensor_noise_v0.json', 'setting': 'main'}},
         'jobs': jobs, 'uses': uses, 'campaigns_not_completed': missing}, indent=1))
    print(json.dumps({'evaluations': len(jobs), 'campaign_recommendations': len(batch['jobs']) - len(missing),
                      'campaigns_not_completed': missing}))


if __name__ == '__main__':
    main()
