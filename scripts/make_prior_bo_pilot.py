#!/usr/bin/env python3
"""Rerun of the frozen main baseline (prior-informed local BO) on the pilot sites (private: holds the pilot seed).

    make_prior_bo_pilot.py campaigns   --secret SEED.json --run-dir DIR --dev-cache C --formal-cache F
    make_prior_bo_pilot.py evaluations --run-dir DIR --dev-cache C --formal-cache F

Same sites, campaign years and evaluation years as the Phase 4 pilot; each site x method seed x
Full/Endpoint. Evaluation job ids follow the pilot naming (site{ii}_pbo_s{seed}_{condition}_y{year})
so scripts/analyse_pilot_evaluations.py reads them next to the pilot evaluations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.private_runs import weather_spec  # noqa: E402
from slowlab.site_distribution import formal_weather_years  # noqa: E402

CONFIG = 'configs/prior_bo_v1.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('campaigns', 'evaluations'))
    ap.add_argument('--secret', type=Path)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--dev-cache', required=True)
    ap.add_argument('--formal-cache', required=True)
    ap.add_argument('--seeds', type=int, default=2)
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()
    if args.stage == 'campaigns':
        secret = json.loads(args.secret.read_text())
        if secret['partition'] != 'pilot' or secret['distribution_id'] != 'slowlab-site-distribution-v1':
            raise SystemExit('not the pilot master seed for distribution v1')
        distribution = json.loads((ROOT / 'configs/site_distribution_v1.json').read_text())
        radius = json.loads((ROOT / CONFIG).read_text())['selected_radius']
        jobs = []
        for index in range(secret['n_sites']):
            site = {'distribution': 'configs/site_distribution_v1.json', 'master_seed': secret['master_seed'],
                    'site_index': index}
            years = formal_weather_years(distribution, secret['master_seed'], index)
            for seed in range(args.seeds):
                for feedback in ('full', 'endpoint'):
                    job = f'site{index:02d}_pbo_s{seed}_{feedback}'
                    jobs.append({'id': job, 'method': 'bo', 'site': site, 'year': years['campaign_year'],
                                 'feedback': feedback, 'evaluation_years': years['evaluation_years'],
                                 'private_dir': str(run_dir / 'private' / job),
                                 'bo': {'agent': 'prior_local', 'config': CONFIG, 'radius': radius, 'seed': 4000 + seed}})
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / 'campaigns.json'
        path.write_text(json.dumps({'out_dir': str(run_dir / 'campaigns'), 'script': 'scripts/run_campaign_job.py',
                                    'purpose': 'main baseline rerun on the pilot sites (private)',
                                    'common': {'contract': 'configs/task_contract_v8.json', 'backend': 'greenlight',
                                               'dev_cache': args.dev_cache, 'formal_cache': args.formal_cache},
                                    'jobs': jobs}, indent=1))
        path.chmod(0o600)
        print(json.dumps({'campaigns': len(jobs)}))
        return
    batch = json.loads((run_dir / 'campaigns.json').read_text())
    jobs, missing = [], []
    for job in batch['jobs']:
        record_path = Path(batch['out_dir']) / f"{job['id']}.json"
        record = json.loads(record_path.read_text()) if record_path.exists() else {}
        if record.get('status') != 'completed':
            missing.append(job['id'])
            continue
        for year in job['evaluation_years']:
            jobs.append({'id': f"{job['id']}_y{year}", 'site': job['site'], 'year': year,
                         'settlement': str(Path(job['private_dir']) / 'settlement.json'),
                         'weather': weather_spec(year, args.dev_cache, args.formal_cache)})
    path = run_dir / 'evaluations.json'
    path.write_text(json.dumps({'out_dir': str(run_dir / 'evaluations'),
                                'purpose': 'main baseline rerun on the pilot sites: evaluations (private)',
                                'common': {'contract': 'configs/task_contract_v8.json', 'backend': 'greenlight',
                                           'sensor_noise': {'config': 'configs/sensor_noise_v0.json', 'setting': 'main'}},
                                'jobs': jobs, 'campaigns_not_completed': missing}, indent=1))
    path.chmod(0o600)
    print(json.dumps({'evaluations': len(jobs), 'campaigns_not_completed': missing}))


if __name__ == '__main__':
    main()
