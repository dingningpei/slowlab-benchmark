#!/usr/bin/env python3
"""Evaluation batch for the pilot (private: built from the pilot campaign batch, which holds the pilot seed).

For every completed campaign job it evaluates, on the site's three formal
evaluation years: LLM Full and Endpoint recommendations (from the private
settlements), the initial recommendation of repeat 0, BO recommendations, and
the frozen fixed reference once per site. Jobs whose campaign is not completed
are left out and listed; rerun this script after retries.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.private_runs import weather_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaigns', type=Path, required=True, help='private pilot campaigns.json')
    ap.add_argument('--dev-cache', required=True)
    ap.add_argument('--formal-cache', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    batch = json.loads(args.campaigns.read_text())
    out_dir = Path(batch['out_dir'])
    ref = json.loads((ROOT / 'configs/fixed_reference_v1.json').read_text())
    jobs, missing, sites = [], [], {}

    def add(job_id, site, years, **target):
        for y in years:
            jobs.append({'id': f'{job_id}_y{y}', 'site': site, 'year': y,
                         'weather': weather_spec(y, args.dev_cache, args.formal_cache), **target})
    for job in batch['jobs']:
        rec_path = out_dir / f"{job['id']}.json"
        record = json.loads(rec_path.read_text()) if rec_path.exists() else {}
        if record.get('status') != 'completed':
            missing.append(job['id'])
            continue
        site, years = job['site'], job['evaluation_years']
        sites[site['site_index']] = (site, years)
        private = Path(job['private_dir'])
        if job['method'] == 'llm':
            for branch in ('full', 'endpoint'):
                add(f"{job['id']}_{branch}", site, years, settlement=str(private / branch / 'settlement.json'))
            if job['id'].endswith('_r0'):
                add(f"{job['id']}_initial", site, years, policy=record['initial_recommendation'])
        else:
            add(job['id'], site, years, settlement=str(private / 'settlement.json'))
    for index, (site, years) in sorted(sites.items()):
        add(f'site{index:02d}_fixed_reference', site, years, policy=ref['policy'])
    eval_batch = {'out_dir': str(args.out.parent / 'evaluations'), 'purpose': 'Phase 4 pilot evaluations (private)',
                  'common': {'contract': 'configs/task_contract_v8.json', 'backend': 'greenlight',
                             'sensor_noise': {'config': 'configs/sensor_noise_v0.json', 'setting': 'main'}},
                  'jobs': jobs, 'campaigns_not_completed': missing}
    args.out.write_text(json.dumps(eval_batch, indent=1))
    args.out.chmod(0o600)
    print(json.dumps({'evaluation_jobs': len(jobs), 'campaigns_not_completed': missing}))


if __name__ == '__main__':
    main()
