#!/usr/bin/env python3
"""Phase 4 pilot campaign batch (private: it contains the pilot master seed; keep it off Git).

Pilot sites are indices 0..n-1 of the committed pilot master seed under
distribution v1, each with its formal campaign year (formal_weather_years).
Jobs: every model x repeat x site as a branched LLM campaign (Full and Endpoint
from one day-0 design), and staggered BO with each seed under both conditions.
Models come from --models (JSON list of {"name", "llm": {...}}); "scripted"
needs no API and is used for dry runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.site_distribution import formal_weather_years  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--secret', type=Path, required=True, help='private pilot master-seed file')
    ap.add_argument('--models', required=True, help='JSON list of {"name", "llm"}, or a path to a models config')
    ap.add_argument('--sites', type=int, default=None, help='default: the committed pilot size')
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--bo-seeds', type=int, default=2)
    ap.add_argument('--dev-cache', required=True)
    ap.add_argument('--formal-cache', required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    secret = json.loads(args.secret.read_text())
    models = args.models
    args.models = (json.loads((ROOT / models).read_text())['models'] if not models.lstrip().startswith('[')
                   else json.loads(models))
    if secret['partition'] != 'pilot' or secret['distribution_id'] != 'slowlab-site-distribution-v1':
        raise SystemExit('not the pilot master seed for distribution v1')
    distribution = json.loads((ROOT / 'configs/site_distribution_v1.json').read_text())
    n = args.sites or secret['n_sites']
    jobs = []
    sites = []
    for index in range(n):
        site = {'distribution': 'configs/site_distribution_v1.json', 'master_seed': secret['master_seed'], 'site_index': index}
        sites.append((index, site, formal_weather_years(distribution, secret['master_seed'], index)))
    # Order: repeat 0 of every site and model first, so a spend cap leaves complete first-round data.
    for r in range(args.repeats):
        for index, site, years in sites:
            for model in args.models:
                job = f"site{index:02d}_{model['name']}_r{r}"
                jobs.append({'id': job, 'method': 'llm', 'model_name': model['name'],
                             'llm': {**model['llm'], 'generation_seed': 9000 + 10 * index + r},
                             'tool_seed': 7000 + 10 * index + r, 'site': site, 'year': years['campaign_year'],
                             'feedback': None, 'evaluation_years': years['evaluation_years'],
                             'private_dir': str(args.run_dir / 'private' / job)})
        if r == 0:
            for index, site, years in sites:
                for seed in range(args.bo_seeds):
                    for feedback in ('full', 'endpoint'):
                        job = f'site{index:02d}_bo_s{seed}_{feedback}'
                        jobs.append({'id': job, 'method': 'bo', 'bo': {'schedule': 'staggered', 'seed': 2000 + seed},
                                     'site': site, 'year': years['campaign_year'], 'feedback': feedback,
                                     'evaluation_years': years['evaluation_years'],
                                     'private_dir': str(args.run_dir / 'private' / job)})
    args.run_dir.mkdir(parents=True, exist_ok=True)
    batch = {'out_dir': str(args.run_dir / 'campaigns'), 'script': 'scripts/run_campaign_job.py',
             'purpose': 'Phase 4 pilot campaigns (private)',
             'common': {'contract': 'configs/task_contract_v8.json', 'backend': 'greenlight',
                        'dev_cache': args.dev_cache, 'formal_cache': args.formal_cache},
             'jobs': jobs}
    path = args.run_dir / 'campaigns.json'
    path.write_text(json.dumps(batch, indent=1))
    path.chmod(0o600)
    print(json.dumps({'jobs': len(jobs), 'llm': sum(j['method'] == 'llm' for j in jobs),
                      'bo': sum(j['method'] == 'bo' for j in jobs)}))


if __name__ == '__main__':
    main()
