#!/usr/bin/env python3
"""Formal E1/E2 campaign batch (private: it contains the test master seed; keep it off Git).

Test sites are indices 0..N-1 of the committed test master seed under distribution v1
(N = 48, RESEARCH_PLAN decision 2026-10-04), each with its formal campaign year. Jobs:
every model x repeat x site as a branched LLM campaign (Full and Endpoint from one day-0
design), and the main baseline (prior-informed local BO, configs/prior_bo_v1.json) with
two seeds under both conditions. Order: repeat 0 of every site and model first (with the
baseline), then repeat 1, so a spend pause leaves complete first-round data.
``--secret`` may also be the pilot seed with ``--dry-run`` (scripted model, no API) to
check the path before the formal run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.site_distribution import formal_weather_years  # noqa: E402

MODELS = 'configs/formal_models_v1.json'
BASELINE = 'configs/prior_bo_v1.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--secret', type=Path, required=True)
    ap.add_argument('--sites', type=int, default=48)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--baseline-seeds', type=int, default=2)
    ap.add_argument('--dev-cache', required=True)
    ap.add_argument('--formal-cache', required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--dry-run', action='store_true', help='pilot seed, scripted model, no API')
    args = ap.parse_args()
    secret = json.loads(args.secret.read_text())
    expected = 'pilot' if args.dry_run else 'test'
    if secret['partition'] != expected or secret['distribution_id'] != 'slowlab-site-distribution-v1':
        raise SystemExit(f'not the {expected} master seed for distribution v1')
    models = json.loads((ROOT / MODELS).read_text())['models']
    if args.dry_run:
        models = [{'name': 'scripted', 'llm': {'model': 'scripted'}}]
    radius = json.loads((ROOT / BASELINE).read_text())['selected_radius']
    distribution = json.loads((ROOT / 'configs/site_distribution_v1.json').read_text())
    run_dir = args.run_dir.resolve()
    sites = []
    for index in range(args.sites):
        site = {'distribution': 'configs/site_distribution_v1.json', 'master_seed': secret['master_seed'], 'site_index': index}
        sites.append((index, site, formal_weather_years(distribution, secret['master_seed'], index)))
    jobs = []
    for r in range(args.repeats):
        for index, site, years in sites:
            for model in models:
                job = f"site{index:02d}_{model['name']}_r{r}"
                jobs.append({'id': job, 'method': 'llm', 'model_name': model['name'],
                             'llm': {**model['llm'], 'generation_seed': 19000 + 10 * index + r},
                             'tool_seed': 17000 + 10 * index + r, 'site': site, 'year': years['campaign_year'],
                             'feedback': None, 'evaluation_years': years['evaluation_years'],
                             'private_dir': str(run_dir / 'private' / job)})
        if r == 0:
            for index, site, years in sites:
                for seed in range(args.baseline_seeds):
                    for feedback in ('full', 'endpoint'):
                        job = f'site{index:02d}_pbo_s{seed}_{feedback}'
                        jobs.append({'id': job, 'method': 'bo', 'site': site, 'year': years['campaign_year'],
                                     'feedback': feedback, 'evaluation_years': years['evaluation_years'],
                                     'private_dir': str(run_dir / 'private' / job),
                                     'bo': {'agent': 'prior_local', 'config': BASELINE, 'radius': radius,
                                            'seed': 6000 + seed}})
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / 'campaigns.json'
    path.write_text(json.dumps({'out_dir': str(run_dir / 'campaigns'), 'script': 'scripts/run_campaign_job.py',
                                'purpose': ('formal E1/E2 campaigns (private)' if not args.dry_run
                                            else 'dry run of the formal batch on pilot sites (scripted model)'),
                                'common': {'contract': 'configs/task_contract_v8.json', 'backend': 'greenlight',
                                           'dev_cache': args.dev_cache, 'formal_cache': args.formal_cache},
                                'jobs': jobs}, indent=1))
    path.chmod(0o600)
    print(json.dumps({'jobs': len(jobs), 'llm': sum(j['method'] == 'llm' for j in jobs),
                      'baseline': sum(j['method'] == 'bo' for j in jobs)}))


if __name__ == '__main__':
    main()
