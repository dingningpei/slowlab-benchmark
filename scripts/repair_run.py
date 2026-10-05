#!/usr/bin/env python3
"""Batches for the field-order repair of the formal run (RESEARCH_PLAN decisions 2026-10-05).

    repair_run.py baseline     --formal-campaigns F/campaigns.json --config configs/prior_bo_v3.json --run-dir R --dev-cache C --formal-cache FC
    repair_run.py llm          --formal-campaigns F/campaigns.json --specs-dir F/campaigns/specs --audits-dir A --run-dir R
                               --dev-cache C --formal-cache FC [--env-file KEYS]
    repair_run.py evaluations  --formal-campaigns F/campaigns.json --run-dir R --dev-cache C --formal-cache FC
    repair_run.py sensitivity  --formal-campaigns F/campaigns.json --picks F/phase6/picks.json --run-dir R --dev-cache C --formal-cache FC

baseline: all main-baseline jobs of the formal batch, same ids, sites, years and seeds, with the fixed
agent and the radius re-selected on development sites (config). llm: one resume command per LLM job whose
Full branch received a process-predictor output (found from the audits), for scripts/resume_llm_branch.py.
evaluations: the changed recommendations (every baseline job; every repaired Full branch) on each site's
three evaluation years, named as in the formal run. sensitivity: the changed recommendations on the
boundary-sensitivity sites (baseline seed-0 Full; repaired repeat-0 Full) at Ueff 0 and 4.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.private_runs import weather_spec  # noqa: E402

PREDICTION_KEY = 'predicted_contribution_margin_eur_m2'
UEFF = (0.0, 4.0)


def repaired_jobs(batch, audits_dir: Path):
    out = []
    for job in batch['jobs']:
        if job['method'] != 'llm':
            continue
        audit = audits_dir / f"{job['id']}.audit.jsonl"
        for line in audit.read_text().splitlines():
            rec = json.loads(line)
            if rec['label'] == 'full' and PREDICTION_KEY in rec['messages'][-1]['content']:
                out.append(job)
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('baseline', 'llm', 'evaluations', 'sensitivity'))
    ap.add_argument('--formal-campaigns', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--config', default='configs/prior_bo_v3.json')
    ap.add_argument('--specs-dir', type=Path)
    ap.add_argument('--audits-dir', type=Path)
    ap.add_argument('--picks', type=Path)
    ap.add_argument('--env-file')
    ap.add_argument('--dev-cache', required=True)
    ap.add_argument('--formal-cache', required=True)
    args = ap.parse_args()
    batch = json.loads(args.formal_campaigns.read_text())
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if args.stage == 'baseline':
        radius = json.loads((ROOT / args.config).read_text())['selected_radius']
        jobs = []
        for job in batch['jobs']:
            if job['method'] != 'bo':
                continue
            jobs.append({**job, 'private_dir': str(run / 'baseline' / 'private' / job['id']),
                         'bo': {**job['bo'], 'config': args.config, 'radius': radius}})
        path = run / 'baseline_campaigns.json'
        path.write_text(json.dumps({'out_dir': str(run / 'baseline' / 'campaigns'), 'script': 'scripts/run_campaign_job.py',
                                    'purpose': 'repair: main baseline with the fixed agent (private)',
                                    'common': {**batch['common'], 'dev_cache': args.dev_cache, 'formal_cache': args.formal_cache},
                                    'jobs': jobs}, indent=1))
        path.chmod(0o600)
        print(json.dumps({'baseline_jobs': len(jobs), 'radius': radius}))
        return
    if args.stage == 'llm':
        jobs = repaired_jobs(batch, args.audits_dir)
        lines = []
        for job in jobs:
            out = run / 'llm' / job['id']
            cmd = [sys.executable, '-B', str(ROOT / 'scripts/resume_llm_branch.py'), '--job-spec', str(args.specs_dir / f"{job['id']}.json"),
                   '--audit', str(args.audits_dir / f"{job['id']}.audit.jsonl"), '--private-dir', str(out / 'private'),
                   '--out', str(out / 'record.json'), '--live-from', 'auto', '--dev-cache', args.dev_cache,
                   '--formal-cache', args.formal_cache] + (['--env-file', args.env_file] if args.env_file else [])
            lines.append(' '.join(shlex.quote(c) for c in cmd) + f" > {shlex.quote(str(out))}.log 2>&1")
        (run / 'llm').mkdir(exist_ok=True)
        (run / 'llm_jobs.json').write_text(json.dumps([j['id'] for j in jobs], indent=1))
        (run / 'llm_resume.txt').write_text('\n'.join(lines) + '\n')
        print(json.dumps({'llm_branches_to_repair': len(jobs)}))
        return
    llm_jobs = json.loads((run / 'llm_jobs.json').read_text())
    by_id = {j['id']: j for j in batch['jobs']}
    common = {'contract': 'configs/task_contract_v8.json', 'backend': 'greenlight',
              'sensor_noise': {'config': 'configs/sensor_noise_v0.json', 'setting': 'main'}}
    if args.stage == 'evaluations':
        jobs = []
        for job in batch['jobs']:
            if job['method'] == 'bo':
                targets = [(job['id'], run / 'baseline' / 'private' / job['id'] / 'settlement.json')]
            elif job['id'] in llm_jobs:
                targets = [(f"{job['id']}_full", run / 'llm' / job['id'] / 'private' / 'full' / 'settlement.json')]
            else:
                continue
            for name, settlement in targets:
                for y in job['evaluation_years']:
                    jobs.append({'id': f'{name}_y{y}', 'site': job['site'], 'year': y, 'settlement': str(settlement),
                                 'weather': weather_spec(y, args.dev_cache, args.formal_cache)})
        path = run / 'evaluations.json'
        path.write_text(json.dumps({'out_dir': str(run / 'evaluations'), 'purpose': 'repair: evaluations of changed recommendations (private)',
                                    'common': common, 'jobs': jobs}, indent=1))
        path.chmod(0o600)
        print(json.dumps({'evaluation_jobs': len(jobs)}))
        return
    picks = json.loads(args.picks.read_text())['boundary_sensitivity_sites']
    jobs = []
    for s in picks:
        targets = [('pbo', run / 'baseline' / 'private' / f'site{s:02d}_pbo_s0_full' / 'settlement.json',
                    by_id[f'site{s:02d}_pbo_s0_full'])]
        for jid in llm_jobs:
            if jid.startswith(f'site{s:02d}_') and jid.endswith('_r0'):
                model = by_id[jid]['model_name']
                targets.append((model, run / 'llm' / jid / 'private' / 'full' / 'settlement.json', by_id[jid]))
        for name, settlement, job in targets:
            for ueff in UEFF:
                for y in job['evaluation_years']:
                    jobs.append({'id': f'site{s:02d}_{name}_ueff{ueff:g}_y{y}', 'site': job['site'], 'year': y,
                                 'site_overrides': {'boundary_ueff_w_m2_k': ueff}, 'settlement': str(settlement),
                                 'weather': weather_spec(y, args.dev_cache, args.formal_cache)})
    path = run / 'sensitivity.json'
    path.write_text(json.dumps({'out_dir': str(run / 'sensitivity'), 'script': 'scripts/evaluate_sensitivity.py',
                                'purpose': 'repair: boundary sensitivity of changed recommendations (private)',
                                'common': common, 'jobs': jobs}, indent=1))
    path.chmod(0o600)
    print(json.dumps({'sensitivity_jobs': len(jobs)}))


if __name__ == '__main__':
    main()
