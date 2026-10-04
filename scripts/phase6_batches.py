#!/usr/bin/env python3
"""Site picks and batches for the best-known reference search and the boundary sensitivity (decision 2026-10-03).

    phase6_batches.py --campaigns RUN/campaigns.json --seed-opening E3_OPENING.json --out DIR \
        --dev-cache C --formal-cache F

Sites are drawn from the 48 test sites with the committed E3 seed (opening verified), each pick in
its own stream: "reference-search" draws 8 sites (the first 2 are searched twice, seeds 0 and 1);
"boundary-sensitivity" draws 16 sites. Writes reference_search.sh (one search per line, run in
parallel; each search uses 12 workers) and sensitivity.json (a batch for run_evaluation_batch.py:
the repeat-0 Full recommendation of each language model, the main baseline's seed-0 Full
recommendation and the fixed reference, each at Ueff 0 and 4 W/m2K on the site's three
evaluation years).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.commitment import verify  # noqa: E402
from slowlab.private_runs import weather_spec  # noqa: E402

UEFF = (0.0, 4.0)


def pick(seed, sites, label, n):
    stream = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], 'big')
    return np.random.default_rng([seed, stream]).choice(sites, n, replace=False).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaigns', type=Path, required=True)
    ap.add_argument('--seed-opening', type=Path, required=True)
    ap.add_argument('--commitment', type=Path, default=ROOT / 'configs/seed_commitment_e3_sample_v1.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--dev-cache', required=True)
    ap.add_argument('--formal-cache', required=True)
    args = ap.parse_args()
    opening = json.loads(args.seed_opening.read_text())
    if not verify(json.loads(args.commitment.read_text()), opening):
        raise SystemExit('seed opening does not match its commitment')
    seed = int(opening['secret']['seed'])
    batch = json.loads(args.campaigns.read_text())
    by_site = {}
    for j in batch['jobs']:
        by_site.setdefault(j['site']['site_index'], j)
    sites = sorted(by_site)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref_sites = pick(seed, sites, 'reference-search', 8)
    lines = []
    for k, s in enumerate(ref_sites):
        job = by_site[s]
        years = job['evaluation_years']
        weather = {str(y): weather_spec(y, args.dev_cache, args.formal_cache) for y in years}
        for search_seed in ((0, 1) if k < 2 else (0,)):
            tag = f'site{s:02d}_seed{search_seed}'
            cmd = [sys.executable, '-B', str(ROOT / 'scripts/search_reference_policy.py'),
                   '--site', json.dumps(job['site']), '--years', ','.join(map(str, years)),
                   '--weather', json.dumps(weather), '--seed', str(search_seed), '--workers', '12',
                   '--max-hours', '2', '--run-dir', str(out / 'reference' / tag),
                   '--out', str(out / 'reference' / f'{tag}.json')]
            lines.append(' '.join(shlex.quote(c) for c in cmd) + f' > {shlex.quote(str(out / "reference" / (tag + ".log")))} 2>&1')
    (out / 'reference').mkdir(exist_ok=True)
    script = out / 'reference_search.sh'
    script.write_text('#!/bin/bash\ncd ' + shlex.quote(str(ROOT)) + '\n' + ' &\n'.join(lines) + ' &\nwait\n')
    script.chmod(0o700)
    sens_sites = pick(seed, sites, 'boundary-sensitivity', 16)
    llms = sorted({j['model_name'] for j in batch['jobs'] if j['method'] == 'llm'})
    reference = json.loads((ROOT / 'configs/fixed_reference_v1.json').read_text())['policy']
    jobs = []
    for s in sens_sites:
        job = by_site[s]
        targets = [(m, {'settlement': str(Path(batch['out_dir']).parent / 'private' / f'site{s:02d}_{m}_r0' / 'full' / 'settlement.json')})
                   for m in llms]
        targets.append(('pbo', {'settlement': str(Path(batch['out_dir']).parent / 'private' / f'site{s:02d}_pbo_s0_full' / 'settlement.json')}))
        targets.append(('fixed_reference', {'policy': reference}))
        for name, target in targets:
            for ueff in UEFF:
                for y in job['evaluation_years']:
                    jobs.append({'id': f'site{s:02d}_{name}_ueff{ueff:g}_y{y}', 'site': job['site'], 'year': y,
                                 'site_overrides': {'boundary_ueff_w_m2_k': ueff},
                                 'weather': weather_spec(y, args.dev_cache, args.formal_cache), **target})
    path = out / 'sensitivity.json'
    path.write_text(json.dumps({'out_dir': str(out / 'sensitivity'), 'script': 'scripts/evaluate_sensitivity.py',
                                'purpose': 'boundary heat-transfer sensitivity (private)',
                                'common': {'contract': 'configs/task_contract_v8.json', 'backend': 'greenlight',
                                           'sensor_noise': {'config': 'configs/sensor_noise_v0.json', 'setting': 'main'}},
                                'jobs': jobs}, indent=1))
    path.chmod(0o600)
    (out / 'picks.json').write_text(json.dumps({'reference_search_sites': ref_sites, 'searched_twice': ref_sites[:2],
                                                'boundary_sensitivity_sites': sens_sites}, indent=1))
    print(json.dumps({'reference_searches': len(lines), 'sensitivity_jobs': len(jobs)}))


if __name__ == '__main__':
    main()
