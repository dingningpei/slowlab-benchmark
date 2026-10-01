#!/usr/bin/env python3
"""Best-known feasible reference policy for one site by numerical search (private, after the main experiments).

Search phase: candidates are scored two per run (one January and one July crop
each) on the site's first evaluation year, so every candidate faces the same
weather. Rounds: a space-filling start, then batches by expected improvement
with kriging-believer fantasies (the toolbox GP). Final phase: the top
candidates by search score get the full evaluation (all evaluation years, four
compartments). The best final mean is the site's best-known reference; the
search error is estimated by repeating the search with another seed on some
sites (decision 2026-10-01).
"""
from __future__ import annotations

import argparse
import json
import math
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.bo_agent import expected_improvement  # noqa: E402
from slowlab.tools import OutcomeModel  # noqa: E402


def feasible_pool(fields, n, rng):
    from scipy.stats import qmc
    sampler = qmc.Sobol(len(fields), scramble=True, seed=rng)
    out = []
    for point in sampler.random_base2(math.ceil(math.log2(4 * n))):
        p = {k: round(s['min'] + float(u) * (s['max'] - s['min']), 3) for (k, s), u in zip(fields.items(), point)}
        if p['night_temperature_c'] <= p['day_temperature_c']:
            out.append(p)
        if len(out) == n:
            break
    return out


def scale(fields, p):
    return [(p[k] - s['min']) / (s['max'] - s['min']) for k, s in fields.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site', type=json.loads, required=True, help='{"distribution", "master_seed", "site_index"}')
    ap.add_argument('--years', required=True, help='evaluation years, comma-separated; the first is the search year')
    ap.add_argument('--weather', type=json.loads, required=True, help='{"year": weather spec} for every year')
    ap.add_argument('--contract', default='configs/task_contract_v8.json')
    ap.add_argument('--backend', default='greenlight')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--initial', type=int, default=12)
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--batch', type=int, default=12)
    ap.add_argument('--final-top', type=int, default=3)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--max-hours', type=float, default=1.5)
    ap.add_argument('--max-memory-gb', type=float, default=6.0)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    years = [int(y) for y in args.years.split(',')]
    contract = json.loads((ROOT / args.contract).read_text())
    fields = contract['policy']['fields']
    rng = np.random.default_rng([args.seed, 0x5EA4C])
    common = {'contract': args.contract, 'backend': args.backend, 'site': args.site,
              **({'sensor_noise': {'config': 'configs/sensor_noise_v0.json', 'setting': 'main'}}
                 if args.backend == 'greenlight' else {})}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    began, counter = time.monotonic(), [0]

    def run_job(script, extra):
        counter[0] += 1
        name = f'job{counter[0]:04d}'
        spec = {**common, **extra, 'out': str(args.run_dir / f'{name}.json')}
        path = args.run_dir / f'{name}.spec.json'
        path.write_text(json.dumps(spec))
        limit = int(args.max_memory_gb * 2 ** 30)

        def cap():
            try:
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except (ValueError, OSError):
                pass
        try:
            subprocess.run([sys.executable, '-B', str(ROOT / script), str(path)], cwd=ROOT, capture_output=True,
                           timeout=args.max_hours * 3600, preexec_fn=cap)
        except subprocess.TimeoutExpired:
            return {'status': 'failed', 'error': 'wall-clock limit'}
        out = Path(spec['out'])
        return json.loads(out.read_text()) if out.exists() else {'status': 'failed', 'error': 'no output'}

    def score_pairs(policies):
        pairs = [policies[i:i + 2] for i in range(0, len(policies), 2)]
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(lambda pr: run_job('scripts/evaluate_policies_job.py', {
                'policies': pr, 'year': years[0], 'weather': args.weather[str(years[0])]}), pairs))
        scored = []
        for pr, res in zip(pairs, results):
            if res['status'] != 'completed':
                scored += [{'policy': p, 'search_score': None, 'error': res.get('error')} for p in pr]
            else:
                scored += [{'policy': r['policy'], 'search_score': r['score_eur_m2']} for r in res['results']]
        return scored

    candidates = score_pairs(feasible_pool(fields, args.initial, rng))
    log = [{'round': 0, 'n': len(candidates)}]
    pool = feasible_pool(fields, 512, rng)
    for rnd in range(1, args.rounds + 1):
        ok = [c for c in candidates if c['search_score'] is not None]
        x = np.array([scale(fields, c['policy']) for c in ok]); y = np.array([c['search_score'] for c in ok])
        model = OutcomeModel(x, y, rng, 4)
        tried = {json.dumps(c['policy'], sort_keys=True) for c in candidates}
        options = [p for p in pool if json.dumps(p, sort_keys=True) not in tried]
        picks, fant_x, fant_y = [], [], []
        best = float(y.max())
        for _ in range(min(args.batch, len(options))):
            m = model.with_fantasies(np.array(fant_x), np.array(fant_y)) if fant_x else model
            xs = np.array([scale(fields, p) for p in options])
            mean, sd, _ = m.predict(xs)
            ei = expected_improvement(mean, sd, best, 0.0)
            k = int(np.argmax(ei))
            picks.append(options.pop(k)); fant_x.append(xs[k]); fant_y.append(float(mean[k]))
        candidates += score_pairs(picks)
        log.append({'round': rnd, 'n': len(picks)})
    ranked = sorted([c for c in candidates if c['search_score'] is not None], key=lambda c: -c['search_score'])
    finalists = ranked[:args.final_top]
    jobs = [(i, y) for i in range(len(finalists)) for y in years]
    with ThreadPoolExecutor(args.workers) as tp:
        results = list(tp.map(lambda iy: run_job('scripts/evaluate_recommendation.py', {
            'policy': finalists[iy[0]]['policy'], 'year': iy[1], 'weather': args.weather[str(iy[1])]}), jobs))
    for f in finalists:
        f['final_by_year'] = {}
    for (i, y), res in zip(jobs, results):
        finalists[i]['final_by_year'][str(y)] = res.get('score_eur_m2') if res['status'] == 'completed' else None
    for f in finalists:
        vals = [v for v in f['final_by_year'].values() if v is not None]
        f['final_mean'] = sum(vals) / len(vals) if len(vals) == len(years) else None
    best = max((f for f in finalists if f['final_mean'] is not None), key=lambda f: f['final_mean'], default=None)
    out = {'purpose': 'best-known feasible reference policy (numerical search)', 'site': args.site, 'years': years,
           'seed': args.seed, 'budget': {k: getattr(args, k) for k in ('initial', 'rounds', 'batch', 'final_top')},
           'evaluation_runs': counter[0], 'elapsed_hours': round((time.monotonic() - began) / 3600, 3),
           'best': best, 'finalists': finalists, 'candidates': candidates, 'log': log}
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'best_final_mean': best and best['final_mean'], 'evaluation_runs': counter[0],
                      'elapsed_hours': out['elapsed_hours']}))


if __name__ == '__main__':
    main()
