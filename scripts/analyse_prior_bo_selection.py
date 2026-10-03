#!/usr/bin/env python3
"""Radius selection for the prior-informed local BO (rule frozen in configs/prior_bo_v0.json).

A recommendation's score is its mean over the site's three development evaluation years.
Per candidate radius: mean score over development sites x method seeds x Full/Endpoint,
the between-seed SD within a site and condition, Full - Endpoint, the share of recommendations
equal to the anchor, and the difference from the fixed reference on the same sites and years.
The gp-bo-v3 scores of the development variance study (same sites and years) are shown for
comparison. Selection: highest mean; a tie (means within 0.01 EUR/m2) goes to the lower SD.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

JOB = re.compile(r'site(\d+)_r(\d+)_seed(\d+)_(full|endpoint)$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--config', type=Path, default=Path('configs/prior_bo_v0.json'))
    ap.add_argument('--fixed-reference', type=Path, default=Path('results/fixed_reference_selection_20261002.json'))
    ap.add_argument('--dev-variance', type=Path, default=Path('results/dev_variance_study_20261002.json'))
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    anchor = json.loads(Path(config['anchor']['file']).read_text())['policy']
    evals = json.loads((args.run_dir / 'evaluations.json').read_text())
    year_scores = defaultdict(dict)
    for job in evals['jobs']:
        r = json.loads((Path(evals['out_dir']) / f"{job['id']}.json").read_text())
        if r.get('status') == 'completed':
            year_scores[job['id'].rsplit('_y', 1)[0]][job['year']] = r['score_eur_m2']
    campaigns = json.loads((args.run_dir / 'campaigns.json').read_text())
    rec = {}
    for job in campaigns['jobs']:
        m = JOB.match(job['id'])
        site, radius, seed, cond = int(m[1]), int(m[2]) / 100, int(m[3]), m[4]
        record = json.loads((Path(campaigns['out_dir']) / f"{job['id']}.json").read_text())
        policy = record['bo']['recommendation']
        keys = [k for k, users in evals['uses'].items() if job['id'] in users]
        scores = {int(k.rsplit('_y', 1)[1]): year_scores[k.rsplit('_y', 1)[0]].get(int(k.rsplit('_y', 1)[1])) for k in keys}
        if len(scores) == 3 and None not in scores.values():
            rec[(radius, site, seed, cond)] = {'score': float(np.mean(list(scores.values()))),
                                               'is_anchor': all(abs(policy[k] - anchor[k]) < 1e-9 for k in anchor),
                                               'completed_crops': record['bo']['completed_crops']}
    fixed = {int(s): v for s, v in json.loads(args.fixed_reference.read_text())['candidates']['grower_standard']['site_means'].items()}
    dev = json.loads(args.dev_variance.read_text())['recommendation_scores']
    v3 = {tuple(int(x) if x.isdigit() else x for x in k.split('_')): v for k, v in dev.items()}
    sites = sorted({k[1] for k in rec})
    out = {'format': 'prior-bo-radius-selection-v1', 'rule': config['radius_selection_rule'],
           'acceptance': config['acceptance'], 'fixed_reference_mean_eur_m2': float(np.mean([fixed[s] for s in sites])),
           'radii': {}}
    for radius in config['candidate_radii']:
        rows = {k: v for k, v in rec.items() if k[0] == radius}
        scores = [v['score'] for v in rows.values()]
        within = [np.std([rows[(radius, s, seed, c)]['score'] for seed in (0, 1) if (radius, s, seed, c) in rows], ddof=1)
                  for s in sites for c in ('full', 'endpoint')
                  if all((radius, s, seed, c) in rows for seed in (0, 1))]
        e2 = [rows[(radius, s, seed, 'full')]['score'] - rows[(radius, s, seed, 'endpoint')]['score']
              for s in sites for seed in (0, 1) if (radius, s, seed, 'full') in rows and (radius, s, seed, 'endpoint') in rows]
        site_means = {s: float(np.mean([v['score'] for k, v in rows.items() if k[1] == s])) for s in sites}
        out['radii'][str(radius)] = {
            'recommendations': len(rows), 'mean_eur_m2': float(np.mean(scores)),
            'mean_full': float(np.mean([v['score'] for k, v in rows.items() if k[3] == 'full'])),
            'mean_endpoint': float(np.mean([v['score'] for k, v in rows.items() if k[3] == 'endpoint'])),
            'within_site_seed_sd': float(np.sqrt(np.mean(np.square(within)))),
            'full_minus_endpoint_mean': float(np.mean(e2)), 'full_minus_endpoint_sd': float(np.std(e2, ddof=1)),
            'minus_fixed_reference_mean': float(np.mean([site_means[s] - fixed[s] for s in sites])),
            'share_recommending_anchor': float(np.mean([v['is_anchor'] for v in rows.values()])),
            'site_means': {s: round(v, 3) for s, v in site_means.items()}}
    v3_scores = [v for k, v in v3.items() if k[0] in sites]
    v3_within = [np.std([v3[(s, seed, c)] for seed in (0, 1)], ddof=1) for s in sites for c in ('full', 'endpoint')
                 if all((s, seed, c) in v3 for seed in (0, 1))]
    out['gp_bo_v3_same_sites'] = {'mean_eur_m2': float(np.mean(v3_scores)),
                                  'within_site_seed_sd': float(np.sqrt(np.mean(np.square(v3_within))))}
    ranked = sorted(out['radii'].items(), key=lambda kv: -kv[1]['mean_eur_m2'])
    best = ranked[0]
    ties = [kv for kv in ranked if best[1]['mean_eur_m2'] - kv[1]['mean_eur_m2'] < 0.01]
    chosen = min(ties, key=lambda kv: kv[1]['within_site_seed_sd'])
    out['selected_radius'] = float(chosen[0])
    out['accepted'] = chosen[1]['mean_eur_m2'] >= out['fixed_reference_mean_eur_m2']
    args.report.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'radii'} | {
        'radii': {r: {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items() if k != 'site_means'}
                  for r, d in out['radii'].items()}}, indent=1))


if __name__ == '__main__':
    main()
