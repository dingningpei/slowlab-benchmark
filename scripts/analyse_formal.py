#!/usr/bin/env python3
"""Formal E1/E2 analysis (RESEARCH_PLAN section 5; written and locked before any formal score was read).

    analyse_formal.py --campaigns RUN/campaigns.json --evaluations RUN/evaluations --report OUT.json

Scores: a recommendation's score is its mean over the site's three evaluation years; repeats are
averaged within a site. Primary family (Holm, 7 tests): E1 each language model's Full minus the
main baseline's Full; E2 Full minus Endpoint for the three models and the main baseline. Each test:
site-level paired mean difference, two-sided sign-flip test, BCa 95% interval, minimum detectable
difference (power 0.8 at the Holm worst-case level). Secondary family (BH): each model's Full minus
its initial recommendation, each method's Full minus the fixed reference, and the pairwise
differences of feedback gains between methods (E3 is added when its analysis runs).
Missing data: a site x method x condition is missing if any repeat lacks a completed evaluation;
the primary analysis uses complete pairs, the sensitivity analysis the mean of available repeats.
Also reported: method means, every difference stratified by evaluation year, and resources of the
recommended policies (heat, light, CO2 per crop) from the evaluation runs.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.statistics import benjamini_hochberg, holm, summarise  # noqa: E402

NAME = re.compile(r'site(\d+)_(.+?)_y(\d{4})\.json$')
TARGET = re.compile(r'(?:(?P<llm>.+)_r(?P<r>\d)_(?P<b>full|endpoint|initial))|(?:pbo_s(?P<s>\d)_(?P<bb>full|endpoint))'
                    r'|(?P<ref>fixed_reference)')
ALPHA = 0.05
PRIMARY = 7


def method_of(target):
    m = TARGET.fullmatch(target)
    if m is None:
        return None, None
    if m['ref']:
        return 'fixed_reference', 0
    if m['bb']:
        return f"pbo_{m['bb']}", int(m['s'])
    return f"{m['llm']}_{m['b']}", int(m['r'])


def expected_repeats(batch):
    """method -> site -> set of repeats the design calls for."""
    exp = defaultdict(lambda: defaultdict(set))
    for job in batch['jobs']:
        s = job['site']['site_index']
        if job['method'] == 'llm':
            r = int(job['id'].rsplit('_r', 1)[1])
            exp[f"{job['model_name']}_full"][s].add(r)
            exp[f"{job['model_name']}_endpoint"][s].add(r)
            if r == 0:
                exp[f"{job['model_name']}_initial"][s].add(0)
        else:
            seed = int(re.search(r'_s(\d)_', job['id'])[1])
            exp[f"pbo_{job['feedback']}"][s].add(seed)
        exp['fixed_reference'][s].add(0)
    return exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaigns', type=Path, required=True)
    ap.add_argument('--evaluations', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20261004)
    args = ap.parse_args()
    batch = json.loads(args.campaigns.read_text())
    sites = sorted({j['site']['site_index'] for j in batch['jobs']})
    exp = expected_repeats(batch)
    years = defaultdict(dict)
    resources = defaultdict(list)
    for f in sorted(args.evaluations.glob('site*_y*.json')):
        m = NAME.search(f.name)
        method, rep = method_of(m[2])
        if method is None:
            continue
        r = json.loads(f.read_text())
        if r.get('status') != 'completed':
            continue
        years[(int(m[1]), method, rep)][int(m[3])] = r['score_eur_m2']
        for p in r['plantings']:
            for c in p['crops']:
                resources[method].append([c['accrued'][k] for k in ('heat_kwh_m2', 'light_kwh_m2', 'co2_kg_m2')])
    score = {k: float(np.mean(list(v.values()))) for k, v in years.items() if len(v) == 3}

    def site_value(method, s, complete):
        reps = exp[method][s]
        got = [score[(s, method, r)] for r in sorted(reps) if (s, method, r) in score]
        if not got or (complete and len(got) < len(reps)):
            return None
        return float(np.mean(got))

    def site_year(method, s, y):
        vals = [years[(s, method, r)][y] for r in sorted(exp[method][s]) if y in years.get((s, method, r), {})]
        return float(np.mean(vals)) if vals else None

    def paired(a, b, complete=True):
        out = {}
        for s in sites:
            va, vb = site_value(a, s, complete), site_value(b, s, complete)
            if va is not None and vb is not None:
                out[s] = va - vb
        return out

    def by_year(a, b):
        acc = defaultdict(list)
        for s in sites:
            for y in sorted({y for (ss, m, _), v in years.items() if ss == s and m == a for y in v}):
                va, vb = site_year(a, s, y), site_year(b, s, y)
                if va is not None and vb is not None:
                    acc[y].append(va - vb)
        return {str(y): {'sites': len(v), 'mean': float(np.mean(v))} for y, v in sorted(acc.items())}

    llms = sorted({j['model_name'] for j in batch['jobs'] if j['method'] == 'llm'})
    contrasts = {}
    for llm in llms:
        contrasts[f'E1_{llm}_full_minus_baseline_full'] = (f'{llm}_full', 'pbo_full')
    for meth in llms + ['pbo']:
        contrasts[f'E2_{meth}_full_minus_endpoint'] = (f'{meth}_full', f'{meth}_endpoint')
    assert len(contrasts) == PRIMARY
    primary, sensitivity = {}, {}
    for i, (name, (a, b)) in enumerate(contrasts.items()):
        d = paired(a, b)
        primary[name] = {**summarise(list(d.values()), seed=args.seed + i, alpha_for_mde=ALPHA / PRIMARY),
                         'by_evaluation_year': by_year(a, b)}
        ds = paired(a, b, complete=False)
        sensitivity[name] = {'sites': len(ds), 'mean': float(np.mean(list(ds.values()))) if ds else float('nan')}
    for name, p in holm({k: v['p_sign_flip'] for k, v in primary.items()}).items():
        primary[name]['p_holm'] = p
    secondary = {}
    k = 100
    for llm in llms:
        d = paired(f'{llm}_full', f'{llm}_initial')
        secondary[f'{llm}_full_minus_initial'] = summarise(list(d.values()), seed=args.seed + k, alpha_for_mde=ALPHA); k += 1
    for meth in llms + ['pbo']:
        d = paired(f'{meth}_full', 'fixed_reference')
        secondary[f'{meth}_full_minus_fixed_reference'] = summarise(list(d.values()), seed=args.seed + k, alpha_for_mde=ALPHA); k += 1
    gains = {m: paired(f'{m}_full', f'{m}_endpoint') for m in llms + ['pbo']}
    for a, b in combinations(llms + ['pbo'], 2):
        common = sorted(set(gains[a]) & set(gains[b]))
        d = [gains[a][s] - gains[b][s] for s in common]
        secondary[f'feedback_gain_{a}_minus_{b}'] = summarise(d, seed=args.seed + k, alpha_for_mde=ALPHA); k += 1
    for name, p in benjamini_hochberg({n: v['p_sign_flip'] for n, v in secondary.items()}).items():
        secondary[name]['p_bh'] = p
    methods = sorted({m for (_, m, _) in score})
    means = {m: {'sites': sum(site_value(m, s, True) is not None for s in sites),
                 'mean': float(np.mean([v for s in sites if (v := site_value(m, s, True)) is not None]))}
             for m in methods}
    missing = {m: [s for s in sites if site_value(m, s, True) is None] for m in exp}
    res = {m: dict(zip(('heat_kwh_m2', 'light_kwh_m2', 'co2_kg_m2'), np.mean(v, axis=0).round(3).tolist()))
           for m, v in sorted(resources.items())}
    out = {'format': 'slowlab-formal-analysis-v1', 'sites': len(sites), 'alpha': ALPHA,
           'primary': primary, 'secondary': secondary, 'missing_data_sensitivity': sensitivity,
           'method_means': means, 'missing_site_method': {m: v for m, v in missing.items() if v},
           'resources_per_crop_of_recommended_policies': res,
           'notes': ['sign-flip p-values: 100,000 Monte Carlo flips; BCa: 10,000 site resamples',
                     'minimum detectable: paired t approximation at power 0.8 (primary: alpha 0.05/7)',
                     'wide intervals that are not significant are not read as equivalence']}
    args.report.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: {'mean': round(v['mean'], 3), 'ci': [round(x, 3) for x in v['ci95_bca']],
                          'p_holm': round(v['p_holm'], 4), 'sites': v['sites']} for k, v in primary.items()}, indent=1))


if __name__ == '__main__':
    main()
