#!/usr/bin/env python3
"""Scores, spreads and preliminary site counts from the Phase 4 pilot evaluations.

A recommendation's score is its mean over the site's three evaluation years.
Methods: each LLM's Full, Endpoint and repeat-0 initial recommendation, BO
Full and Endpoint, the fixed reference, and (when given) the prior-informed local BO
("pbo", the main baseline from 2026-10-03). For the seven primary tests (E1:
each LLM Full - main baseline Full; E2: Full - Endpoint for the three LLMs and
the main baseline; the main baseline is pbo when present, else bo) it
reports the mean site-level difference, the within-site (between-repeat) SD,
the between-site SD, and the site count for a two-sided paired t-test (power
0.8) at alpha 0.05 and at the Holm worst case 0.05/7, for 2 and 5 EUR/m2.
The t-test approximates the planned sign-flip test. Pilot sites, formal years.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyse_dev_variance import n_paired

NAME = re.compile(r'site(\d+)_(.+?)_y(\d{4})\.json$')
TARGET = re.compile(r'(?:(?P<llm>.+)_r(?P<r>\d)_(?P<b>full|endpoint|initial))|(?:(?P<bo>p?bo)_s(?P<s>\d)_(?P<bb>full|endpoint))'
                    r'|(?P<ref>fixed_reference)')


def method_of(target):
    m = TARGET.fullmatch(target)
    if m['ref']:
        return 'fixed_reference', 0
    if m['bb']:
        return f"{m['bo']}_{m['bb']}", int(m['s'])
    return f"{m['llm']}_{m['b']}", int(m['r'])


def components(diffs):
    """diffs: {site: [difference per repeat]} -> summary with SDs for R = 2 repeats."""
    within = [np.var(v, ddof=1) for v in diffs.values() if len(v) > 1]
    s2_within = float(np.mean(within)) if within else float('nan')
    means = [float(np.mean(v)) for v in diffs.values()]
    s2_means = float(np.var(means, ddof=1))
    r = float(np.mean([len(v) for v in diffs.values()]))
    s2_between = max(s2_means - s2_within / r, 0.0) if within else float('nan')
    out = {'sites': len(means), 'mean': float(np.mean(means)), 'site_means': [round(x, 3) for x in means],
           'site_mean_sd': math.sqrt(s2_means), 'within_site_sd': math.sqrt(s2_within),
           'between_site_sd': math.sqrt(s2_between), 'sites_needed': {}}
    for delta in (2.0, 5.0):
        for label, alpha in (('alpha_0.05', 0.05), ('holm_worst_0.05_over_7', 0.05 / 7)):
            out['sites_needed'][f'{delta:g}_eur_{label}'] = n_paired(delta, out['site_mean_sd'], alpha)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluations', type=Path, nargs='+', required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    years, reasons, statuses = defaultdict(dict), defaultdict(Counter), Counter()
    for f in sorted(f for d in args.evaluations for f in d.glob('site*_y*.json')):
        m = NAME.search(f.name)
        r = json.loads(f.read_text())
        statuses[r.get('status')] += 1
        if r.get('status') != 'completed':
            continue
        method, rep = method_of(m[2])
        years[(int(m[1]), method, rep)][int(m[3])] = r['score_eur_m2']
        for p in r['plantings']:
            for c in p['crops']:
                reasons[method][c['reason']] += 1
    score = {k: float(np.mean(list(v.values()))) for k, v in years.items() if len(v) == 3}
    sites = sorted({k[0] for k in score})
    methods = sorted({k[1] for k in score})
    by = defaultdict(dict)
    for (s, meth, rep), v in score.items():
        by[meth].setdefault(s, {})[rep] = v

    table = {}
    for meth in methods:
        site_means = [float(np.mean(list(by[meth][s].values()))) for s in sites if s in by[meth]]
        reps = [v for s in sites for v in by[meth].get(s, {}).values()]
        within = [np.std(list(by[meth][s].values()), ddof=1) for s in sites if len(by[meth].get(s, {})) > 1]
        table[meth] = {'n_recommendations': len(reps), 'mean': float(np.mean(reps)), 'site_mean_sd': float(np.std(site_means, ddof=1)),
                       'within_site_repeat_sd': float(np.sqrt(np.mean(np.square(within)))) if within else None,
                       'by_site': {s: round(float(np.mean(list(by[meth][s].values()))), 3) for s in sites if s in by[meth]},
                       'crop_end_reasons': dict(reasons[meth])}

    def paired(a, b, same_repeat):
        d = {}
        for s in sites:
            ra, rb = by[a].get(s, {}), by[b].get(s, {})
            if not ra or not rb:
                continue
            if same_repeat:
                d[s] = [ra[i] - rb[i] for i in sorted(ra) if i in rb]
            else:
                mb = float(np.mean(list(rb.values())))
                d[s] = [v - mb for v in ra.values()]
        return {s: v for s, v in d.items() if v}

    llms = sorted({m[:-len('_full')] for m in methods if m.endswith('_full') and not m.startswith(('bo', 'pbo'))})
    base = 'pbo' if 'pbo_full' in methods else 'bo'
    primary = {}
    for llm in llms:
        primary[f'E1_{llm}_full_minus_{base}_full'] = components(paired(f'{llm}_full', f'{base}_full', True))
    for meth in llms + [base]:
        primary[f'E2_{meth}_full_minus_endpoint'] = components(paired(f'{meth}_full', f'{meth}_endpoint', True))
    secondary = {}
    for llm in llms:
        secondary[f'{llm}_full_minus_initial'] = components(paired(f'{llm}_full', f'{llm}_initial', False))
    for meth in [f'{x}_full' for x in llms] + [f'{b}_full' for b in ('bo', 'pbo') if f'{b}_full' in methods]:
        secondary[f'{meth}_minus_fixed_reference'] = components(paired(meth, 'fixed_reference', False))
    by_year = defaultdict(lambda: defaultdict(list))
    for (s, meth, rep), v in years.items():
        for y, sc in v.items():
            by_year[meth][y].append(sc)
    out = {'format': 'pilot-evaluation-analysis-v1', 'sites': sites, 'evaluation_statuses': dict(statuses),
           'scope': 'Phase 4 pilot: 8 pilot sites x 3 formal evaluation years; 2 repeats per LLM and BO',
           'methods': table, 'primary_tests_preview': primary, 'secondary_preview': secondary,
           'method_mean_by_year_rank': {m: [round(float(np.mean(by_year[m][y])), 3) for y in sorted(by_year[m])] for m in methods},
           'note': ('Descriptive pilot only; no hypothesis is tested on pilot data. Site counts use a paired t-test '
                    'approximation with the pilot site-mean SD at 2 repeats; 8 sites give a rough SD estimate.')}
    args.report.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'statuses': dict(statuses),
                      'methods': {m: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in t.items()
                                      if k in ('mean', 'site_mean_sd', 'within_site_repeat_sd')} for m, t in table.items()},
                      'primary': {k: {'mean': round(v['mean'], 2), 'site_mean_sd': round(v['site_mean_sd'], 2),
                                      'within': round(v['within_site_sd'], 2), 'between': round(v['between_site_sd'], 2),
                                      'n': v['sites_needed']} for k, v in primary.items()},
                      'secondary': {k: {'mean': round(v['mean'], 2), 'site_mean_sd': round(v['site_mean_sd'], 2)}
                                    for k, v in secondary.items()},
                      'reasons': {m: t['crop_end_reasons'] for m, t in table.items()}}, indent=1))


if __name__ == '__main__':
    main()
