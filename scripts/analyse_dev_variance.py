#!/usr/bin/env python3
"""Variance components from the development variance study and a preliminary site count.

Inputs: the study's campaign records (to find each campaign's site, seed and
feedback) and evaluation records (score per evaluation year). A
recommendation's score is its mean over the site's evaluation years. For the
staggered BO, Full - Endpoint per site and seed gives the E2 difference;
optionally BO (Full) - fixed reference per site gives an E1-like difference.
Within-site (between-seed) and between-site spreads give the SD of a site's
mean difference with R repeats, and the site count for a two-sided paired
t-test (power 0.8) at the Holm worst-case level (alpha/7) for 2 EUR/m2.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def n_paired(delta, sd, alpha, power=0.8):
    for n in range(2, 5000):
        df, tcrit = n - 1, stats.t.ppf(1 - alpha / 2, n - 1)
        nc = delta / (sd / math.sqrt(n))
        if 1 - stats.nct.cdf(tcrit, df, nc) + stats.nct.cdf(-tcrit, df, nc) >= power:
            return n
    return None


def components(diffs):
    """diffs: {site: [difference per repeat]} -> (between-site SD of true diffs, within-site SD, site-mean SD for R)."""
    sites = [v for v in diffs.values() if len(v) >= 1]
    within = [np.var(v, ddof=1) for v in sites if len(v) > 1]
    s2_within = float(np.mean(within)) if within else 0.0
    r = np.mean([len(v) for v in sites])
    means = [np.mean(v) for v in sites]
    s2_means = float(np.var(means, ddof=1)) if len(means) > 1 else float('nan')
    s2_between = max(s2_means - s2_within / r, 0.0)
    return math.sqrt(s2_between), math.sqrt(s2_within), math.sqrt(s2_means), float(np.mean(means))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--fixed-reference-dir', type=Path, default=None)
    ap.add_argument('--reference', default=None, help='fixed-reference candidate name for the E1-like difference')
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    scores = defaultdict(list)
    for f in glob.glob(str(args.run_dir / 'evaluations' / 'site*_eval*.json')):
        m = re.search(r'site(\d+)_seed(\d+)_(full|endpoint)_eval(\d{4})', f)
        r = json.loads(Path(f).read_text())
        if r.get('status') == 'completed':
            scores[(int(m[1]), int(m[2]), m[3])].append(r['score_eur_m2'])
    rec = {k: float(np.mean(v)) for k, v in scores.items() if len(v) == 3}
    sites = sorted({k[0] for k in rec}); seeds = sorted({k[1] for k in rec})
    e2 = {s: [rec[(s, r, 'full')] - rec[(s, r, 'endpoint')] for r in seeds if (s, r, 'full') in rec and (s, r, 'endpoint') in rec]
          for s in sites}
    out = {'sites': len(sites), 'seeds': seeds, 'recommendation_scores': {f'{k[0]}_{k[1]}_{k[2]}': v for k, v in sorted(rec.items())}}
    b, w, sm, mean = components(e2)
    out['e2_staggered_bo'] = {'mean_full_minus_endpoint': mean, 'between_site_sd': b, 'within_site_seed_sd': w,
                              'site_mean_sd_with_repeats': sm}
    if args.fixed_reference_dir and args.reference:
        ref = defaultdict(list)
        for f in glob.glob(str(args.fixed_reference_dir / f'site*_{args.reference}_*.json')):
            m = re.search(r'site(\d+)_', f)
            r = json.loads(Path(f).read_text())
            if r.get('status') == 'completed':
                ref[int(m[1])].append(r['score_eur_m2'])
        ref = {s: float(np.mean(v)) for s, v in ref.items() if len(v) == 3}
        e1 = {s: [rec[(s, r, 'full')] - ref[s] for r in seeds if (s, r, 'full') in rec] for s in sites if s in ref}
        b1, w1, sm1, mean1 = components(e1)
        out['bo_full_minus_fixed_reference'] = {'reference': args.reference, 'mean': mean1, 'between_site_sd': b1,
                                                'within_site_seed_sd': w1, 'site_mean_sd_with_repeats': sm1}
    table = {}
    for name in ('e2_staggered_bo', 'bo_full_minus_fixed_reference'):
        if name in out:
            sd = out[name]['site_mean_sd_with_repeats']
            table[name] = {'sd_used': sd, 'sites_for_2_eur_alpha_0.05': n_paired(2.0, sd, 0.05),
                           'sites_for_2_eur_holm_worst_alpha_0.05_over_7': n_paired(2.0, sd, 0.05 / 7)}
    out['preliminary_site_counts'] = table
    out['note'] = ('BO only: the LLM campaign spread is unknown until the Phase 4 pilot; the final N uses the pilot '
                   'variance. Development sites, development years.')
    args.report.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('sites', 'e2_staggered_bo', 'preliminary_site_counts')}, indent=1))


if __name__ == '__main__':
    main()
