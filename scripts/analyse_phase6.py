#!/usr/bin/env python3
"""Phase 6 analysis: E3 Readers, boundary and dry-matter sensitivity, best-known reference (locked before the data).

    analyse_phase6.py --formal RUN --phase6 DIR --e3 E3DIR --formal-report FORMAL.json --report OUT.json

E3: a recommendation's score is its mean over the packet site's three evaluation years. Reader
swap gain = Reader score minus the source method's own recommendation score on the same packet.
Per Reader and source method (16 packets on distinct sites) and per Reader pooled (packets
averaged within a site first): mean, BCa 95% interval, sign-flip p. The pooled Reader tests join
the secondary family: BH is recomputed over the E1/E2 secondary tests and these. History utility
for a fixed Reader: mean Reader score by source method with BCa intervals (different sites per
source, so descriptive). Boundary sensitivity: for the 16 sites, each method's score at Ueff 0 and 4
(three-year mean) beside its main score; the E1-type differences (each model minus the main
baseline) and each method minus the fixed reference at every Ueff; whether the method ranking by
mean score changes. Dry matter 0.05 and 0.07: every primary contrast recomputed from the main
evaluations (fresh-equivalent harvest scales by 0.06 / f; revenue changes by that harvest change
times price minus handling). Reference search: per searched site the best-known final mean, the
fixed reference and each method's Full score on that site (headroom), and the search error from
the sites searched twice.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.statistics import bca_interval, benjamini_hochberg, sign_flip_test  # noqa: E402

DM_BASE, DM_LEVELS = 0.06, (0.05, 0.07)


def stats(d, seed):
    d = [x for x in d if x is not None]
    if not d:
        return {'n': 0}
    lo, hi = bca_interval(d, seed=seed)
    return {'n': len(d), 'mean': float(np.mean(d)), 'ci95_bca': [lo, hi], 'p_sign_flip': sign_flip_test(d, seed=seed)}


def mean3(scores):
    return float(np.mean(scores)) if len(scores) == 3 else None


def load_scores(directory, pattern):
    out = defaultdict(dict)
    for f in directory.glob('*.json'):
        m = re.fullmatch(pattern, f.name)
        if not m:
            continue
        r = json.loads(f.read_text())
        if r.get('status') == 'completed':
            out[m['key']][int(m['year'])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--formal', type=Path, required=True)
    ap.add_argument('--phase6', type=Path, required=True)
    ap.add_argument('--e3', type=Path, required=True)
    ap.add_argument('--formal-report', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    formal_eval = load_scores(args.formal / 'evaluations', r'(?P<key>site\d+_.+)_y(?P<year>\d{4})\.json')
    score = {k: mean3([r['score_eur_m2'] for r in v.values()]) for k, v in formal_eval.items()}
    out = {'format': 'slowlab-phase6-analysis-v1'}
    # E3
    sample = json.loads((args.e3 / 'sample.json').read_text())['packets']
    readers_eval = load_scores(args.e3 / 'evaluations', r'(?P<key>p\d+_.+)_y(?P<year>\d{4})\.json')
    rscore = {k: mean3([r['score_eur_m2'] for r in v.values()]) for k, v in readers_eval.items()}
    readers = sorted({k.split('_', 1)[1] for k in rscore})
    gains = defaultdict(lambda: defaultdict(list))
    by_site = defaultdict(lambda: defaultdict(list))
    utility = defaultdict(lambda: defaultdict(list))
    for p in sample:
        own = score.get(p['source_job'] + ('_full' if not p['source_job'].endswith('_full') else ''))
        site = int(p['source_job'][4:6])
        for rd in readers:
            s = rscore.get(f"{p['packet_id']}_{rd}")
            if s is None:
                continue
            utility[rd][p['source_method']].append(s)
            if own is not None:
                gains[rd][p['source_method']].append(s - own)
                by_site[rd][site].append(s - own)
    e3 = {'reader_swap_gain': {}, 'reader_swap_gain_pooled': {}, 'history_utility': {}}
    k = 300
    for rd in readers:
        e3['reader_swap_gain'][rd] = {m: stats(v, k + i) for i, (m, v) in enumerate(sorted(gains[rd].items()))}
        e3['reader_swap_gain_pooled'][rd] = stats([float(np.mean(v)) for v in by_site[rd].values()], k + 50)
        e3['history_utility'][rd] = {m: {'n': len(v), 'mean': float(np.mean(v)), 'ci95_bca': list(bca_interval(v, seed=k + 70))}
                                     for m, v in sorted(utility[rd].items())}
        k += 100
    formal = json.loads(args.formal_report.read_text())
    family = {n: v['p_sign_flip'] for n, v in formal['secondary'].items()}
    family.update({f'E3_reader_swap_{rd}': v['p_sign_flip'] for rd, v in e3['reader_swap_gain_pooled'].items() if v.get('n')})
    e3['secondary_bh_with_e3'] = benjamini_hochberg(family)
    out['e3'] = e3
    # boundary sensitivity
    sens = load_scores(args.phase6 / 'sensitivity', r'(?P<key>site\d+_.+_ueff[\d.]+)_y(?P<year>\d{4})\.json')
    sscore = {k: mean3([r['score_eur_m2'] for r in v.values()]) for k, v in sens.items()}
    picks = json.loads((args.phase6 / 'picks.json').read_text())
    methods = sorted({re.match(r'site\d+_(.+)_ueff', k)[1] for k in sscore})
    main_key = {m: (f'{m}_r0_full' if m not in ('pbo', 'fixed_reference') else ('pbo_s0_full' if m == 'pbo' else 'fixed_reference'))
                for m in methods}
    table = {}
    for setting in ('main', '0', '4'):
        per = {}
        for m in methods:
            vals = {}
            for s in picks['boundary_sensitivity_sites']:
                key = f'site{s:02d}_{main_key[m]}' if setting == 'main' else f'site{s:02d}_{m}_ueff{setting}'
                v = (score if setting == 'main' else sscore).get(key)
                if v is not None:
                    vals[s] = v
            per[m] = vals
        llms = [m for m in methods if m not in ('pbo', 'fixed_reference')]
        diffs = {}
        for m in llms:
            diffs[f'{m}_minus_baseline'] = stats([per[m][s] - per['pbo'][s] for s in per[m] if s in per['pbo']], 400)
        for m in llms + ['pbo']:
            diffs[f'{m}_minus_fixed_reference'] = stats([per[m][s] - per['fixed_reference'][s] for s in per[m]
                                                         if s in per['fixed_reference']], 450)
        means = {m: float(np.mean(list(v.values()))) for m, v in per.items() if v}
        table[setting] = {'means': means, 'ranking': sorted(means, key=lambda m: -means[m]), 'differences': diffs}
    out['boundary_sensitivity'] = {'sites': picks['boundary_sensitivity_sites'], 'settings': table,
                                   'ranking_changes': {s: table[s]['ranking'] != table['main']['ranking'] for s in ('0', '4')}}
    # dry matter
    from slowlab.private_runs import prepare
    econ = json.loads((ROOT / 'configs/task_contract_v8.json').read_text())['economics']
    batch = json.loads((args.formal / 'campaigns.json').read_text())
    price = {}
    for j in batch['jobs']:
        s = j['site']['site_index']
        if s not in price:
            site = prepare({'contract': 'configs/task_contract_v8.json', 'year': j['year'], 'site': j['site'], 'backend': 'fake'})['site']
            price[s] = site['prices'].get('price_eur_per_kg_fresh_equivalent', econ['price_eur_per_kg_fresh_equivalent'])
    dm = {}
    for f in DM_LEVELS:
        adj = {}
        for key, years in formal_eval.items():
            s = int(key[4:6])
            vals = []
            for r in years.values():
                plantings = []
                for p in r['plantings']:
                    margins = [c['margin_eur_m2'] + c['accrued']['harvest_kg_m2'] * (DM_BASE / f - 1) *
                               (price[s] - econ['harvest_handling_eur_per_kg']) for c in p['crops']]
                    plantings.append(np.mean(margins))
                vals.append(float(np.mean(plantings)))
            adj[key] = mean3(vals)
        contrasts = {}
        for name in formal['primary']:
            m = re.match(r'E1_(.+)_full_minus_baseline_full|E2_(.+)_full_minus_endpoint', name)
            a, b = ((f'{m[1]}_r{{r}}_full', 'pbo_s{r}_full') if m[1] else
                    ((f'{m[2]}_r{{r}}_full', f'{m[2]}_r{{r}}_endpoint') if m[2] != 'pbo' else ('pbo_s{r}_full', 'pbo_s{r}_endpoint')))
            d = []
            for s in sorted(price):
                va = [adj.get(f'site{s:02d}_' + a.format(r=r)) for r in (0, 1)]
                vb = [adj.get(f'site{s:02d}_' + b.format(r=r)) for r in (0, 1)]
                if None not in va and None not in vb:
                    d.append(float(np.mean(va) - np.mean(vb)))
            contrasts[name] = stats(d, 500)
        dm[str(f)] = contrasts
    out['dry_matter_sensitivity'] = dm
    # reference search
    ref = {}
    for f in sorted((args.phase6 / 'reference').glob('site*_seed*.json')):
        r = json.loads(f.read_text())
        s, sd = re.match(r'site(\d+)_seed(\d)', f.stem).groups()
        ref.setdefault(int(s), {})[int(sd)] = r['best']['final_mean'] if r.get('best') else None
    headroom = {}
    for s, seeds in sorted(ref.items()):
        best = max(v for v in seeds.values() if v is not None) if any(v is not None for v in seeds.values()) else None
        row = {'best_known': best, 'by_seed': seeds, 'fixed_reference': score.get(f'site{s:02d}_fixed_reference')}
        for m in sorted({j['model_name'] for j in batch['jobs'] if j['method'] == 'llm'}) + ['pbo']:
            keys = [f'site{s:02d}_{m}_r{r}_full' if m != 'pbo' else f'site{s:02d}_pbo_s{r}_full' for r in (0, 1)]
            vals = [score.get(k) for k in keys]
            row[m] = float(np.mean(vals)) if None not in vals else None
        headroom[s] = row
    twice = [abs(v[0] - v[1]) for v in ref.values() if len(v) == 2 and None not in v.values()]
    out['reference_search'] = {'sites': headroom, 'search_error_abs_difference': twice,
                               'mean_headroom_over_fixed_reference': float(np.mean([r['best_known'] - r['fixed_reference'] for r in headroom.values()
                                                                                    if r['best_known'] is not None and r['fixed_reference'] is not None]))
                               if headroom else None}
    args.report.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'e3_pooled': {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'ci95_bca'}
                                    for k, v in e3['reader_swap_gain_pooled'].items()},
                      'ranking_changes': out['boundary_sensitivity']['ranking_changes']}, indent=1))


if __name__ == '__main__':
    main()
