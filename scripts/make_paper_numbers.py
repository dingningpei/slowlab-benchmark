#!/usr/bin/env python3
"""Every result number the paper prints, generated from the locked analysis outputs.

    make_paper_numbers.py --out paper/generated/numbers.tex

Writes LaTeX definitions used as \\res{key} (keys like E1.deepseek.mean) and a JSON copy
(paper/generated/numbers.json) that scripts/verify_results_claims.py checks against the sources.
Phase 6 numbers are added when results/phase6_analysis_*.json exists.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHORT = {'deepseek-v4.1-flash': 'deepseek', 'glm-5.3-flash': 'glm', 'mimo-v2.6-flash': 'mimo', 'pbo': 'baseline'}
SOURCES = {'formal': 'results/formal_analysis_e1e2_20261005.json', 'accounting': 'results/formal_accounting_20261005.json',
           'feedback': 'results/formal_process_feedback_20261005.json', 'audit': 'results/formal_audit_20261005.json'}


def fmt(x, digits=2, sign=False):
    if x is None:
        return '--'
    if isinstance(x, int):
        return str(x)
    s = f'{x:+.{digits}f}' if sign else f'{x:.{digits}f}'
    return s.replace('-', '$-$') if s.startswith('-') else s


def p_fmt(p):
    if p < 0.001:
        return '$<$0.001'
    return f'{p:.3f}' if p < 0.1 else f'{p:.2f}'


def numbers():
    src = {k: json.loads((ROOT / v).read_text()) for k, v in SOURCES.items()}
    f = src['formal']
    n = {'sites': f['sites'], 'campaigns': src['audit']['campaigns_planned'], 'evaluations': src['audit']['evaluations_planned'],
         'spend.campaigns': fmt(src['audit']['ledger_campaign_usd']), 'spend.total': fmt(src['audit']['ledger_total_usd']),
         'calls.total': sum(src['audit']['llm_calls_by_model'].values())}
    for m, v in f['method_means'].items():
        model, _, cond = m.rpartition('_') if m != 'fixed_reference' else ('fixed', '', 'reference')
        key = f"mean.{SHORT.get(model, model)}.{cond}"
        n[key] = fmt(v['mean'])
    for name, v in f['primary'].items():
        exp, rest = name.split('_', 1)
        model = rest.split('_full_minus')[0]
        k = f"{exp}.{SHORT[model]}"
        n[f'{k}.mean'] = fmt(v['mean'], sign=True)
        n[f'{k}.lo'] = fmt(v['ci95_bca'][0], sign=True)
        n[f'{k}.hi'] = fmt(v['ci95_bca'][1], sign=True)
        n[f'{k}.p'] = p_fmt(v['p_holm'])
        n[f'{k}.mde'] = fmt(v['minimum_detectable_80'])
        n[f'{k}.sd'] = fmt(v['sd'])
    for name, v in f['secondary'].items():
        if name.endswith('_full_minus_initial'):
            k = f"learn.{SHORT[name.split('_full_minus')[0]]}"
        elif name.endswith('_full_minus_fixed_reference'):
            k = f"vsfixed.{SHORT[name.split('_full_minus')[0]]}"
        else:
            a, b = name[len('feedback_gain_'):].split('_minus_')
            k = f"gaindiff.{SHORT[a]}.{SHORT[b]}"
        n[f'{k}.mean'] = fmt(v['mean'], sign=True)
        n[f'{k}.lo'] = fmt(v['ci95_bca'][0], sign=True)
        n[f'{k}.hi'] = fmt(v['ci95_bca'][1], sign=True)
        n[f'{k}.p'] = p_fmt(v['p_bh'])
    res = f['resources_per_crop_of_recommended_policies']
    for m, v in res.items():
        model, _, cond = m.rpartition('_') if m != 'fixed_reference' else ('fixed', '', 'reference')
        for r in ('heat_kwh_m2', 'light_kwh_m2', 'co2_kg_m2'):
            n[f"res.{SHORT.get(model, model)}.{cond}.{r.split('_')[0]}"] = fmt(v[r], 0 if r != 'co2_kg_m2' else 1)
    acc = src['accounting']['by_model']
    for m, v in acc.items():
        n[f'cost.{SHORT[m]}'] = fmt(v['mean_usd'], 3)
        n[f'calls.{SHORT[m]}'] = fmt(v['mean_calls'], 0)
        n[f'formaterr.{SHORT[m]}'] = fmt(v['mean_full_format_errors'], 1)
        n[f'observe.{SHORT[m]}'] = fmt(v['mean_full_observe_calls'], 1)
    for m, v in src['feedback']['summary_by_model'].items():
        n[f'fb.{SHORT[m]}.campaigns'] = v['campaigns']
        n[f'fb.{SHORT[m]}.stops'] = v['any_stop']
        n[f'fb.{SHORT[m]}.sameschedule'] = v['same_later_schedule']
        n[f'fb.{SHORT[m]}.reads'] = fmt(v['mean_full_reads_before_first_later_start'], 1)
        n[f'fb.{SHORT[m]}.predictor'] = fmt(v['mean_full_predictor_calls'], 1)
    phase6 = sorted(glob.glob(str(ROOT / 'results/phase6_analysis_*.json')))
    if phase6:
        p6 = json.loads(Path(phase6[-1]).read_text())
        for rd, v in p6['e3']['reader_swap_gain_pooled'].items():
            if v.get('n'):
                k = f"E3.{SHORT.get(rd, 'gp' if rd == 'gp-reader' else rd)}"
                n[f'{k}.mean'] = fmt(v['mean'], sign=True)
                n[f'{k}.lo'] = fmt(v['ci95_bca'][0], sign=True)
                n[f'{k}.hi'] = fmt(v['ci95_bca'][1], sign=True)
                n[f'{k}.p'] = p_fmt(p6['e3']['secondary_bh_with_e3'][f'E3_reader_swap_{rd}'])
        rs = p6['reference_search']
        if rs.get('mean_headroom_over_fixed_reference') is not None:
            n['ref.headroom'] = fmt(rs['mean_headroom_over_fixed_reference'])
    return {k: (v if isinstance(v, str) else str(v)) for k, v in n.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'paper/generated/numbers.tex')
    args = ap.parse_args()
    n = numbers()
    lines = ['% Generated by scripts/make_paper_numbers.py; do not edit.',
             '\\makeatletter', '\\newcommand{\\res}[1]{\\@ifundefined{res@#1}{\\textbf{??#1}}{\\@nameuse{res@#1}}}']
    lines += [f'\\@namedef{{res@{k}}}{{{v}}}' for k, v in sorted(n.items())]
    lines.append('\\makeatother')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(lines) + '\n')
    args.out.with_suffix('.json').write_text(json.dumps({'sources': SOURCES, 'numbers': n}, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'numbers': len(n)}))


if __name__ == '__main__':
    main()
