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
           'feedback': 'results/formal_process_feedback_20261005.json', 'audit': 'results/formal_audit_20261005.json',
           'pilot': 'results/pilot_evaluation_analysis_20261003.json', 'pilot_pbo': 'results/pilot_evaluation_analysis_pbo_20261004.json',
           'kernel': 'results/prior_bo_hyperparameter_fit_20261003.json', 'radius': 'results/prior_bo_radius_selection_20261004.json',
           'agc': 'configs/agc/agc2019_holdout_result_v8.json', 'parity': 'results/server_parity_check_20261004.json'}


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
    pl, pp = src['pilot'], src['pilot_pbo']
    n['pilot.bo.vsfixed'] = fmt(pl['secondary_preview']['bo_full_minus_fixed_reference']['mean'], sign=True)
    n['pilot.bo.repeatsd'] = fmt(pl['methods']['bo_full']['within_site_repeat_sd'])
    n['pilot.baseline.repeatsd'] = fmt(pp['methods']['pbo_full']['within_site_repeat_sd'])
    n['pilot.llm.repeatsd.lo'] = fmt(min(pl['methods'][f'{m}_full']['within_site_repeat_sd'] for m in SHORT if m != 'pbo'))
    n['pilot.llm.repeatsd.hi'] = fmt(max(pl['methods'][f'{m}_full']['within_site_repeat_sd'] for m in SHORT if m != 'pbo'))
    k = src['kernel']['held_out_check']
    n['kernel.cross.product'] = fmt(k['cross_season']['product']['ranking_accuracy'])
    n['kernel.cross.interaction'] = fmt(k['cross_season']['season_interaction']['ranking_accuracy'])
    n['kernel.rmse.interaction'] = fmt(k['random']['season_interaction']['rmse'], 1)
    n['kernel.rmse.refit'] = fmt(k['random']['refit_gp_bo_v3']['rmse'], 1)
    n['kernel.rmse.mean'] = fmt(k['random']['mean_of_training']['rmse'], 1)
    n['kernel.groups'] = src['kernel']['hyperparameters']['groups']
    n['kernel.crops'] = src['kernel']['hyperparameters']['observations']
    rad = src['radius']
    n['radius.selected'] = str(int(round(rad['selected_radius'] * 100)))
    n['radius.fixed'] = fmt(rad['fixed_reference_mean_eur_m2'])
    for r, v in rad['radii'].items():
        n[f"radius.{int(round(float(r) * 100))}.mean"] = fmt(v['mean_eur_m2'])
        n[f"radius.{int(round(float(r) * 100))}.sd"] = fmt(v['within_site_seed_sd'])
    n['radius.v3.mean'] = fmt(rad['gp_bo_v3_same_sites']['mean_eur_m2'])
    import math
    agg = src['agc']['result']['aggregate']
    total = sum(v['samples'] for v in agg.values())
    for metric, name in (('tAir', 'temp'), ('rhIn', 'rh'), ('co2InPpm', 'co2')):
        pooled = math.sqrt(sum(v['samples'] * v['metrics'][metric]['rmse'] ** 2 for v in agg.values()) / total)
        n[f'agc.{name}.rmse'] = fmt(pooled, 3 if name != 'co2' else 1)
        n[f'agc.{name}.limit'] = fmt(float(src['agc']['limits'][metric]), 2 if name != 'co2' else 0)
    n['parity.identical'] = 'bit-identical' if src['parity']['records_identical_except_timing'] else 'NOT identical'
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
        # The preregistered secondary family includes E3: once E3 exists, BH p-values come from the joint family.
        joint = p6['e3']['secondary_bh_with_e3']
        for name in f['secondary']:
            if name.endswith('_full_minus_initial'):
                k = f"learn.{SHORT[name.split('_full_minus')[0]]}"
            elif name.endswith('_full_minus_fixed_reference'):
                k = f"vsfixed.{SHORT[name.split('_full_minus')[0]]}"
            else:
                a, b = name[len('feedback_gain_'):].split('_minus_')
                k = f"gaindiff.{SHORT[a]}.{SHORT[b]}"
            n[f'{k}.p'] = p_fmt(joint[name])
        n['secondary.family'] = len(joint)
        for rd, per in p6['e3']['reader_swap_gain'].items():
            for src_m, v in per.items():
                if v.get('n'):
                    n[f"E3.{SHORT.get(rd, 'gp' if rd == 'gp-reader' else rd)}.on.{SHORT[src_m]}"] = fmt(v['mean'], sign=True)
        rs = p6['reference_search']
        if rs.get('mean_headroom_over_fixed_reference') is not None:
            n['ref.headroom'] = fmt(rs['mean_headroom_over_fixed_reference'])
            n['ref.sites'] = len(rs['sites'])
            errs = rs['search_error_abs_difference']
            n['ref.err.lo'], n['ref.err.hi'] = fmt(min(errs)), fmt(max(errs))
            import numpy as _np
            for m in SHORT:
                vals = [r[m] - r['fixed_reference'] for r in rs['sites'].values() if r.get(m) is not None]
                n[f'ref.{SHORT[m]}.vsfixed'] = fmt(float(_np.mean(vals)), sign=True)
        b = p6['boundary_sensitivity']['settings']
        for s, v in b.items():
            for k, x in v['differences'].items():
                if k.endswith('_minus_baseline'):
                    n[f"ueff.{s}.{SHORT[k[:-len('_minus_baseline')]]}"] = fmt(x['mean'], sign=True)
        n['ueff.sites'] = len(p6['boundary_sensitivity']['sites'])
        for f_, v in p6['dry_matter_sensitivity'].items():
            for name, x in v.items():
                if name.startswith('E1_'):
                    n[f"dm.{f_}.{SHORT[name[3:].split('_full_minus')[0]]}"] = fmt(x['mean'], sign=True)
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
