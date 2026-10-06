#!/usr/bin/env python3
"""Figures 2 (E1) and 3 (E2) from the formal evaluations and the locked analysis report.

    make_figures_e1e2.py --evaluations EVAL_DIR --campaigns campaigns.json --report results/formal_analysis_e1e2_20261005.json \
        --out paper/figures

Site-level values follow the locked analysis (three-year mean per recommendation, repeats averaged
within a site). Intervals and p-values are taken from the report, not recomputed.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABEL = {'deepseek-v4.1-flash': 'DeepSeek V4.1 Flash', 'glm-5.3-flash': 'GLM-5.3 Flash',
         'mimo-v2.6-flash': 'MiMo V2.6 Flash', 'pbo': 'Prior-informed local BO'}
# Categorical palette validated with the dataviz validator (light surface): CVD and normal-vision checks pass;
# aqua is below 3:1 contrast, so every series is also named by its axis label or legend.
COLOR = {'deepseek-v4.1-flash': '#2a78d6', 'glm-5.3-flash': '#eb6834', 'mimo-v2.6-flash': '#1baf7a', 'pbo': '#4a3aa7'}
NEUTRAL = '#8a8984'



def p_label(p):
    """Same rounding as the paper text (scripts/make_paper_numbers.py p_fmt)."""
    if p < 0.001:
        return '<0.001'
    return '=' + (f'{p:.3f}' if p < 0.1 else f'{p:.2f}')

def site_scores(eval_dir):
    years = defaultdict(dict)
    for f in eval_dir.glob('site*_y*.json'):
        m = re.match(r'site(\d+)_(.+)_y(\d{4})\.json$', f.name)
        r = json.loads(f.read_text())
        if r.get('status') == 'completed':
            years[(int(m[1]), m[2])][int(m[3])] = r
    score = {k: float(np.mean([r['score_eur_m2'] for r in v.values()])) for k, v in years.items() if len(v) == 3}
    res = {k: np.mean([[c['accrued'][x] for x in ('heat_kwh_m2', 'light_kwh_m2', 'co2_kg_m2')]
                       for r in v.values() for p in r['plantings'] for c in p['crops']], axis=0)
           for k, v in years.items() if len(v) == 3}
    return score, res


def method_site(score, method, site):
    if method == 'fixed_reference':
        keys = [f'fixed_reference']
    elif method.startswith('pbo_'):
        cond = method[4:]
        keys = [f'pbo_s{s}_{cond}' for s in (0, 1)]
    elif method.endswith('_initial'):
        keys = [f'{method[:-8]}_r0_initial']
    else:
        model, cond = method.rsplit('_', 1)
        keys = [f'{model}_r{r}_{cond}' for r in (0, 1)]
    vals = [score.get((site, k)) for k in keys]
    return None if None in vals else float(np.mean(vals))


def interval_plot(ax, rows, report_rows, title, xlabel):
    for i, (name, diffs) in enumerate(rows):
        y = len(rows) - 1 - i
        jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(diffs))
        ax.scatter(diffs, y + jitter, s=7, color=COLOR.get(name, 'grey'), alpha=0.35, linewidths=0)
        r = report_rows[name]
        ax.errorbar(r['mean'], y, xerr=[[r['mean'] - r['ci95_bca'][0]], [r['ci95_bca'][1] - r['mean']]], fmt='o',
                    color='black', capsize=3, ms=4)
        kind, p = ('Holm', r['p_holm']) if 'p_holm' in r else ('BH', r['p_bh'])
        ax.text(1.01, y, f"{r['mean']:+.2f}\np$_{{{kind}}}${p_label(p)}", transform=ax.get_yaxis_transform(), va='center', fontsize=7)
    ax.axvline(0, color='grey', lw=0.8)
    for v in (-2, 2):
        ax.axvline(v, color='grey', lw=0.6, ls=':')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABEL.get(n, n) for n, _ in reversed(rows)], fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluations', type=Path, required=True)
    ap.add_argument('--campaigns', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--phase6', type=Path, default=None,
                    help='Phase 6 report: secondary p-values then come from the joint family that includes E3')
    args = ap.parse_args()
    report = json.loads(args.report.read_text())
    if args.phase6:
        joint = json.loads(args.phase6.read_text())['e3']['secondary_bh_with_e3']
        for name, v in report['secondary'].items():
            v['p_bh'] = joint[name]
    batch = json.loads(args.campaigns.read_text())
    sites = sorted({j['site']['site_index'] for j in batch['jobs']})
    llms = sorted({j['model_name'] for j in batch['jobs'] if j['method'] == 'llm'})
    score, res = site_scores(args.evaluations)

    def diffs(a, b):
        out = []
        for s in sites:
            va, vb = method_site(score, a, s), method_site(score, b, s)
            if va is not None and vb is not None:
                out.append(va - vb)
        return out
    args.out.mkdir(parents=True, exist_ok=True)
    # Figure 2: E1
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), gridspec_kw={'width_ratios': [1.2, 1.2, 0.9], 'wspace': 0.45})
    e1 = [(m, diffs(f'{m}_full', 'pbo_full')) for m in llms]
    interval_plot(axes[0], e1, {m: report['primary'][f'E1_{m}_full_minus_baseline_full'] for m in llms},
                  'E1: final policy, model minus main baseline', 'EUR/m² (site-level paired difference)')
    learn = [(m, diffs(f'{m}_full', f'{m}_initial')) for m in llms]
    interval_plot(axes[1], learn, {m: report['secondary'][f'{m}_full_minus_initial'] for m in llms},
                  'Gain over the pre-experiment recommendation', 'EUR/m² (final minus initial)')
    axes[1].set_yticklabels([])  # same model order as the left panel
    methods = llms + ['pbo', 'fixed_reference']
    names = ['heat_kwh_m2', 'light_kwh_m2', 'co2_kg_m2']
    resources = report['resources_per_crop_of_recommended_policies']
    keys = [f'{m}_full' for m in llms] + ['pbo_full', 'fixed_reference']
    width = 0.8 / len(keys)
    for i, k in enumerate(keys):
        vals = [resources[k][n] for n in names[:2]]
        axes[2].bar(np.arange(2) + i * width, vals, width, label=LABEL.get(k.replace('_full', ''), 'Fixed reference'),
                    color=COLOR.get(k.replace('_full', ''), NEUTRAL))
    axes[2].set_xticks(np.arange(2) + 0.4 - width / 2)
    axes[2].set_xticklabels(['Heat', 'Light'], fontsize=8)
    axes[2].set_ylabel('kWh/m² per crop', fontsize=8)
    axes[2].set_title('Resources of recommended policies', fontsize=9)
    axes[2].legend(fontsize=6, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    axes[2].tick_params(labelsize=7)
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.3, top=0.88)
    for ext in ('pdf', 'png'):
        fig.savefig(args.out / f'fig2_e1.{ext}', dpi=200)
    # Figure 3: E2
    fig, ax = plt.subplots(figsize=(6, 2.6))
    e2 = [(m, diffs(f'{m}_full' if m != 'pbo' else 'pbo_full', f'{m}_endpoint' if m != 'pbo' else 'pbo_endpoint'))
          for m in llms + ['pbo']]
    interval_plot(ax, e2, {m: report['primary'][f'E2_{m}_full_minus_endpoint'] for m in llms + ['pbo']},
                  'E2: process feedback, Full minus Endpoint', 'EUR/m² (site-level paired difference)')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(args.out / f'fig3_e2.{ext}', dpi=200)
    print(json.dumps({'sites': len(sites), 'e1_n': [len(d) for _, d in e1], 'e2_n': [len(d) for _, d in e2]}))


if __name__ == '__main__':
    main()
