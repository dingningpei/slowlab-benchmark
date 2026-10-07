#!/usr/bin/env python3
"""Figure 1 (teaser): one real campaign year and the headline E1 result.

    make_figure_teaser.py --campaigns A/campaigns --report results/repaired_formal_analysis_e1e2_20261006.json --out paper/figures

Selection rule (fixed before plotting, RESEARCH_PLAN §6): test site 0, repeat 0, Full branch of the first
language model in alphabetical order (DeepSeek V4.1 Flash). Left: crops in the four compartments (bars),
channel reads (ticks), the recommendation and the private evaluation. Right: E1 differences with BCa
intervals, taken from the locked analysis report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figure1 import CLEAN, CROP, events  # noqa: E402
from make_figures_e1e2 import COLOR, p_label  # noqa: E402

MODEL = 'deepseek-v4.1-flash'
LABEL = {'deepseek-v4.1-flash': 'DeepSeek V4.1 Flash', 'glm-5.3-flash': 'GLM-5.3 Flash', 'mimo-v2.6-flash': 'MiMo V2.6 Flash'}
INK, MUTED = '#2b2b2b', '#8a8984'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaigns', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    tr = json.loads((args.campaigns / f'site00_{MODEL}_r0.json').read_text())['branches']['full']['public_transcript']
    starts, stops, reads = events(tr)
    report = json.loads(args.report.read_text())['primary']

    fig = plt.figure(figsize=(7.2, 2.35))
    ax = fig.add_axes([0.055, 0.2, 0.56, 0.68])
    for day, unit in starts:
        end = min([d for d, u in stops if u == unit and d > day] + [day + CROP, 365])
        ax.barh(unit, end - day, left=day, height=0.56, color=COLOR[MODEL], alpha=0.85, edgecolor='white', lw=0.6)
        ax.barh(unit, min(CLEAN, 365 - end), left=end, height=0.56, color='#d9d8d4')
    for day, unit in reads:
        ax.plot([day, day], [unit - 0.38, unit + 0.38], color=INK, lw=1.0)
    ax.axvline(365, color=MUTED, lw=0.8, ls=':')
    ax.set_xlim(0, 400)
    ax.set_ylim(3.6, -0.5)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f'C{u + 1}' for u in range(4)], fontsize=7)
    ax.set_xticks([0, 90, 180, 270, 365])
    ax.tick_params(axis='x', labelsize=7)
    ax.set_xlabel('Campaign day', fontsize=7)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.set_title(f'One campaign: {LABEL[MODEL]}, test site 0 (Full feedback)', fontsize=8, loc='left')
    ax.annotate('day-0 design commits\ntwo compartments', xy=(4, 1.0), xytext=(14, 3.3), fontsize=6, color=INK,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
    ax.annotate('in-season reads\n(Full feedback only)', xy=(90, 0.3), xytext=(96, 2.3), fontsize=6, color=INK,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
    ax.annotate('second wave only after\nthe first wave closes', xy=(182, 2.0), xytext=(205, 1.1), fontsize=6, color=INK,
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
    ax.text(368, 1.5, 'recommend\none policy\n\n→ private\nevaluator,\n3 unseen\nweather years', fontsize=6,
            color=INK, va='center')

    bx = fig.add_axes([0.73, 0.2, 0.25, 0.68])
    names = ['deepseek-v4.1-flash', 'mimo-v2.6-flash', 'glm-5.3-flash']
    for y, m in enumerate(names):
        r = report[f'E1_{m}_full_minus_baseline_full']
        lo, hi = r['ci95_bca']
        bx.plot([lo, hi], [y, y], color=COLOR[m], lw=2.0, solid_capstyle='round')
        bx.plot(r['mean'], y, 'o', color=COLOR[m], ms=5, mec='white', mew=0.8)
        bx.text(hi + 0.25, y, f"{r['mean']:+.1f}", va='center', fontsize=6.5, color=INK)
    bx.axvline(0, color=MUTED, lw=0.8)
    bx.axvspan(-2, 2, color='#ecebe7', zorder=0)
    bx.set_yticks(range(len(names)))
    bx.set_yticklabels([LABEL[m].replace(' Flash', '') for m in names], fontsize=7)
    bx.set_ylim(len(names) - 0.4, -0.6)
    bx.set_xlim(-8, 3)
    bx.tick_params(axis='x', labelsize=7)
    bx.set_xlabel('Final policy minus baseline (EUR/m$^2$)', fontsize=7)
    for side in ('top', 'right'):
        bx.spines[side].set_visible(False)
    p = max(report[f'E1_{m}_full_minus_baseline_full']['p_holm'] for m in names)
    bx.set_title(f'48 sites, all p$_{{Holm}}${p_label(p)}', fontsize=8, loc='left')

    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(args.out / f'fig_teaser.{ext}', dpi=300)
    print(json.dumps({'starts': len(starts), 'reads': len(reads), 'stops': len(stops)}))


if __name__ == '__main__':
    main()
