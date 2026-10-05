#!/usr/bin/env python3
"""Figure 4 (E3): Reader swap gain by Reader and source method, from the locked Phase 6 analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures_e1e2 import COLOR, LABEL  # noqa: E402

READERS = [('gp-reader', 'GP Reader'), ('deepseek-v4.1-flash', 'DeepSeek Reader'), ('glm-5.3-flash', 'GLM Reader'),
           ('mimo-v2.6-flash', 'MiMo Reader')]
SOURCES = ['deepseek-v4.1-flash', 'glm-5.3-flash', 'mimo-v2.6-flash', 'pbo']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    e3 = json.loads(args.report.read_text())['e3']
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for i, (reader, label) in enumerate(READERS):
        y0 = len(READERS) - 1 - i
        pooled = e3['reader_swap_gain_pooled'][reader]
        ax.errorbar(pooled['mean'], y0 + 0.3, xerr=[[pooled['mean'] - pooled['ci95_bca'][0]], [pooled['ci95_bca'][1] - pooled['mean']]],
                    fmt='D', color='black', ms=5, capsize=3, lw=1.2)
        p = e3['secondary_bh_with_e3'][f'E3_reader_swap_{reader}']
        ax.text(1.01, y0 + 0.3, f"{pooled['mean']:+.2f}  p$_{{BH}}$={p:.2g}", transform=ax.get_yaxis_transform(), va='center', fontsize=7)
        for j, src in enumerate(SOURCES):
            v = e3['reader_swap_gain'][reader].get(src)
            if not v or not v.get('n'):
                continue
            y = y0 + 0.12 - 0.12 * j
            ax.errorbar(v['mean'], y, xerr=[[v['mean'] - v['ci95_bca'][0]], [v['ci95_bca'][1] - v['mean']]], fmt='o', ms=4,
                        color=COLOR[src], lw=1, capsize=0, label=LABEL[src] if i == 0 else None)
    ax.axvline(0, color='grey', lw=0.8)
    ax.set_yticks([len(READERS) - 1 - i for i in range(len(READERS))])
    ax.set_yticklabels([label for _, label in READERS], fontsize=8)
    ax.set_xlabel('Reader score minus the source method\'s own recommendation (EUR/m²)', fontsize=8)
    ax.set_title('E3: one-shot Readers on the same history packets (diamond: pooled over sites)', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(title='Packet source', fontsize=7, title_fontsize=7, frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, -0.18), ncol=4)
    fig.subplots_adjust(left=0.17, right=0.8, bottom=0.27, top=0.9)
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(args.out / f'fig4_e3.{ext}', dpi=200)


if __name__ == '__main__':
    main()
