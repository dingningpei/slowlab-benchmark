#!/usr/bin/env python3
"""Figure 1: real campaign timelines from public transcripts (no scores).

Selection rule (fixed before plotting): test site 0, the Full branch of repeat 0 of each language model,
and the main baseline's seed-0 Full campaign. Bars are crops (start to close), ticks are channel reads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

CROP, CLEAN = 180, 2
ROWS = [('deepseek-v4.1-flash', 'DeepSeek V4.1 Flash'), ('glm-5.3-flash', 'GLM-5.3 Flash'),
        ('mimo-v2.6-flash', 'MiMo V2.6 Flash'), ('pbo', 'Prior-informed local BO')]


def events(transcript):
    starts, stops, reads = [], [], []
    for e in transcript:
        if not e.get('ok', True):
            continue
        req, res = e['request'], e.get('result') or {}
        day = (res.get('clock') or 0) / 86400
        if req['action'] == 'start' and 'run_index' in res:
            starts.append((day, req['unit']))
        elif req['action'] == 'stop':
            stops.append((day, req['unit']))
        elif req['action'] == 'observe' and 'variable' in req:
            reads.append((day, req['unit']))
    return starts, stops, reads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--campaigns', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    fig, axes = plt.subplots(len(ROWS), 1, figsize=(8, 5.2), sharex=True)
    for ax, (name, label) in zip(axes, ROWS):
        if name == 'pbo':
            tr = json.loads((args.campaigns / 'site00_pbo_s0_full.json').read_text())['public_transcript']
        else:
            tr = json.loads((args.campaigns / f'site00_{name}_r0.json').read_text())['branches']['full']['public_transcript']
        starts, stops, reads = events(tr)
        for day, unit in starts:
            end = min([d for d, u in stops if u == unit and d > day] + [day + CROP, 365])
            ax.barh(unit, end - day, left=day, height=0.6, color='#88a', edgecolor='black', lw=0.4)
            ax.barh(unit, min(CLEAN, 365 - end), left=end, height=0.6, color='#ddd')
        for day, unit in reads:
            ax.plot([day, day], [unit - 0.35, unit + 0.35], color='#c33', lw=0.8)
        ax.set_yticks(range(4))
        ax.set_yticklabels([f'C{u + 1}' for u in range(4)], fontsize=7)
        ax.set_ylim(-0.6, 3.6)
        ax.invert_yaxis()
        ax.set_title(f'{label}: {len(starts)} crops, {len(reads)} channel reads', fontsize=8, loc='left')
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel('Campaign day (day 0 = 1 January)', fontsize=8)
    axes[-1].set_xlim(0, 365)
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(args.out / f'fig1_timelines.{ext}', dpi=200)


if __name__ == '__main__':
    main()
