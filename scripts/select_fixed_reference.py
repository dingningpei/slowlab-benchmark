#!/usr/bin/env python3
"""Apply the frozen selection rule to the fixed-reference candidates' development evaluations."""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    cand = json.loads((ROOT / 'configs/fixed_reference_candidates_v0.json').read_text())
    by = defaultdict(lambda: defaultdict(list)); energy = defaultdict(list); failed = []
    for f in glob.glob(str(args.out_dir / 'site*.json')):
        m = re.search(r'site(\d+)_(\w+?)_(\d{4})\.json$', f)
        r = json.loads(Path(f).read_text())
        if r.get('status') != 'completed':
            failed.append(Path(f).name); continue
        by[m[2]][int(m[1])].append(r['score_eur_m2'])
        for p in r['plantings']:
            for c in p['crops']:
                energy[m[2]].append(c['accrued']['heat_kwh_m2'] + c['accrued']['light_kwh_m2'])
    summary = {}
    for name in cand['candidates']:
        site_means = [float(np.mean(v)) for v in by[name].values()]
        summary[name] = {'sites': len(site_means), 'mean_score_eur_m2': float(np.mean(site_means)) if site_means else None,
                         'site_means': {str(s): float(np.mean(v)) for s, v in sorted(by[name].items())},
                         'mean_energy_kwh_m2_per_crop': float(np.mean(energy[name])) if energy[name] else None}
    ok = [n for n, v in summary.items() if v['mean_score_eur_m2'] is not None]
    winner = max(ok, key=lambda n: (round(summary[n]['mean_score_eur_m2'], 9), -summary[n]['mean_energy_kwh_m2_per_crop']))
    report = {'rule': cand['selection_rule'], 'candidates': summary, 'failed_runs': failed, 'winner': winner,
              'winner_policy': cand['candidates'][winner]}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'winner': winner, 'means': {n: round(v['mean_score_eur_m2'], 3) for n, v in summary.items() if v['mean_score_eur_m2'] is not None}, 'failed': len(failed)}))


if __name__ == '__main__':
    main()
