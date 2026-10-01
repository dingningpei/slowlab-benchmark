#!/usr/bin/env python3
"""Fit the frozen process predictor (decision 2026-10-01: simple quadratic ridge).

Fits one model per feature set ('harvest_lai', 'all') on every example day in
DAYS from all completed predictor-data records, and measures site-grouped
cross-validated error per day (margin at each held-out site's public prices,
and each physical total). The output config holds only normalisation and
coefficients plus provenance; no site values or records.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from slowlab.process_predictor import (FEATURE_SETS, TARGETS, QuadraticRidge, examples, group_folds,  # noqa: E402
                                       margin)
from validate_process_predictor import economics_for, margins  # noqa: E402

DAYS = list(range(10, 180, 5))
PENALTY = 10.0
SETS = ('harvest_lai', 'all')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    contract_path = ROOT / 'configs/task_contract_v6.json'
    contract = json.loads(contract_path.read_text())
    fields = contract['policy']['fields']
    files = sorted(glob.glob(args.data))
    records = [json.loads(Path(f).read_text()) for f in files]
    records = [r for r in records if r.get('status') == 'completed']
    out = {'predictor_id': 'slowlab-process-predictor-v0', 'decision': 'RESEARCH_PLAN decision 2026-10-01',
           'model': f'quadratic ridge, penalty {PENALTY:g}, on standardized inputs; targets are remaining totals',
           'feature_sets': {k: list(v) for k, v in FEATURE_SETS.items() if k in SETS},
           'feature_order': 'scaled policy fields (contract order), planting-season sin/cos, day/crop_days, then the '
                            'feature-set readings: increase since planting per day for cumulative channels, current value '
                            'for the canopy proxy',
           'targets': list(TARGETS), 'days_trained': DAYS,
           'training': {'records': len(records), 'crops': sum(len(r['crops']) for r in records),
                        'sites': sorted({r['identity']['site_index'] for r in records}),
                        'years': sorted({r['year'] for r in records}),
                        'master_seed': 'development 20260930', 'data_files_sha256': hashlib.sha256(
                            b''.join(hashlib.sha256(Path(f).read_bytes()).digest() for f in files)).hexdigest(),
                        'code_commit': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                                                      capture_output=True, text=True).stdout.strip() or None},
           'contract': contract['contract_id'], 'models': {}, 'cv_rmse_by_day': {}}
    for fs in SETS:
        x, y, info = examples(records, fields, DAYS, fs)
        sites = [i['site'] for i in info]
        day_of = np.array([i['day'] for i in info])
        pred = np.zeros_like(y)
        for test in group_folds(sites, args.folds):
            te = np.array([s in test for s in sites])
            pred[te] = QuadraticRidge(PENALTY).fit(x[~te], y[~te]).predict(x[te])
        true = np.array([margin(i['final'], i['event_cost'], economics_for(contract, i['prices'])) for i in info])
        err = margins(pred, info, contract) - true
        out['cv_rmse_by_day'][fs] = {
            str(d): {'margin_eur_m2': float(np.sqrt(np.mean(err[day_of == d] ** 2))),
                     **{t: float(np.sqrt(np.mean((pred[day_of == d, j] - y[day_of == d, j]) ** 2)))
                        for j, t in enumerate(TARGETS)}} for d in DAYS}
        out['models'][fs] = QuadraticRidge(PENALTY).fit(x, y).to_dict()
        print(fs, 'examples', len(y), 'margin rmse d30/90/150',
              [round(out['cv_rmse_by_day'][fs][str(d)]['margin_eur_m2'], 2) for d in (30, 90, 150)])
    args.out.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
