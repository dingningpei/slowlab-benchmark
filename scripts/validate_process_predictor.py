#!/usr/bin/env python3
"""Site-grouped cross-validation of process-predictor candidates.

Reads predictor-data records, builds examples at fixed checkpoint days, and
for each fold of held-out sites fits each candidate on the other sites. Error
is measured on the predicted final contribution margin at the held-out site's
public prices (and on each physical total). Candidates: naive linear
extrapolation, prior only (policy and planting day), quadratic ridge on
harvest+canopy readings and on all readings, and a per-checkpoint Gaussian
process on all readings.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.agent_protocol import public_task_view  # noqa: E402
from slowlab.process_predictor import (TARGETS, QuadraticRidge, examples, group_folds, margin)  # noqa: E402
from slowlab.site_parameters import site_contract  # noqa: E402


def economics_for(contract, prices, cache={}):
    key = json.dumps(prices, sort_keys=True)
    if key not in cache:
        cache[key] = public_task_view(site_contract(contract, {'prices': prices} if prices else {}), 'full')['economics']
    return cache[key]


def margins(pred_remaining, info, contract):
    out = []
    for p, i in zip(pred_remaining, info):
        totals = {t: (i['so_far'][t] if t in i['read'] else 0.0) + max(float(v), 0.0) for t, v in zip(TARGETS, p)}
        out.append(margin(totals, i['event_cost'], economics_for(contract, i['prices'])))
    return np.array(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='glob of predictor-data records')
    parser.add_argument('--days', default='15,30,45,60,90,120,150')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--penalties', default='0.1,1,10,100')
    parser.add_argument('--gp', action='store_true', help='also fit per-checkpoint Gaussian processes (slow)')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((ROOT / 'configs/task_contract_v6.json').read_text())
    fields = contract['policy']['fields']
    records = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(args.data))]
    records = [r for r in records if r.get('status') == 'completed']
    days = [int(d) for d in args.days.split(',')]
    penalties = [float(p) for p in args.penalties.split(',')]
    sets = {fs: examples(records, fields, days, fs) for fs in ('prior', 'harvest_lai', 'all')}
    _, y_all, info = sets['all']
    sites = [i['site'] for i in info]
    true_margin = np.array([margin(i['final'], i['event_cost'], economics_for(contract, i['prices'])) for i in info])
    folds = group_folds(sites, args.folds)
    preds = {}
    # naive: scale the so-far totals to the full crop
    naive = np.array([[i['so_far'][t] * (180.0 / i['day'] - 1.0) for t in TARGETS] for i in info])  # 'all' reads every total
    preds['naive_extrapolation'] = naive
    for fs in ('prior', 'harvest_lai', 'all'):
        x, y, _ = sets[fs]
        for pen in penalties:
            p = np.zeros_like(y)
            for test in folds:
                te = np.array([s in test for s in sites])
                p[te] = QuadraticRidge(pen).fit(x[~te], y[~te]).predict(x[te])
            preds[f'ridge_{fs}_pen{pen:g}'] = p
    if args.gp:
        from slowlab.tools import OutcomeModel
        x, y, _ = sets['all']
        p = np.zeros_like(y)
        day_of = np.array([i['day'] for i in info])
        for test in folds:
            te = np.array([s in test for s in sites])
            for d in days:
                tr_d, te_d = (~te) & (day_of == d), te & (day_of == d)
                lo, hi = x[tr_d].min(0), x[tr_d].max(0)
                scale = lambda a: (a - lo) / np.where(hi > lo, hi - lo, 1.0)
                for j in range(len(TARGETS)):
                    model = OutcomeModel(scale(x[tr_d]), y[tr_d, j], np.random.default_rng([d, j]), 2)
                    p[te_d, j] = model.predict(scale(x[te_d]))[0]
        preds['gp_all_per_day'] = p
    report = {'records': len(records), 'crops': sum(len(r['crops']) for r in records), 'sites': len(set(sites)),
              'days': days, 'folds': args.folds, 'true_margin_sd_by_day': {}, 'candidates': {}}
    day_of = np.array([i['day'] for i in info])
    for d in days:
        report['true_margin_sd_by_day'][str(d)] = float(true_margin[day_of == d].std())
    for name, p in preds.items():
        m = margins(p, info, contract)
        err = m - true_margin
        entry = {'margin_rmse_by_day': {str(d): float(np.sqrt(np.mean(err[day_of == d] ** 2))) for d in days},
                 'margin_rmse': float(np.sqrt(np.mean(err ** 2))),
                 'total_rmse_by_target': {t: float(np.sqrt(np.mean((p[:, j] - y_all[:, j]) ** 2)))
                                          for j, t in enumerate(TARGETS)}}
        report['candidates'][name] = entry
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    best = sorted(report['candidates'].items(), key=lambda kv: kv[1]['margin_rmse'])
    print('true margin SD by day', {k: round(v, 2) for k, v in report['true_margin_sd_by_day'].items()})
    for name, e in best[:8] + [kv for kv in best if kv[0] == 'naive_extrapolation']:
        print(f"{name:32s} {e['margin_rmse']:7.2f}  " + ' '.join(f"{v:6.2f}" for v in e['margin_rmse_by_day'].values()))


if __name__ == '__main__':
    main()
