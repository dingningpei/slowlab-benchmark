#!/usr/bin/env python3
"""Fit the frozen GP hyperparameters of the prior-informed local BO on development-site data.

Data: the process-predictor training runs (development sites 100-139, development years; random
feasible policies; ``slowlab/predictor_data.py``). One group is one run (one site, one year), as
within a campaign; each crop that completed normally gives one observation: scaled policy and
planting-season inputs -> contribution margin under that site's public prices. Length scales and
the noise ratio are shared by all groups; each group has its own constant mean and scale.

Also reports a held-out check (5 folds over groups): for crops of held-out groups, predict each
crop from 4 other crops of its group (campaign-sized data) with (a) the frozen hyperparameters fitted
on the other folds, (b) the per-campaign refit GP of gp-bo-v3, (c) the mean of the 4 crops.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.prior_bo import FrozenGP, fit_shared_hyperparameters  # noqa: E402
from slowlab.tools import OutcomeModel, contribution_margin, season_features  # noqa: E402

FIELDS_CONTRACT = 'configs/task_contract_v8.json'


def load_groups(data_dir: Path):
    contract = json.loads((ROOT / FIELDS_CONTRACT).read_text())
    fields, econ = contract['policy']['fields'], contract['economics']
    groups, digest, files = [], hashlib.sha256(), sorted(glob.glob(str(data_dir / '*.json')))
    for f in files:
        raw = Path(f).read_bytes()
        record = json.loads(raw)
        if record.get('status') != 'completed' or record.get('format') != 'slowlab-predictor-data-v1':
            continue
        digest.update(raw)
        prices = record['site_prices']
        economics = {'fruit_price_eur_per_kg_fresh': prices['price_eur_per_kg_fresh_equivalent'],
                     'harvest_handling_eur_per_kg': econ['harvest_handling_eur_per_kg'],
                     'delivered_heat_eur_per_kwh': prices['delivered_heat_eur_per_kwh'],
                     'electricity_eur_per_kwh': prices['electricity_eur_per_kwh'],
                     'co2_eur_per_kg': prices['co2_eur_per_kg'],
                     'background_service_eur_per_m2_day': econ['background_service_eur_per_m2_day']}
        xs, ys = [], []
        for crop in record['crops']:
            if crop.get('final_reason') != 'normal_completion':
                continue
            p = crop['policy']
            xs.append([(p[k] - s['min']) / (s['max'] - s['min']) for k, s in fields.items()]
                      + season_features(crop['planting_day']))
            ys.append(contribution_margin(crop['final_accrued'], crop['final_event_cost_eur_m2'], economics))
        if len(ys) >= 3:
            groups.append((np.array(xs), np.array(ys), Path(f).stem))
    return groups, digest.hexdigest(), len(files)


def held_out_check(groups, folds, rng, train_size=4, restarts=8):
    order = rng.permutation(len(groups))
    errors = {'frozen': [], 'refit': [], 'mean_of_training': []}
    for k in range(folds):
        test = set(order[k::folds].tolist())
        hyper = fit_shared_hyperparameters([(x, y) for i, (x, y, _) in enumerate(groups) if i not in test], rng)
        for i in sorted(test):
            x, y, _ = groups[i]
            for j in range(len(y)):
                others = [m for m in range(len(y)) if m != j]
                if len(others) < train_size:
                    continue
                pick = rng.choice(others, size=train_size, replace=False)
                frozen = FrozenGP(x[pick], y[pick], hyper['lengthscales'], hyper['noise_ratio'])
                errors['frozen'].append(float(frozen.predict(x[j:j + 1])[0][0] - y[j]))
                refit = OutcomeModel(x[pick][:, :6], y[pick], rng, restarts)
                errors['refit'].append(float(refit.predict(x[j:j + 1, :6])[0][0] - y[j]))
                errors['mean_of_training'].append(float(y[pick].mean() - y[j]))
    return {name: {'rmse': float(np.sqrt(np.mean(np.square(e)))), 'mae': float(np.mean(np.abs(e))), 'n': len(e)}
            for name, e in errors.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20261003)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    groups, data_sha, files = load_groups(args.data_dir)
    rng = np.random.default_rng(args.seed)
    hyper = fit_shared_hyperparameters([(x, y) for x, y, _ in groups], rng)
    check = held_out_check(groups, args.folds, rng)
    contract = json.loads((ROOT / FIELDS_CONTRACT).read_text())
    out = {'inputs': list(contract['policy']['fields']) + ['planting_season_sin', 'planting_season_cos'],
           'input_scaling': f'policy fields scaled to [0, 1] by the bounds of {FIELDS_CONTRACT}; season as in slowlab.tools',
           'hyperparameters': hyper, 'data': {'files': files, 'groups': len(groups),
                                              'data_files_sha256': data_sha, 'group_ids': [g for _, _, g in groups]},
           'held_out_check': check, 'seed': args.seed}
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'hyperparameters': hyper, 'groups': len(groups), 'held_out_check': check}, indent=1))


if __name__ == '__main__':
    main()
