#!/usr/bin/env python3
"""Fit the frozen GP hyperparameters of the prior-informed local BO on development-site data.

Data: the process-predictor training runs (development sites 100-139, development years; random
feasible policies; ``slowlab/predictor_data.py``). One group is one run (one site, one year), as
within a campaign; each crop that completed normally gives one observation: scaled policy and
planting-season inputs -> contribution margin under that site's public prices. Length scales and
the noise ratio are shared by all groups; each group has its own constant mean and scale.

Both kernel forms (``product`` and ``season_interaction``, see slowlab.prior_bo.prior_kernel) are
fitted. Held-out checks (5 folds over groups; hyperparameters fitted on the other folds):

* random: predict each crop of a held-out group from 4 other crops of that group (campaign-sized
  data), also with the per-campaign refit GP of gp-bo-v3 and the mean of the 4 crops;
* cross_season: in held-out two-wave groups (4 crops planted on day 0, 4 on day 182), train on the
  day-0 crops plus one day-182 crop and predict the other day-182 crops; this is the use in a
  campaign, where the second-wave choice and the recommendation rest on first-season crops.
  Reported as RMSE and as pairwise ranking accuracy among the predicted crops.

The kernel form is chosen by the higher cross-season ranking accuracy (ties: lower random RMSE).
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


def concordance(pred, true):
    pairs = [(i, j) for i in range(len(true)) for j in range(i + 1, len(true)) if true[i] != true[j]]
    return [float((pred[i] - pred[j]) * (true[i] - true[j]) > 0) for i, j in pairs]


def held_out_check(groups, folds, rng, forms, train_size=4, restarts=8):
    order = rng.permutation(len(groups))
    random = {name: [] for name in [*forms, 'refit_gp_bo_v3', 'mean_of_training']}
    cross = {form: {'errors': [], 'concordant': []} for form in forms}
    for k in range(folds):
        test = set(order[k::folds].tolist())
        train = [(x, y) for i, (x, y, _) in enumerate(groups) if i not in test]
        hyper = {form: fit_shared_hyperparameters(train, rng, form=form) for form in forms}
        for i in sorted(test):
            x, y, _ = groups[i]
            for j in range(len(y)):
                others = [m for m in range(len(y)) if m != j]
                if len(others) < train_size:
                    continue
                pick = rng.choice(others, size=train_size, replace=False)
                for form in forms:
                    gp = FrozenGP(x[pick], y[pick], hyper[form]['kernel'], hyper[form]['noise_ratio'])
                    random[form].append(float(gp.predict(x[j:j + 1])[0][0] - y[j]))
                refit = OutcomeModel(x[pick][:, :6], y[pick], rng, restarts)
                random['refit_gp_bo_v3'].append(float(refit.predict(x[j:j + 1, :6])[0][0] - y[j]))
                random['mean_of_training'].append(float(y[pick].mean() - y[j]))
            first = [m for m in range(len(y)) if x[m, 7] > 0.99]   # cos input 1: planted on day 0
            second = [m for m in range(len(y)) if x[m, 7] < 0.01]  # cos input near 0: planted on day 182
            if len(first) >= 3 and len(second) >= 3:
                anchor = int(rng.choice(second))
                rest = [m for m in second if m != anchor]
                train_idx = first + [anchor]
                for form in forms:
                    gp = FrozenGP(x[train_idx], y[train_idx], hyper[form]['kernel'], hyper[form]['noise_ratio'])
                    pred = gp.predict(x[rest])[0]
                    cross[form]['errors'].extend((pred - y[rest]).tolist())
                    cross[form]['concordant'].extend(concordance(pred, y[rest]))
    out = {'random': {name: {'rmse': float(np.sqrt(np.mean(np.square(e)))), 'n': len(e)} for name, e in random.items()},
           'cross_season': {form: {'rmse': float(np.sqrt(np.mean(np.square(v['errors'])))),
                                   'ranking_accuracy': float(np.mean(v['concordant'])), 'pairs': len(v['concordant'])}
                            for form, v in cross.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20261003)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    groups, data_sha, files = load_groups(args.data_dir)
    rng = np.random.default_rng(args.seed)
    forms = ('product', 'season_interaction')
    fits = {form: fit_shared_hyperparameters([(x, y) for x, y, _ in groups], rng, form=form) for form in forms}
    check = held_out_check(groups, args.folds, rng, forms)
    cs = check['cross_season']
    chosen = max(forms, key=lambda f: (cs[f]['ranking_accuracy'], -check['random'][f]['rmse']))
    hyper = fits[chosen]
    contract = json.loads((ROOT / FIELDS_CONTRACT).read_text())
    out = {'inputs': list(contract['policy']['fields']) + ['planting_season_sin', 'planting_season_cos'],
           'input_scaling': f'policy fields scaled to [0, 1] by the bounds of {FIELDS_CONTRACT}; season as in slowlab.tools',
           'selection_rule': 'higher cross-season ranking accuracy; ties by lower random RMSE',
           'chosen_form': chosen, 'fits': fits, 'hyperparameters': hyper, 'data': {'files': files, 'groups': len(groups),
                                              'data_files_sha256': data_sha, 'group_ids': [g for _, _, g in groups]},
           'held_out_check': check, 'seed': args.seed}
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'chosen_form': chosen, 'fits': fits, 'groups': len(groups), 'held_out_check': check}, indent=1))


if __name__ == '__main__':
    main()
