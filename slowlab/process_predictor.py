"""Process predictor: in-season public readings -> a crop's final totals.

Inputs available to an agent at day ``d`` after planting: the policy, the
planting day, and the increase since planting of the cumulative public
channels (harvest, heating, lighting, CO2) plus the current canopy proxy.
Targets are the remaining totals to the end of the crop, so the predicted
final totals are the readings so far plus the prediction, and the margin
follows from the site's public prices. Feature construction here is shared by
training, validation and the public tool.
"""
from __future__ import annotations

import math

import numpy as np

from .tools import season_features

CUMULATIVE = {'harvest_kg_m2': 'cumulative_harvest_fresh_equivalent', 'heat_kwh_m2': 'heating_energy',
              'light_kwh_m2': 'lighting_energy', 'co2_kg_m2': 'co2_dosed'}
TARGETS = tuple(CUMULATIVE)
LAI = 'canopy_lai_proxy'
FEATURE_SETS = {
    'prior': (),
    'harvest_lai': ('harvest_kg_m2', LAI),
    'all': ('harvest_kg_m2', 'heat_kwh_m2', 'light_kwh_m2', 'co2_kg_m2', LAI),
}


def scaled_policy(fields: dict, policy: dict) -> list[float]:
    return [(policy[k] - s['min']) / (s['max'] - s['min']) for k, s in fields.items()]


def feature_row(fields, policy, planting_day, day, so_far, lai, feature_set, crop_days=180.0):
    """``so_far``: increase since planting per target; ``day``: days since planting (> 0)."""
    row = scaled_policy(fields, policy) + season_features(planting_day) + [day / crop_days]
    for name in FEATURE_SETS[feature_set]:
        row.append(lai if name == LAI else so_far[name] / day)
    return row


def examples(records, fields, days, feature_set, crop_days=180.0):
    """Rows (x, remaining targets, so-far, crop info) from predictor-data records."""
    xs, ys, info = [], [], []
    for rec_index, rec in enumerate(records):
        for crop in rec['crops']:
            if crop['final_reason'] != 'normal_completion':
                continue
            snaps, grid = crop['snapshots'], crop['snapshot_days']
            for d in days:
                k = grid.index(float(d))
                so_far = {t: snaps[c][k] - snaps[c][0] for t, c in CUMULATIVE.items()}
                final = {t: crop['final_accrued'][t] for t in TARGETS}
                xs.append(feature_row(fields, crop['policy'], crop['planting_day'], d, so_far, snaps[LAI][k],
                                      feature_set, crop_days))
                ys.append([final[t] - so_far[t] for t in TARGETS])
                info.append({'record': rec_index, 'site': rec['identity']['site_index'], 'day': d,
                             'so_far': so_far, 'final': final, 'event_cost': crop['final_event_cost_eur_m2'],
                             'prices': rec.get('site_prices')})
    return np.array(xs, dtype=float), np.array(ys, dtype=float), info


def quadratic(x: np.ndarray) -> np.ndarray:
    n, d = x.shape
    cols = [np.ones((n, 1)), x] + [x[:, i:i + 1] * x[:, i:] for i in range(d)]
    return np.hstack(cols)


class QuadraticRidge:
    """Ridge regression on standardized inputs with all pairwise products (numpy only)."""

    def __init__(self, penalty: float = 1.0):
        self.penalty = float(penalty)

    def fit(self, x, y):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-12
        z = quadratic((x - self.mu) / self.sd)
        self.y_mu = y.mean(0)
        reg = self.penalty * np.eye(z.shape[1])
        reg[0, 0] = 0.0
        self.coef = np.linalg.solve(z.T @ z + reg, z.T @ (y - self.y_mu))
        resid = y - self.predict(x)
        self.resid_sd = resid.std(0)
        return self

    def predict(self, x):
        return quadratic((x - self.mu) / self.sd) @ self.coef + self.y_mu

    def to_dict(self):
        return {'kind': 'quadratic_ridge', 'penalty': self.penalty, 'mu': self.mu.tolist(), 'sd': self.sd.tolist(),
                'y_mu': self.y_mu.tolist(), 'coef': self.coef.tolist(), 'resid_sd': self.resid_sd.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d['penalty'])
        m.mu, m.sd, m.y_mu = (np.array(d[k]) for k in ('mu', 'sd', 'y_mu'))
        m.coef, m.resid_sd = np.array(d['coef']), np.array(d['resid_sd'])
        return m


def margin(totals: dict, event_cost: float, economics: dict, crop_days: float = 180.0) -> float:
    from .tools import contribution_margin
    return contribution_margin({**totals, 'days': crop_days}, event_cost, economics)


def group_folds(groups, k, seed=0):
    unique = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return [set(part.tolist()) for part in np.array_split(unique, k)]
