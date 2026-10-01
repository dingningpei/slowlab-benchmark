"""Process predictor: in-season public readings -> a crop's final totals.

Inputs available to an agent at day ``d`` after planting: the policy, the
planting day, and the increase since planting of the cumulative public
channels (harvest, heating, lighting, CO2) plus the current canopy proxy.
For channels whose readings are used, targets are the remaining amounts to
the end of the crop (predicted final = so far + remaining); for channels not
read, targets are whole-crop totals. The margin follows from the site's public
prices. Feature construction here is shared by
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


def read_targets(feature_set) -> set:
    """Targets whose so-far amount the feature set reads; the others are predicted as whole-crop totals."""
    return {t for t in TARGETS if t in FEATURE_SETS[feature_set]}


def examples(records, fields, days, feature_set, crop_days=180.0):
    """Rows (x, targets, crop info) from predictor-data records.

    For a target whose so-far amount is read, the target is the remaining amount
    to the end of the crop; otherwise it is the whole-crop total (the agent has
    not read how much was used so far).
    """
    read = read_targets(feature_set)
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
                ys.append([final[t] - so_far[t] if t in read else final[t] for t in TARGETS])
                info.append({'record': rec_index, 'site': rec['identity']['site_index'], 'day': d, 'read': read,
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


# ── Public tool: prediction from the readings already in a transcript ──────
PREDICTOR_PATH = 'configs/process_predictor_v0.json'
MIN_DAY = 10.0
SAME_TIME_SECONDS = 86400.0


class PredictorInputError(ValueError):
    """The transcript lacks the readings the predictor needs; the message says which."""


def load_predictor(path) -> dict:
    import json
    from pathlib import Path
    config = json.loads(Path(path).read_text())
    config['_models'] = {k: QuadraticRidge.from_dict(v) for k, v in config['models'].items()}
    return config


def _typical_error(config, feature_set, day):
    table = config['cv_rmse_by_day'][feature_set]
    days = sorted(float(d) for d in table)
    x = min(max(day, days[0]), days[-1])
    lo = max(d for d in days if d <= x)
    hi = min(d for d in days if d >= x)
    a, b = table[f'{lo:g}'], table[f'{hi:g}']
    w = 0.0 if hi == lo else (x - lo) / (hi - lo)
    return {k: a[k] + w * (b[k] - a[k]) for k in a}


def predict_from_history(task: dict, history, unit: int, run_index: int, config: dict) -> dict:
    """Predicted final totals and margin of one running crop from readings in ``history``.

    Uses cumulative harvest and the canopy proxy (required) and heating,
    lighting and CO2 (all three, if present) read at the same time. Increases
    are measured from a reading at or just before planting; a crop planted at
    the campaign start begins from zero.
    """
    from .tools import contribution_margin
    run = history.runs.get((unit, run_index))
    if run is None or run.get('policy') is None or run.get('start_day') is None:
        raise PredictorInputError('no crop with this compartment and crop number')
    if run['status'] != 'running':
        raise PredictorInputError('this crop has closed; its released totals are in runs')
    crop_days = float(task['calendar']['crop_days'])
    t0 = run['start_day'] * 86400.0

    def series(channel):
        return history.records.get((unit, channel), {})

    required = (CUMULATIVE['harvest_kg_m2'], LAI)
    times = [t for channel in required for t in series(channel) if t0 < t < t0 + crop_days * 86400.0]
    if not times:
        raise PredictorInputError('observe cumulative_harvest_fresh_equivalent and canopy_lai_proxy for this crop first')
    now = max(times)
    day = (now - t0) / 86400.0
    if day < MIN_DAY:
        raise PredictorInputError(f'readings must be at least {MIN_DAY:g} days after planting')
    if day >= crop_days:
        raise PredictorInputError('the latest reading is not inside the running crop')

    def reading_at(channel, required):
        values = {t: v for t, v in series(channel).items() if t0 < t <= now + SAME_TIME_SECONDS}
        near = [t for t in values if abs(t - now) <= SAME_TIME_SECONDS]
        if not near:
            if required:
                raise PredictorInputError(f'observe {channel} within one day of your latest reading of this crop')
            return None
        return values[max(near, key=lambda t: (-abs(t - now), t))]

    def baseline(channel):
        before = {t: v for t, v in series(channel).items() if t0 - 3600.0 <= t <= t0}
        if before:
            return before[max(before)]
        if t0 == 0.0:
            return 0.0
        raise PredictorInputError(f'observe {channel} on the planting day of this crop (needed as its starting value)')

    lai = reading_at(LAI, True)
    so_far = {'harvest_kg_m2': reading_at(CUMULATIVE['harvest_kg_m2'], True) - baseline(CUMULATIVE['harvest_kg_m2'])}
    optional = {t: reading_at(CUMULATIVE[t], False) for t in ('heat_kwh_m2', 'light_kwh_m2', 'co2_kg_m2')}
    feature_set = 'all' if all(v is not None for v in optional.values()) else 'harvest_lai'
    if feature_set == 'all':
        for t, v in optional.items():
            so_far[t] = v - baseline(CUMULATIVE[t])
    fields = task['policy']['fields']
    model = config['_models'][feature_set]
    filled = {t: so_far.get(t, 0.0) for t in TARGETS}
    x = np.array([feature_row(fields, run['policy'], run['start_day'], day, filled, lai, feature_set, crop_days)])
    predicted = np.maximum(model.predict(x)[0], 0.0)
    read = read_targets(feature_set)
    final = {t: (so_far[t] if t in read else 0.0) + float(p) for t, p in zip(TARGETS, predicted)}
    economics = task['economics']
    event_cost = economics['planting_eur_per_m2'] + economics['cleanup_eur_per_m2']
    predicted_margin = contribution_margin({**final, 'days': crop_days}, event_cost, economics)
    return {'unit': unit, 'run_index': run_index, 'reading_day_since_planting': day, 'inputs_used': feature_set,
            'predicted_final_per_m2': final, 'predicted_contribution_margin_eur_m2': predicted_margin,
            'typical_error': _typical_error(config, feature_set, day),
            'note': 'model fitted on separate development greenhouses; typical_error is its cross-validated root '
                    'mean square error at this crop age'}
