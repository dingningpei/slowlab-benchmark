import json
from pathlib import Path

import numpy as np
import pytest

from slowlab.process_predictor import (FEATURE_SETS, TARGETS, QuadraticRidge, examples, feature_row, group_folds,
                                       quadratic)

ROOT = Path(__file__).resolve().parents[1]
FIELDS = json.loads((ROOT / 'configs/task_contract_v5.json').read_text())['policy']['fields']
POLICY = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())['policy_a']


def record(site, start=0.0, harvest0=0.0):
    days = [float(d) for d in range(181)]
    lin = lambda total, base=0.0: [base + total * d / 180 for d in days]
    crop = {'policy': POLICY, 'planting_day': start, 'snapshot_days': days, 'final_reason': 'normal_completion',
            'snapshots': {'cumulative_harvest_fresh_equivalent': lin(30.0, harvest0), 'heating_energy': lin(100.0),
                          'lighting_energy': lin(50.0), 'co2_dosed': lin(10.0), 'canopy_lai_proxy': [2.5] * 181},
            'final_accrued': {'harvest_kg_m2': 30.0, 'heat_kwh_m2': 100.0, 'light_kwh_m2': 50.0, 'co2_kg_m2': 10.0,
                              'days': 180.0},
            'final_event_cost_eur_m2': 2.5}
    return {'identity': {'site_index': site}, 'site_prices': None, 'crops': [crop]}


def test_feature_rows_have_the_declared_width():
    so_far = dict.fromkeys(TARGETS, 1.0)
    for name, extra in FEATURE_SETS.items():
        row = feature_row(FIELDS, POLICY, 182, 30, so_far, 2.0, name)
        assert len(row) == len(FIELDS) + 2 + 1 + len(extra)


def test_examples_use_increase_since_planting_and_remaining_targets():
    x, y, info = examples([record(1, start=182.0, harvest0=12.0)], FIELDS, [60], 'all')
    assert info[0]['so_far']['harvest_kg_m2'] == pytest.approx(10.0)
    assert y[0].tolist() == pytest.approx([20.0, 100 * 2 / 3, 50 * 2 / 3, 10 * 2 / 3])
    assert x[0][-5] == pytest.approx(10.0 / 60) and x[0][-1] == 2.5


def test_ridge_fits_a_quadratic_and_round_trips():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(300, 3))
    y = np.stack([1 + x[:, 0] * x[:, 1] - 2 * x[:, 2] ** 2, x[:, 0]], axis=1)
    m = QuadraticRidge(1e-6).fit(x, y)
    assert np.abs(m.predict(x) - y).max() < 1e-6
    assert np.allclose(QuadraticRidge.from_dict(json.loads(json.dumps(m.to_dict()))).predict(x), m.predict(x))
    assert quadratic(x).shape == (300, 1 + 3 + 6)


def test_site_folds_partition_sites():
    folds = group_folds([3, 1, 2, 3, 5, 8, 1], 3)
    assert set().union(*folds) == {1, 2, 3, 5, 8} and sum(len(f) for f in folds) == 5


def test_unread_resources_are_whole_crop_targets():
    x, y, info = examples([record(1)], FIELDS, [60], 'harvest_lai')
    assert info[0]['read'] == {'harvest_kg_m2'}
    assert y[0].tolist() == pytest.approx([20.0, 100.0, 50.0, 10.0])
