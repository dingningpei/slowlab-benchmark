import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from slowlab import fake_backend
from slowlab.predictor_data import CHANNELS, collect, layout_plan, random_policy

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'configs/task_contract_v5.json').read_text())
T = 300 / 86400
FAKE = {'lifecycle_factory': fake_backend.FakeLifecycle, 'sample_endpoint': fake_backend.sample_endpoint}


def small():
    c = copy.deepcopy(CONTRACT)
    c['budget'].update(campaign_days=40 * T, crop_days=12 * T, cleanup_days=T, latest_start_day=13 * T)
    return c


def test_random_policies_are_feasible():
    rng = np.random.default_rng(0)
    fields = CONTRACT['policy']['fields']
    for _ in range(200):
        p = random_policy(fields, rng)
        assert p['night_temperature_c'] <= p['day_temperature_c']
        assert all(fields[k]['min'] <= v <= fields[k]['max'] for k, v in p.items())


def test_layouts():
    rng = np.random.default_rng(0)
    assert layout_plan(CONTRACT, 'two_wave', rng) == [(0.0, u) for u in range(4)] + [(182.0, u) for u in range(4)]
    single = layout_plan(CONTRACT, 'single', rng)
    assert sorted(u for _, u in single) == [0, 1, 2, 3] and all(0 <= d <= 183 for d, _ in single)
    assert [d for d, _ in single] == sorted(d for d, _ in single)


def test_snapshots_match_final_totals_for_both_waves():
    r = collect(small(), layout='two_wave', layout_seed=3, year=2017, weather=None, source=None,
                executor_kwargs=FAKE, snapshot_step_days=T)
    assert len(r['crops']) == 8 and all(c['final_reason'] == 'normal_completion' for c in r['crops'])
    for c in r['crops']:
        s = c['snapshots']
        assert set(s) == set(CHANNELS) and len(s['heating_energy']) == 13
        assert s['cumulative_harvest_fresh_equivalent'][-1] - s['cumulative_harvest_fresh_equivalent'][0] == \
            pytest.approx(c['final_accrued']['harvest_kg_m2'], abs=1e-9)
        assert s['co2_dosed'][-1] - s['co2_dosed'][0] == pytest.approx(c['final_accrued']['co2_kg_m2'], abs=1e-9)
    second = [c for c in r['crops'] if c['run_index'] == 2]
    assert second and all(c['snapshots']['heating_energy'][0] > 0 for c in second)


def test_script_runs_a_single_layout(tmp_path):
    (tmp_path / 'c.json').write_text(json.dumps(small()))
    spec = {'backend': 'fake', 'contract': str(tmp_path / 'c.json'), 'year': 2018, 'layout': 'single', 'layout_seed': 5,
            'site': {'distribution': str(ROOT / 'configs/site_distribution_v0.json'), 'master_seed': 1, 'site_index': 100},
            'out': str(tmp_path / 'out.json')}
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    subprocess.run([sys.executable, str(ROOT / 'scripts/generate_predictor_data.py'), str(tmp_path / 'spec.json')],
                   check=True, capture_output=True)
    record = json.loads((tmp_path / 'out.json').read_text())
    assert record['status'] == 'completed' and len(record['crops']) == 4
    assert record['site_prices']['co2_eur_per_kg'] == 0.2
