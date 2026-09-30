import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from slowlab import fake_backend
from slowlab.evaluator import evaluate_year, planting_units

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'configs/task_contract_v5.json').read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
TICK = 300 / 86400
FAKE = {'lifecycle_factory': fake_backend.FakeLifecycle, 'sample_endpoint': fake_backend.sample_endpoint}


def small_contract():
    c = copy.deepcopy(CONTRACT)
    c['budget'].update(campaign_days=26 * TICK, crop_days=12 * TICK, cleanup_days=TICK, latest_start_day=13 * TICK)
    c['evaluation']['horizon_days'] = 12 * TICK
    c['evaluation']['deployment']['plantings_calendar_day'] = [0, 13 * TICK]
    return c


def test_planting_units_split_compartments():
    assert planting_units(CONTRACT) == [(0.0, [0, 1]), (182.0, [2, 3])]


def test_evaluation_scores_both_plantings_with_the_agents_margin():
    r = evaluate_year(small_contract(), POLICIES['policy_a'], site=None, unit_parameters=None, weather=None,
                      source=None, year=2017, executor_kwargs=FAKE)
    assert [p['planting_day'] for p in r['plantings']] == [0.0, 13 * TICK]
    assert [[c['unit'] for c in p['crops']] for p in r['plantings']] == [[0, 1], [2, 3]]
    assert r['all_normal_completion']
    means = [p['mean_margin_eur_m2'] for p in r['plantings']]
    assert r['score_eur_m2'] == pytest.approx(sum(means) / 2)
    assert all(c['accrued']['days'] == pytest.approx(12 * TICK) for p in r['plantings'] for c in p['crops'])


def test_site_prices_change_the_score_and_calendar_is_checked():
    base = evaluate_year(small_contract(), POLICIES['policy_b'], site=None, unit_parameters=None, weather=None,
                         source=None, year=2017, executor_kwargs=FAKE)
    dear = evaluate_year(small_contract(), POLICIES['policy_b'], site={'prices': {'delivered_heat_eur_per_kwh': 0.1, 'co2_eur_per_kg': 0.5}},
                         unit_parameters=None, weather=None, source=None, year=2017, executor_kwargs=FAKE)
    assert dear['score_eur_m2'] < base['score_eur_m2']
    late = small_contract()
    late['evaluation']['deployment']['plantings_calendar_day'] = [0, 20 * TICK]
    with pytest.raises(ValueError, match='do not fit'):
        evaluate_year(late, POLICIES['policy_a'], site=None, unit_parameters=None, weather=None, source=None,
                      year=2017, executor_kwargs=FAKE)


def run_script(tmp_path, spec):
    (tmp_path / 'c.json').write_text(json.dumps(small_contract()))
    spec = {'backend': 'fake', 'contract': str(tmp_path / 'c.json'), 'year': 2004, 'out': str(tmp_path / 'out.json'), **spec}
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_recommendation.py'), str(tmp_path / 'spec.json')],
                          capture_output=True, text=True)
    return proc.returncode, json.loads((tmp_path / 'out.json').read_text())


def test_script_evaluates_a_settlement_on_a_sampled_site(tmp_path):
    (tmp_path / 'settlement.json').write_text(json.dumps({'recommendation': POLICIES['policy_b']}))
    site = {'distribution': str(ROOT / 'configs/site_distribution_v0.json'), 'master_seed': 17, 'site_index': 3}
    code, record = run_script(tmp_path, {'settlement': str(tmp_path / 'settlement.json'), 'site': site})
    assert code == 0 and record['status'] == 'completed' and record['policy'] == POLICIES['policy_b']
    from slowlab.site_distribution import evaluation_draws
    dist = json.loads((ROOT / 'configs/site_distribution_v0.json').read_text())
    assert record['unit_parameters'] == evaluation_draws(dist, 17, 3, 2004)['unit_parameters']
    assert record['identity']['site_index'] == 3 and record['peak_rss_mb'] > 0


def test_script_records_failures(tmp_path):
    code, record = run_script(tmp_path, {'policy': dict(POLICIES['policy_a'], night_temperature_c=99)})
    assert code == 3 and record['status'] == 'failed' and 'traceback' in record
