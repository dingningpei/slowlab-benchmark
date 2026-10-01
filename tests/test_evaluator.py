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


def test_batch_runner_runs_and_resumes(tmp_path):
    (tmp_path / 'c.json').write_text(json.dumps(small_contract()))
    out = tmp_path / 'out'
    jobs = [{'id': f'j{i}', 'year': 2004 + i, 'policy': POLICIES['policy_a' if i else 'policy_b']} for i in range(3)]
    jobs.append({'id': 'bad', 'year': 2004, 'policy': dict(POLICIES['policy_a'], night_temperature_c=99)})
    batch = {'out_dir': str(out), 'common': {'backend': 'fake', 'contract': str(tmp_path / 'c.json')}, 'jobs': jobs}
    (tmp_path / 'batch.json').write_text(json.dumps(batch))
    cmd = [sys.executable, str(ROOT / 'scripts/run_evaluation_batch.py'), str(tmp_path / 'batch.json'), '--workers', '2']
    first = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    assert json.loads((out / 'DONE').read_text()) == {'j0': 'completed', 'j1': 'completed', 'j2': 'completed', 'bad': 'failed'}
    second = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    assert all(json.loads(line)['seconds'] == 0.0 for line in second.splitlines())
    assert first != second


def test_weather_and_campaign_specs_follow_the_site(tmp_path):
    from slowlab.private_runs import campaign_spec, weather_spec
    from slowlab.site_distribution import sample_site
    assert weather_spec(2014, 'dev')['kind'] == 'development_expanded'
    assert weather_spec(2018, 'dev')['plan'] == 'configs/weather_gapfilled_plan.json'
    formal = weather_spec(2009, 'dev', 'formal')
    assert formal['kind'] == 'formal' and formal['year'] == 2009 and formal['dev_cache'] == 'dev'
    with pytest.raises(ValueError):
        weather_spec(2003, 'dev')
    site = {'distribution': 'configs/site_distribution_v1.json', 'master_seed': 5, 'site_index': 2}
    spec = campaign_spec(site=site, year=2009, contract='configs/task_contract_v8.json', feedback_mode='full',
                         fallback_policy=POLICIES['policy_a'], private_dir=tmp_path, dev_cache='dev', formal_cache='formal')
    sampled = sample_site(json.loads((ROOT / 'configs/site_distribution_v1.json').read_text()), 5, 2)
    assert spec['origin_utc'] == '2008-12-31T23:00:00+00:00' and spec['site'] == sampled['site']
    assert spec['sensor_noise']['seed'] == sampled['sensor_noise_seed'] and spec['weather']['year'] == 2009


def test_two_policies_share_one_run_and_each_gets_both_plantings():
    from slowlab.evaluator import evaluate_policies_year
    r = evaluate_policies_year(small_contract(), [POLICIES['policy_a'], POLICIES['policy_b']], site=None,
                               unit_parameters=None, weather=None, source=None, year=2017, executor_kwargs=FAKE)
    single = {name: evaluate_year(small_contract(), POLICIES[name], site=None, unit_parameters=None, weather=None,
                                  source=None, year=2017, executor_kwargs=FAKE)['score_eur_m2']
              for name in ('policy_a', 'policy_b')}
    assert [x['policy'] for x in r['results']] == [POLICIES['policy_a'], POLICIES['policy_b']]
    for x, name in zip(r['results'], ('policy_a', 'policy_b')):
        assert len(x['by_planting_eur_m2']) == 2 and x['score_eur_m2'] == pytest.approx(single[name])
    with pytest.raises(ValueError):
        evaluate_policies_year(small_contract(), [POLICIES['policy_a']] * 3, site=None, unit_parameters=None,
                               weather=None, source=None, year=2017, executor_kwargs=FAKE)


def test_reference_search_runs_end_to_end_on_the_toy_backend(tmp_path):
    contract = small_contract()
    contract['evaluation']['deployment']['plantings_calendar_day'] = [0, 13 * TICK]
    (tmp_path / 'c.json').write_text(json.dumps(contract))
    site = json.dumps({'distribution': 'configs/site_distribution_v1.json', 'master_seed': 20260930, 'site_index': 1})
    subprocess.run([sys.executable, str(ROOT / 'scripts/search_reference_policy.py'), '--site', site,
                    '--years', '2017,2018', '--weather', '{"2017": null, "2018": null}', '--contract',
                    str(tmp_path / 'c.json'), '--backend', 'fake', '--initial', '4', '--rounds', '1', '--batch', '2',
                    '--final-top', '2', '--workers', '4', '--run-dir', str(tmp_path / 'runs'), '--out',
                    str(tmp_path / 'search.json')], check=True, capture_output=True, cwd=ROOT)
    out = json.loads((tmp_path / 'search.json').read_text())
    assert len(out['candidates']) == 6 and out['evaluation_runs'] == 3 + 4
    best = out['best']
    assert best['final_mean'] == max(f['final_mean'] for f in out['finalists'])
    assert set(best['final_by_year']) == {'2017', '2018'}
    searched = {json.dumps(c['policy'], sort_keys=True) for c in out['candidates']}
    assert len(searched) == 6
