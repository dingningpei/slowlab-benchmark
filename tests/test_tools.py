import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from slowlab.agent_client import CampaignProcess
from slowlab.agent_protocol import canonical, public_task_view
from slowlab.prompt_firewall import assert_outbound_messages_safe, load_blinding_policy
from slowlab.tools import LIMITS, PublicHistory, ToolError, Toolbox, contribution_margin

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'configs/task_contract_v4.json').read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
TASK = public_task_view(CONTRACT, 'full')
TICK_DAY = 300 / 86400
FIELDS = list(TASK['policy']['fields'])


# ── fake-backend campaign through the real process boundary ───────────────

def run_fake_campaign(tmp_path, name, origin='2016-12-31T23:00:00+00:00', soil=None):
    folder = tmp_path / name
    folder.mkdir()
    contract = copy.deepcopy(CONTRACT)
    contract['budget'].update(campaign_days=8 * TICK_DAY, crop_days=3 * TICK_DAY,
                              cleanup_days=TICK_DAY, latest_start_day=2 * TICK_DAY)
    (folder / 'contract.json').write_text(json.dumps(contract))
    spec = {'backend': 'fake', 'contract': str(folder / 'contract.json'), 'feedback_mode': 'full',
            'fallback_policy': POLICIES['policy_a'], 'origin_utc': origin, 'soil_boundary_c': soil,
            'trace': None, 'settlement_out': str(folder / 'settlement.json'), 'failure_out': str(folder / 'failure.json')}
    (folder / 'site.json').write_text(json.dumps(spec))
    campaign = CampaignProcess(folder / 'site.json', private_log=folder / 'server.log')
    s = campaign.session
    policies = [POLICIES['policy_a'], POLICIES['policy_b'],
                dict(POLICIES['policy_a'], day_temperature_c=24), dict(POLICIES['policy_b'], co2_target_ppm=900)]
    for unit, policy in enumerate(policies):
        s.dispatch({'action': 'start', 'unit': unit, 'policy': policy})
    s.dispatch({'action': 'advance', 'day': 1 * TICK_DAY})
    s.dispatch({'action': 'observe', 'unit': 0, 'variable': 'air_temperature_c'})
    s.dispatch({'action': 'advance', 'day': 4 * TICK_DAY})
    for unit in range(4):
        s.dispatch({'action': 'observe', 'unit': unit})
    s.dispatch({'action': 'advance', 'day': 8 * TICK_DAY})
    s.dispatch({'action': 'recommend', 'policy': POLICIES['policy_a']})
    campaign.close()
    return s, json.loads((folder / 'settlement.json').read_text())


def test_runs_table_and_margin_match_the_private_ledger(tmp_path):
    session, settlement = run_fake_campaign(tmp_path, 'a')
    result = Toolbox(session, 7).call('runs')['result']
    assert result['completed_crops'] == 4
    for row in result['runs']:
        assert row['reason'] == 'normal_completion' and row['closed_day'] == pytest.approx(3 * TICK_DAY, rel=1e-5)
        ledger = settlement['ledger_by_unit'][str(row['unit'])]
        active = ledger['by_phase_per_m2']['active']
        expected = contribution_margin(active, ledger['event_cost_eur_m2'], TASK['economics'])
        assert row['contribution_margin_eur_m2'] == pytest.approx(expected, rel=1e-5)


def test_tool_outputs_are_identical_across_private_inputs_and_repeatable(tmp_path):
    outputs = []
    for name, origin, soil in (('x', '2016-12-31T23:00:00+00:00', None), ('y', '2019-12-31T23:00:00+00:00', 8.0)):
        session, _ = run_fake_campaign(tmp_path, name, origin, soil)
        box = Toolbox(session, 11)
        calls = [('runs', {}), ('series_summary', {'unit': 0, 'variable': 'air_temperature_c'}),
                 ('fit_outcome_model', {}), ('predict', {'policies': [POLICIES['policy_a']]}),
                 ('space_filling_candidates', {'n': 8})]
        outputs.append(canonical([box.call(n, a) for n, a in calls]))
        assert outputs[-1] == canonical([Toolbox(session, 11).call(n, a) for n, a in reversed(calls)][::-1])
    assert outputs[0] == outputs[1]


# ── pure-function tests with a stub session ──────────────────────────────

def stub_session(runs):
    """runs: list of (policy, margin-like harvest) turned into a public transcript."""
    transcript, clock = [], 0
    for index, (policy, harvest) in enumerate(runs):
        unit = index % 4
        run_index = index // 4 + 1
        transcript.append({'ok': True, 'request': {'action': 'start', 'unit': unit, 'policy': policy},
                           'result': {'event': 'start', 'unit': str(unit), 'clock': clock, 'run_index': run_index}})
        transcript.append({'ok': True, 'request': {'action': 'observe', 'unit': unit},
                           'result': {'event': 'observe', 'clock': clock,
                                      'status': {'unit': str(unit), 'phase': 'cleanup', 'clock': clock,
                                                 'event': 'normal_completion', 'run_index': run_index},
                                      'final_aggregate': {'unit': str(unit), 'run_index': run_index,
                                                          'reason': 'normal_completion', 'closed_at': clock,
                                                          'accrued': {'harvest_kg_m2': harvest, 'heat_kwh_m2': 0.0,
                                                                      'light_kwh_m2': 0.0, 'co2_kg_m2': 0.0,
                                                                      'days': 0.0},
                                                          'event_cost_eur_m2': 0.0}}})
    return SimpleNamespace(task=copy.deepcopy(TASK), transcript=transcript)


def policy_at(u):
    spec = TASK['policy']['fields']
    p = {k: spec[k]['min'] + u[i] * (spec[k]['max'] - spec[k]['min']) for i, k in enumerate(FIELDS)}
    p['night_temperature_c'] = min(p['night_temperature_c'], p['day_temperature_c'])
    return p


def truth(u):
    return 60.0 - 40.0 * (u[0] - 0.6) ** 2 - 15.0 * (u[2] - 0.4) ** 2


def test_outcome_model_recovers_a_smooth_response_and_reports_uncertainty():
    rng = np.random.default_rng(3)
    points = rng.random((14, len(FIELDS)))
    price = TASK['economics']['fruit_price_eur_per_kg_fresh'] - TASK['economics']['harvest_handling_eur_per_kg']
    session = stub_session([(policy_at(u), truth(u) / price) for u in points])
    box = Toolbox(session, 5)
    fit = box.call('fit_outcome_model')['result']
    assert fit['completed_crops_used'] == 14
    scales = fit['length_scales_fraction_of_range']
    assert scales['day_temperature_c'] < scales['vent_rh_threshold_pct']
    test_u = [np.array([0.6, 0.3, 0.4, 0.5, 0.5, 0.5]), np.array([0.05, 0.05, 0.95, 0.5, 0.5, 0.5])]
    pred = box.call('predict', {'policies': [policy_at(u) for u in test_u]})['result']['predictions']
    assert abs(pred[0]['margin_mean_eur_m2'] - truth(test_u[0])) < 5.0
    assert pred[0]['margin_mean_eur_m2'] > pred[1]['margin_mean_eur_m2']
    assert all(p['sd_of_single_crop'] >= p['sd_of_mean'] > 0 for p in pred)


def test_candidates_are_feasible_seeded_and_respect_fixed_fields():
    box = Toolbox(stub_session([]), 1)
    out = box.call('space_filling_candidates', {'n': 20, 'fixed': {'co2_target_ppm': 800}})['result']['candidates']
    assert len(out) == 20 and all(c['co2_target_ppm'] == 800 for c in out)
    spec = TASK['policy']['fields']
    for c in out:
        assert c['night_temperature_c'] <= c['day_temperature_c']
        assert all(spec[k]['min'] <= c[k] <= spec[k]['max'] for k in FIELDS)
    again = Toolbox(stub_session([]), 1).call('space_filling_candidates', {'n': 20, 'fixed': {'co2_target_ppm': 800}})
    other = Toolbox(stub_session([]), 2).call('space_filling_candidates', {'n': 20, 'fixed': {'co2_target_ppm': 800}})
    assert again['result']['candidates'] == out and other['result']['candidates'] != out


def test_errors_budget_and_minimum_data():
    box = Toolbox(stub_session([(POLICIES['policy_a'], 10.0)]), 1, max_calls=3)
    with pytest.raises(ToolError, match='need at least 3 completed crops'):
        box.call('fit_outcome_model')
    with pytest.raises(ToolError, match='unknown tool'):
        box.call('optimise')
    with pytest.raises(ToolError, match='invalid arguments'):
        box.call('runs', {'extra': 1})
    with pytest.raises(ToolError, match='exactly the policy fields'):
        box.call('predict', {'policies': [{'day_temperature_c': 20}]})
    box.call('runs')
    with pytest.raises(ToolError, match='budget exhausted'):
        box.call('runs')


def test_new_run_after_a_closed_one_is_still_running():
    first = {'ok': True, 'request': {'action': 'start', 'unit': 0, 'policy': POLICIES['policy_a']},
             'result': {'event': 'start', 'unit': '0', 'clock': 0, 'run_index': 1}}
    closed = {'ok': True, 'request': {'action': 'advance', 'day': 1},
              'result': {'event': 'advance', 'clock': 86400,
                         'status': [{'unit': '0', 'phase': 'cleanup', 'clock': 86400, 'event': 'stop', 'run_index': 1}]}}
    second = {'ok': True, 'request': {'action': 'start', 'unit': 0, 'policy': POLICIES['policy_b']},
              'result': {'event': 'start', 'unit': '0', 'clock': 2 * 86400, 'run_index': 2}}
    runs = PublicHistory([first, closed, second]).runs
    assert runs[(0, 1)]['status'] == 'closed_totals_not_yet_observed'
    assert runs[(0, 2)]['status'] == 'running'


def test_catalog_and_outputs_pass_the_blinding_firewall(tmp_path):
    blinding = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
    session, _ = run_fake_campaign(tmp_path, 'a')
    box = Toolbox(session, 1)
    text = canonical([Toolbox.catalog(), box.call('runs'), box.call('fit_outcome_model')])
    assert_outbound_messages_safe([{'role': 'user', 'content': text}], blinding)
    assert LIMITS['max_calls'] == 64 and not any(math.isnan(v) for v in [])
