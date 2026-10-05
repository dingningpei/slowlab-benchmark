"""Frozen models must see policy fields in their training order, whatever the key order of the task they
are given (agent sessions deliver the task with sorted keys). These tests would have caught the bug found
on 2026-10-05: they compare the deployed path (through an agent session) with the offline path."""
import copy
import json
from pathlib import Path

import numpy as np

from slowlab.agent_client import CampaignProcess
from slowlab.agent_protocol import public_task_view
from slowlab.bo_agent import BOConfig, GPBOAgent
from slowlab.history_packet import build_packet, gp_reader
from slowlab.prior_bo import PriorBOConfig, PriorLocalBOAgent
from slowlab.process_predictor import POLICY_ORDER, feature_row, predict_from_history
from slowlab.tools import PublicHistory, _predictor

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'configs/task_contract_v8.json').read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
PRIOR = json.loads((ROOT / 'configs/prior_bo_v1.json').read_text())
ANCHOR = json.loads((ROOT / 'configs/fixed_reference_v1.json').read_text())['policy']


def campaign(tmp_path, make_agent, *, mode='full', real_calendar=True):
    contract = copy.deepcopy(CONTRACT)
    if not real_calendar:
        T = 300 / 86400
        contract['budget'].update(campaign_days=26 * T, crop_days=12 * T, cleanup_days=T, latest_start_day=13 * T)
    (tmp_path / 'c.json').write_text(json.dumps(contract))
    spec = {'backend': 'fake', 'contract': str(tmp_path / 'c.json'), 'feedback_mode': mode,
            'fallback_policy': POLICIES['policy_a'], 'origin_utc': '2016-12-31T23:00:00+00:00', 'soil_boundary_c': None,
            'trace': None, 'settlement_out': str(tmp_path / 's.json'), 'failure_out': str(tmp_path / 'f.json')}
    (tmp_path / 'site.json').write_text(json.dumps(spec))
    with CampaignProcess(tmp_path / 'site.json', private_log=tmp_path / 'log') as cp:
        agent = make_agent(cp.session)
        summary = agent.run()
        task, transcript = cp.session.task, cp.session.transcript
        cp.close()
    return agent, summary, task, transcript


def reordered(task, order):
    out = copy.deepcopy(task)
    out['policy']['fields'] = {k: task['policy']['fields'][k] for k in order}
    return out


def test_the_session_task_is_sorted_so_order_must_not_matter():
    view = public_task_view(CONTRACT, 'full')
    assert list(view['policy']['fields']) == list(POLICY_ORDER)


def test_feature_rows_ignore_the_key_order():
    fields = CONTRACT['policy']['fields']
    sorted_fields = {k: fields[k] for k in sorted(fields)}
    so_far = {'harvest_kg_m2': 1.0, 'heat_kwh_m2': 20.0, 'light_kwh_m2': 10.0, 'co2_kg_m2': 1.0}
    a = feature_row(fields, ANCHOR, 0.0, 30.0, so_far, 1.5, 'all')
    b = feature_row(sorted_fields, ANCHOR, 0.0, 30.0, so_far, 1.5, 'all')
    assert a == b


def test_predictor_in_a_session_equals_the_offline_predictor(tmp_path):
    agent, summary, task, transcript = campaign(
        tmp_path, lambda s: GPBOAgent(s, 3, BOConfig(schedule='staggered', initial_units=2, stagger_fraction=0.5)))
    assert list(task['policy']['fields']) == sorted(POLICY_ORDER)          # the deployed path sorts keys
    # the evidence the agent held when it predicted: everything before the mid-season starts
    starts = [i for i, e in enumerate(transcript) if e['request']['action'] == 'start']
    history = PublicHistory(transcript[:starts[2]])
    deployed = predict_from_history(task, history, 0, 1, _predictor())
    offline = predict_from_history(reordered(task, POLICY_ORDER), history, 0, 1, _predictor())
    assert deployed == offline
    logged = next(e for e in summary['log'] if e['action'] == 'predicted_final_margins')['values'][0]
    assert abs(logged - offline['predicted_contribution_margin_eur_m2']) < 1e-6


def test_prior_bo_uses_the_kernel_order_and_matches_the_offline_reader(tmp_path):
    config = PriorBOConfig(anchor=ANCHOR, kernel=PRIOR['hyperparameters']['kernel'],
                           noise_ratio=PRIOR['hyperparameters']['noise_ratio'], radius=0.15)
    agent, summary, task, transcript = campaign(tmp_path, lambda s: PriorLocalBOAgent(s, 5, config), mode='endpoint',
                                                real_calendar=False)
    assert agent.fields == list(POLICY_ORDER)
    x = agent._scale(ANCHOR)
    fields = CONTRACT['policy']['fields']
    assert np.allclose(x, [(ANCHOR[k] - fields[k]['min']) / (fields[k]['max'] - fields[k]['min']) for k in POLICY_ORDER])
    # The baseline's final step and the E3 GP Reader on the same evidence must agree exactly.
    packet = build_packet(task, transcript)
    reader_config = {**PRIOR, 'anchor_policy': ANCHOR, 'policy_order': list(POLICY_ORDER)}
    assert gp_reader(task, packet, reader_config)['policy'] == summary['recommendation']
    assert gp_reader(reordered(task, POLICY_ORDER), packet, reader_config)['policy'] == summary['recommendation']
