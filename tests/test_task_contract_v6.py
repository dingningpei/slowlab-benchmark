import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from slowlab import fake_backend
from slowlab.agent_protocol import canonical, public_task_view
from slowlab.cached_solver import SolverStall
from slowlab.campaign_executor import CampaignExecutor
from slowlab.prompt_firewall import assert_outbound_messages_safe, load_blinding_policy

ROOT = Path(__file__).resolve().parents[1]
V5_PATH = ROOT / 'configs/task_contract_v5.json'
V5 = json.loads(V5_PATH.read_text())
V6 = json.loads((ROOT / 'configs/task_contract_v6.json').read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
T = 300 / 86400


def test_v6_supersedes_the_exact_v5_file_and_changes_only_events():
    assert V6['supersedes']['contract_id'] == V5['contract_id']
    assert V6['supersedes']['sha256'] == hashlib.sha256(V5_PATH.read_bytes()).hexdigest()
    old, new = copy.deepcopy(V5), copy.deepcopy(V6)
    for contract in (old, new):
        for key in ('contract_id', 'date', 'supersedes'):
            contract.pop(key)
        contract['events'].pop('solver_stall', None)
        contract['events'].pop('numerical_failure')
    assert old == new
    assert V6['events']['solver_stall']['max_rhs_evaluations_per_control_step'] == 200000
    assert (ROOT / V6['events']['solver_stall']['evidence']).is_file()


def test_public_view_adds_the_collapse_rule_without_leaking():
    view, old = public_task_view(V6, 'full'), public_task_view(V5, 'full')
    rule = V6['events']['solver_stall']['public_rule']
    assert rule in view['rules'] and rule not in old['rules']
    assert view['rules'].index(rule) == old['rules'].index(
        'If indoor air stays below 5 C or above 40 C for one hour, the crop is stopped for safety.') + 1
    assert [r for r in view['rules'] if r != rule] == old['rules']
    text = canonical(view).lower()
    assert 'solver' not in text and 'evaluation' not in rule.lower()
    assert_outbound_messages_safe([{'role': 'user', 'content': canonical(view)}],
                                  load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json'))


class StallingLifecycle(fake_backend.FakeLifecycle):
    """Unit '1' stalls once its active crop reaches 900 s; empty compartments never stall."""
    stall_at = 900.0
    stall_empty = False

    def __init__(self, contract, source, start, **kwargs):
        super().__init__(contract, source, start, **kwargs)
        self.unit = str(StallingLifecycle.counter)
        StallingLifecycle.counter += 1

    def step(self, commands, end):
        if self.unit == '1' and end > self.stall_at and (self.mode == 'active' or self.stall_empty):
            raise SolverStall(self.clock, end, self.clock + 1e-9, np.zeros(2), ['a', 'b'], 200001)
        return super().step(commands, end)


def run(contract, stall_empty=False):
    StallingLifecycle.counter = 0
    StallingLifecycle.stall_empty = stall_empty
    c = copy.deepcopy(contract)
    c['budget'].update(campaign_days=12 * T, crop_days=8 * T, cleanup_days=T, latest_start_day=2 * T)
    x = CampaignExecutor(c, None, None, feedback_mode='endpoint', fallback_policy=POLICIES['policy_a'],
                         lifecycle_factory=StallingLifecycle, sample_endpoint=fake_backend.sample_endpoint)
    for unit in range(4):
        x.dispatch({'action': 'start', 'unit': unit, 'policy': POLICIES['policy_a']})
    return x


def test_active_stall_becomes_a_safety_stop_and_the_campaign_continues():
    x = run(V6)
    x.dispatch({'action': 'advance', 'day': 12 * T})
    closed = {r['unit']: r for r in x.private_crop_totals()}
    assert closed['1']['reason'] == 'safety_stop' and closed['1']['closed_at'] == 900
    assert {closed[u]['reason'] for u in '023'} == {'normal_completion'}
    assert [e for e in x.event_log if e['event'] == 'crop_collapse_solver_stall'][0]['clock'] == 900
    observed = x.dispatch({'action': 'observe', 'unit': 1})
    assert observed['final_aggregate']['reason'] == 'safety_stop'
    assert 'stall' not in json.dumps(observed).lower()
    x.dispatch({'action': 'recommend', 'policy': POLICIES['policy_b']})
    settlement = x.settlement()
    assert settlement['clock'] == round(12 * 300)


def test_without_the_v6_rule_or_in_an_empty_compartment_a_stall_is_infrastructure():
    x = run(V5)
    with pytest.raises(SolverStall):
        x.dispatch({'action': 'advance', 'day': 12 * T})
    assert x._failed['unit'] == '1'
    y = run(V6, stall_empty=True)
    with pytest.raises(SolverStall):
        y.dispatch({'action': 'advance', 'day': 12 * T})
    assert [e['event'] for e in y.event_log].count('crop_collapse_solver_stall') == 1 and y._failed['unit'] == '1'
