import copy
import hashlib
import json
from pathlib import Path

from slowlab.agent_protocol import canonical, public_task_view
from slowlab.prompt_firewall import assert_outbound_messages_safe, load_blinding_policy

ROOT = Path(__file__).resolve().parents[1]
V4_PATH = ROOT / 'configs/task_contract_v4.json'
V5 = json.loads((ROOT / 'configs/task_contract_v5.json').read_text())
V4 = json.loads(V4_PATH.read_text())


def test_v5_supersedes_the_exact_v4_file_and_changes_only_evaluation():
    assert V5['supersedes']['contract_id'] == V4['contract_id']
    assert V5['supersedes']['sha256'] == hashlib.sha256(V4_PATH.read_bytes()).hexdigest()
    old, new = copy.deepcopy(V4), copy.deepcopy(V5)
    for contract in (old, new):
        for key in ('contract_id', 'date', 'supersedes'):
            contract.pop(key)
        contract['evaluation'].pop('deployment', None)
    assert old == new


def test_v5_deployment_is_january_and_july_mean():
    deployment = V5['evaluation']['deployment']
    assert deployment['plantings_calendar_day'] == [0, 182]
    assert deployment['aggregation'] == 'mean' and deployment['public'] is True
    assert (ROOT / 'results/bo_dev_runs_20260930.json').is_file()


def test_public_view_states_the_scoring_rule_and_the_start_date():
    view = public_task_view(V5, 'full')
    assert view['scoring']['planting_calendar_days'] == [0, 182]
    assert view['calendar']['campaign_day_0'].startswith('1 January')
    assert view['rules'][-1] == view['scoring']['rule']
    blinding = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
    assert_outbound_messages_safe([{'role': 'user', 'content': canonical(view)}], blinding)
    assert public_task_view(V4, 'full')['scoring'] is None
