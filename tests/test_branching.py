import copy
import json
from pathlib import Path

import pytest

from slowlab.agent_client import CampaignProcess
from slowlab.branching import BranchDivergence, replay_prefix, run_branched_campaign
from slowlab.outbound_audit import AuditedCompleter, read_audit
from slowlab.prompt_firewall import load_blinding_policy
from slowlab.scripted_model import ScriptedModel

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'configs/task_contract_v5.json').read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
BLINDING = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
TICK_DAY = 300 / 86400


def small_contract(tmp_path, compartments=4):
    contract = copy.deepcopy(CONTRACT)
    contract['facility']['compartments'] = compartments
    contract['budget'].update(campaign_days=26 * TICK_DAY, crop_days=12 * TICK_DAY,
                              cleanup_days=TICK_DAY, latest_start_day=13 * TICK_DAY)
    path = tmp_path / f'contract_{compartments}.json'
    path.write_text(json.dumps(contract))
    return contract, path


def run(tmp_path):
    contract, path = small_contract(tmp_path)
    base = {'backend': 'fake', 'contract': str(path), 'origin_utc': '2016-12-31T23:00:00+00:00', 'soil_boundary_c': None}
    audit = tmp_path / 'audit.jsonl'
    model = ScriptedModel(__import__('slowlab.agent_protocol', fromlist=['x']).public_task_view(contract, None))
    out = run_branched_campaign(contract, base, tmp_path / 'private',
                                lambda label: AuditedCompleter(model, BLINDING, audit, label=label),
                                tool_seed=3, default_policy=POLICIES['policy_a'])
    return out, read_audit(audit)


def test_both_branches_share_the_day_zero_prefix_and_complete(tmp_path):
    out, audit = run(tmp_path)
    prefix = out['shared_day_0']['transcript']
    assert out['shared_day_0']['starts'] == 4
    for mode in ('full', 'endpoint'):
        assert out[mode]['server_exit_code'] == 0 and out[mode]['summary']['counts']['recommended']
        branch = out[mode]['transcript']
        assert [e['response_sha256'] for e in branch[:len(prefix)]] == [e['response_sha256'] for e in prefix]
        settlement = json.loads((tmp_path / 'private' / mode / 'settlement.json').read_text())
        assert settlement['recommendation_fallback'] is False and settlement['starts'] == 8
    assert not any('variable' in e['request'] for e in out['endpoint']['transcript'])


def test_shared_prompts_never_name_a_feedback_condition(tmp_path):
    out, audit = run(tmp_path)
    labels = [r['label'] for r in audit]
    assert labels[0] == 'initial_recommendation' and 'shared_day_0' in labels
    for record in audit:
        if record['label'] in ('initial_recommendation', 'shared_day_0'):
            text = json.dumps(record['messages'])
            assert 'assigned_after_day_0_design' in text
            assert '\\"mode\\":\\"full\\"' not in text and '\\"mode\\":\\"endpoint\\"' not in text


def test_branches_continue_the_same_conversation_after_one_notice(tmp_path):
    out, audit = run(tmp_path)
    first = {label: next(r for r in audit if r['label'] == label) for label in ('full', 'endpoint')}
    full, endpoint = first['full']['messages'], first['endpoint']['messages']
    assert full[1:-1] == endpoint[1:-1]
    assert full[0] != endpoint[0]
    notices = {label: json.loads(first[label]['messages'][-1]['content'])['latest']['harness'] for label in first}
    assert notices['full']['mode'] == 'full' and notices['endpoint']['mode'] == 'endpoint'
    shared_calls = out['shared_day_0']['counts']['llm_calls']
    for mode in ('full', 'endpoint'):
        assert out[mode]['summary']['counts']['llm_calls'] > shared_calls


def test_replay_divergence_stops_the_run(tmp_path):
    out, _ = run(tmp_path)
    prefix = out['shared_day_0']['transcript']
    _, path = small_contract(tmp_path, compartments=3)
    spec = {'backend': 'fake', 'contract': str(path), 'feedback_mode': 'endpoint', 'fallback_policy': POLICIES['policy_a'],
            'origin_utc': '2016-12-31T23:00:00+00:00', 'soil_boundary_c': None, 'trace': None,
            'settlement_out': str(tmp_path / 's.json'), 'failure_out': str(tmp_path / 'f.json')}
    (tmp_path / 'site3.json').write_text(json.dumps(spec))
    with CampaignProcess(tmp_path / 'site3.json', private_log=tmp_path / 'log') as campaign:
        with pytest.raises(BranchDivergence):
            replay_prefix(campaign.session, prefix)
