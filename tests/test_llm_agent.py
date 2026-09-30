import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slowlab.agent_client import CampaignProcess
from slowlab.agent_protocol import canonical, public_task_view
from slowlab.llm_agent import (LLMCampaignAgent, LLMConfig, initial_recommendation, parse_reply,
                               FormatError)
from slowlab.outbound_audit import AuditedCompleter, FirewallBlock, read_audit
from slowlab.prompt_firewall import load_blinding_policy
from slowlab.tools import Toolbox

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'configs/task_contract_v5.json').read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
BLINDING = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
TICK_DAY = 300 / 86400
CROP, END = 12 * TICK_DAY, 26 * TICK_DAY


def launch(tmp_path, name, origin='2016-12-31T23:00:00+00:00', soil=None):
    folder = tmp_path / name
    folder.mkdir()
    contract = copy.deepcopy(CONTRACT)
    contract['budget'].update(campaign_days=END, crop_days=CROP, cleanup_days=TICK_DAY, latest_start_day=13 * TICK_DAY)
    (folder / 'contract.json').write_text(json.dumps(contract))
    spec = {'backend': 'fake', 'contract': str(folder / 'contract.json'), 'feedback_mode': 'full',
            'fallback_policy': POLICIES['policy_a'], 'origin_utc': origin, 'soil_boundary_c': soil,
            'trace': None, 'settlement_out': str(folder / 'settlement.json'), 'failure_out': str(folder / 'failure.json')}
    (folder / 'site.json').write_text(json.dumps(spec))
    return CampaignProcess(folder / 'site.json', private_log=folder / 'server.log'), folder


def rule_model(messages):
    """A deterministic 'model' that reads only the prompt's state digest."""
    final = json.loads(messages[-1]['content'])
    state, instruction = final['state'], final['instruction']
    policies = [POLICIES['policy_a'], POLICIES['policy_b'], dict(POLICIES['policy_a'], day_temperature_c=21.0),
                dict(POLICIES['policy_b'], co2_target_ppm=700)]
    closed = [c for c in state['crops'] if c.get('contribution_margin_eur_m2') is not None]
    best = max(closed, key=lambda c: c['contribution_margin_eur_m2'])['policy'] if closed else POLICIES['policy_a']
    if 'Reply only' in instruction or state['days_left'] <= 1e-9:
        return json.dumps({'type': 'campaign', 'action': {'action': 'recommend', 'policy': best}})
    if not state['crops']:
        unit = 0
    else:
        unit = len(state['crops'])
    if unit < 4:
        return json.dumps({'type': 'campaign', 'action': {'action': 'start', 'unit': unit, 'policy': policies[unit]},
                           'reason': 'spread the first wave'})
    running = [c for c in state['crops'] if c['status'] != 'closed']
    if running and state['day'] >= CROP - 1e-9:
        c = running[0]
        return json.dumps({'type': 'campaign', 'action': {'action': 'observe', 'unit': c['unit'], 'run_index': c['crop']}})
    if state['analysis_tool_calls_left'] > 62:
        return json.dumps({'type': 'tool', 'name': 'runs', 'args': {}})
    target = CROP if state['day'] < CROP else END
    return json.dumps({'type': 'campaign', 'action': {'action': 'advance', 'day': target}})


def run_agent(tmp_path, name, model, *, config=LLMConfig(), **site):
    campaign, folder = launch(tmp_path, name, **site)
    audit = folder / 'audit.jsonl'
    complete = AuditedCompleter(model, BLINDING, audit, label=name)
    with campaign:
        agent = LLMCampaignAgent(campaign.session, Toolbox(campaign.session, 1), complete, config)
        summary = agent.run()
        campaign.close()
    return summary, read_audit(audit), json.loads((folder / 'settlement.json').read_text())


def test_rule_model_runs_a_whole_campaign_from_the_digest(tmp_path):
    summary, audit, settlement = run_agent(tmp_path, 'a', rule_model)
    assert summary['counts']['recommended'] and summary['counts']['format_errors'] == 0
    assert settlement['recommendation_fallback'] is False and settlement['starts'] == 4
    assert len(audit) == summary['counts']['llm_calls']
    assert all(record['firewall'] == 'pass' for record in audit)
    for record in audit:
        roles = [m['role'] for m in record['messages']]
        assert roles[:2] == ['system', 'user'] and roles[-1] == 'user'
        assert all(a != b for a, b in zip(roles[1:], roles[2:]))


def test_prompts_are_identical_across_private_site_inputs(tmp_path):
    a = run_agent(tmp_path, 'x', rule_model)[1]
    b = run_agent(tmp_path, 'y', rule_model, origin='2013-12-31T23:00:00+00:00', soil=8.0)[1]
    assert [r['messages_sha256'] for r in a] == [r['messages_sha256'] for r in b]
    assert canonical(a[-1]['messages']) == canonical(b[-1]['messages'])


def test_format_errors_invalid_actions_and_tool_errors_are_fed_back(tmp_path):
    script = iter(['I think we should start.',
                   json.dumps({'type': 'campaign', 'action': {'action': 'start', 'unit': 9, 'policy': POLICIES['policy_a']}}),
                   json.dumps({'type': 'tool', 'name': 'oracle', 'args': {}}),
                   '```json\n' + json.dumps({'type': 'campaign', 'action': {'action': 'advance', 'day': END}}) + '\n```'])
    seen = []

    def model(messages):
        seen.append(json.loads(messages[-1]['content'])['latest'])
        return next(script, json.dumps({'type': 'campaign', 'action': {'action': 'recommend', 'policy': POLICIES['policy_b']}}))

    summary, audit, settlement = run_agent(tmp_path, 'a', model)
    counts = summary['counts']
    assert (counts['format_errors'], counts['invalid_actions'], counts['tool_errors']) == (1, 1, 1)
    assert [s.get('kind') for s in seen[1:4]] == ['format_error', 'invalid_action', 'tool_error']
    assert settlement['recommendation'] == POLICIES['policy_b']


def test_harness_forces_the_endgame_when_decision_calls_run_low(tmp_path):
    def greedy(messages):
        final = json.loads(messages[-1]['content'])
        if 'Reply only' in final['instruction']:
            return json.dumps({'type': 'campaign', 'action': {'action': 'recommend', 'policy': POLICIES['policy_b']}})
        return json.dumps({'type': 'campaign', 'action': {'action': 'observe', 'unit': 0}})

    summary, audit, settlement = run_agent(tmp_path, 'a', greedy)
    assert summary['counts']['forced_advance'] and summary['counts']['recommended']
    assert settlement['clock'] == round(END * 86400) and settlement['recommendation_fallback'] is False
    assert settlement['decision_calls'] == CONTRACT['budget']['max_decision_calls']
    notices = [json.loads(r['messages'][-1]['content'])['latest'] for r in audit]
    assert any(isinstance(n, dict) and n.get('harness', {}).get('kind') == 'harness_forced_advance' for n in notices)


def test_prompt_length_stays_within_budget(tmp_path):
    def chatty(messages):
        final = json.loads(messages[-1]['content'])
        if final['state']['analysis_tool_calls_left'] > 30:
            return json.dumps({'type': 'tool', 'name': 'space_filling_candidates', 'args': {'n': 64}})
        if final['state']['days_left'] > 0:
            return json.dumps({'type': 'campaign', 'action': {'action': 'advance', 'day': END}})
        return json.dumps({'type': 'campaign', 'action': {'action': 'recommend', 'policy': POLICIES['policy_a']}})

    config = LLMConfig(prompt_char_budget=40_000)
    summary, audit, _ = run_agent(tmp_path, 'a', chatty, config=config)
    sizes = [sum(len(m['content']) for m in r['messages']) for r in audit]
    assert max(sizes) <= config.prompt_char_budget + 64
    assert len(audit[-1]['messages']) < 2 * summary['counts']['llm_calls']


def test_firewall_block_aborts_before_the_model_is_called(tmp_path):
    task = public_task_view(CONTRACT, 'full')
    task['setting'] += ' Powered by GreenLight.'
    session = SimpleNamespace(task=task, transcript=[], dispatch=None)
    called = []
    complete = AuditedCompleter(lambda m: called.append(m) or '{}', BLINDING, tmp_path / 'audit.jsonl')
    agent = LLMCampaignAgent(session, Toolbox(session, 1), complete)
    with pytest.raises(FirewallBlock):
        agent.run()
    assert not called and read_audit(tmp_path / 'audit.jsonl')[0]['firewall'] == 'blocked'


def test_model_mentioning_a_simulator_name_is_not_blocked(tmp_path):
    def speculative(messages):
        final = json.loads(messages[-1]['content'])
        if final['state']['days_left'] > 0:
            return json.dumps({'type': 'campaign', 'action': {'action': 'advance', 'day': END},
                               'reason': 'this looks like GreenLight to me'})
        return json.dumps({'type': 'campaign', 'action': {'action': 'recommend', 'policy': POLICIES['policy_a']}})
    summary, audit, _ = run_agent(tmp_path, 'a', speculative)
    assert summary['counts']['recommended'] and all(r['firewall'] == 'pass' for r in audit)


def test_initial_recommendation_validates_and_retries():
    task = public_task_view(CONTRACT, 'full')
    replies = iter(['nothing useful',
                    json.dumps({'type': 'initial_recommendation', 'policy': dict(POLICIES['policy_a'], night_temperature_c=30)}),
                    json.dumps({'type': 'initial_recommendation', 'policy': POLICIES['policy_b'], 'reason': 'prior'})])
    policy, log = initial_recommendation(task, lambda m: next(replies), attempts=3)
    assert policy == POLICIES['policy_b'] and [e['ok'] for e in log] == [False, False, True]
    policy, log = initial_recommendation(task, lambda m: 'no', attempts=2)
    assert policy is None and len(log) == 2


def test_parse_reply_shapes():
    assert parse_reply('```json\n{"type": "tool", "name": "runs", "args": {}}\n```')['name'] == 'runs'
    for bad in ('', '{"type": "campaign"}', '{"type": "other"}', '{"type": "tool", "name": 3}', '[1, 2]'):
        with pytest.raises(FormatError):
            parse_reply(bad)


def test_scripted_model_runs_the_two_wave_plan_end_to_end(tmp_path):
    from slowlab.scripted_model import ScriptedModel
    campaign, folder = launch(tmp_path, 'a')
    with campaign:
        model = ScriptedModel(campaign.session.task)
        complete = AuditedCompleter(model, BLINDING, folder / 'audit.jsonl')
        summary = LLMCampaignAgent(campaign.session, Toolbox(campaign.session, 1), complete).run()
        campaign.close()
    settlement = json.loads((folder / 'settlement.json').read_text())
    assert summary['counts']['recommended'] and summary['counts']['format_errors'] == 0
    assert settlement['starts'] == 8 and settlement['recommendation_fallback'] is False
