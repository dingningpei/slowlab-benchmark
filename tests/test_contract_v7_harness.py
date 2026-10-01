import copy
import hashlib
import json
from pathlib import Path

import pytest

from slowlab.agent_client import CampaignProcess
from slowlab.agent_protocol import public_task_view
from slowlab.llm_agent import LLMCampaignAgent
from slowlab.outbound_audit import AuditedCompleter, read_audit
from slowlab.prompt_firewall import load_blinding_policy
from slowlab.tools import Toolbox

ROOT = Path(__file__).resolve().parents[1]
V6_PATH = ROOT / 'configs/task_contract_v6.json'
V6 = json.loads(V6_PATH.read_text())
V7 = json.loads((ROOT / 'configs/task_contract_v7.json').read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
BLINDING = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
T = 300 / 86400


def test_v7_supersedes_v6_and_changes_only_budget_and_observations():
    assert V7['supersedes']['sha256'] == hashlib.sha256(V6_PATH.read_bytes()).hexdigest()
    old, new = copy.deepcopy(V6), copy.deepcopy(V7)
    for c in (old, new):
        for key in ('contract_id', 'date', 'supersedes'):
            c.pop(key)
        for key in ('max_context_tokens', 'decision_call_actions', 'context'):
            c['budget'].pop(key, None)
        c['observations'].pop('daily_summary', None)
    assert old == new
    assert V7['budget']['context']['max_input_chars_per_campaign'] == 2_000_000


def launch(tmp_path, name, contract, mode='full', **budget):
    folder = tmp_path / name
    folder.mkdir()
    c = copy.deepcopy(contract)
    c['budget'].update(campaign_days=40 * T, crop_days=24 * T, cleanup_days=T, latest_start_day=13 * T)
    for key, value in budget.items():
        c['budget']['context'][key] = value
    (folder / 'c.json').write_text(json.dumps(c))
    spec = {'backend': 'fake', 'contract': str(folder / 'c.json'), 'feedback_mode': mode,
            'fallback_policy': POLICIES['policy_a'], 'origin_utc': '2016-12-31T23:00:00+00:00', 'soil_boundary_c': None,
            'trace': None, 'settlement_out': str(folder / 's.json'), 'failure_out': str(folder / 'f.json')}
    (folder / 'site.json').write_text(json.dumps(spec))
    return CampaignProcess(folder / 'site.json', private_log=folder / 'log'), folder


def test_reads_do_not_spend_decision_calls_and_daily_summaries_aggregate(tmp_path):
    for contract, counted in ((V6, True), (V7, False)):
        campaign, _ = launch(tmp_path, contract['contract_id'], contract)
        with campaign:
            s = campaign.session
            s.dispatch({'action': 'start', 'unit': 0, 'policy': POLICIES['policy_a']})
            s.dispatch({'action': 'advance', 'day': 20 * T})
            r = s.dispatch({'action': 'observe', 'unit': 0, 'variable': 'heating_energy'})
            assert r['budget']['decision_calls_used'] == (2 if counted else 1)
            if contract is V7:
                raw = r['records']
                d = s.dispatch({'action': 'observe', 'unit': 0, 'variable': 'heating_energy', 'resolution': 'daily'})
                assert d['daily'] == [{'day': 0, 'n': len(raw), 'mean': pytest.approx(sum(x['value'] for x in raw) / len(raw)),
                                       'min': min(x['value'] for x in raw), 'max': max(x['value'] for x in raw)}]
            else:
                with pytest.raises(Exception, match='daily'):
                    s.dispatch({'action': 'observe', 'unit': 0, 'variable': 'heating_energy', 'resolution': 'daily'})
            campaign.close()
    campaign, _ = launch(tmp_path, 'endpoint', V7, mode='endpoint')
    with campaign:
        assert 'resolution' not in campaign.session.task['actions']['observe']
        campaign.session.dispatch({'action': 'start', 'unit': 0, 'policy': POLICIES['policy_a']})
        with pytest.raises(Exception, match='endpoint'):
            campaign.session.dispatch({'action': 'observe', 'unit': 0, 'variable': 'heating_energy',
                                       'resolution': 'daily'})
        campaign.close()


class Reader:
    """Scripted model: start one crop, set notes, read the whole heating series every turn, then recommend."""

    def __init__(self, reads, notes='plan: watch heating'):
        self.reads, self.notes, self.turn = reads, notes, 0

    def __call__(self, messages):
        final = json.loads(messages[-1]['content'])
        state, self.turn = final['state'], self.turn + 1
        if 'Reply only' in final['instruction'] or state['days_left'] <= 1e-9:
            return json.dumps({'type': 'campaign', 'action': {'action': 'recommend', 'policy': POLICIES['policy_b']}})
        if not state['crops']:
            return json.dumps({'type': 'campaign', 'action': {'action': 'start', 'unit': 0, 'policy': POLICIES['policy_a']},
                               'notes': self.notes})
        if state['day'] < 20 * T:
            return json.dumps({'type': 'campaign', 'action': {'action': 'advance', 'day': 20 * T}})
        if self.reads > 0:
            self.reads -= 1
            return json.dumps({'type': 'campaign', 'action': {'action': 'observe', 'unit': 0, 'variable': 'heating_energy'}})
        return json.dumps({'type': 'campaign', 'action': {'action': 'advance', 'day': 40 * T}})


def run(tmp_path, name, model, **budget):
    campaign, folder = launch(tmp_path, name, V7, **budget)
    with campaign:
        complete = AuditedCompleter(model, BLINDING, folder / 'audit.jsonl')
        agent = LLMCampaignAgent(campaign.session, Toolbox(campaign.session, 1), complete)
        summary = agent.run()
        campaign.close()
    return summary, read_audit(folder / 'audit.jsonl'), json.loads((folder / 's.json').read_text())


def test_old_results_become_notes_and_recent_ones_stay_whole(tmp_path):
    summary, audit, _ = run(tmp_path, 'retention', Reader(reads=9))
    assert summary['counts']['observe_calls'] == 9 and summary['counts']['records_received'] == 9 * 21
    last = audit[-1]['messages']
    users = [json.loads(m['content']) for m in last[2:-1] if m['role'] == 'user']
    full = [u for u in users if isinstance(u.get('result'), dict) and 'records' in u['result']]
    stubs = [u for u in users if 'note' in u]
    assert stubs and all('no longer shown' in u['note'] for u in stubs)
    assert len(full) <= V7['budget']['context']['raw_result_turns']
    assert all(len(u['result']['records']) == 21 for u in full)
    assert summary['counts']['input_chars'] == sum(sum(len(m['content']) for m in r['messages']) for r in audit)


def test_notes_live_in_the_models_own_turn_and_are_bounded(tmp_path):
    summary, audit, _ = run(tmp_path, 'notes', Reader(reads=1, notes='GreenLight-looking text is the model\'s own'))
    later = audit[-1]['messages']
    assert 'YOUR NOTES' in later[-2]['content'] and later[-2]['role'] == 'assistant'
    assert not any('YOUR NOTES' in m['content'] for m in later if m['role'] != 'assistant')
    assert all(r['firewall'] == 'pass' for r in audit) and summary['final_notes'].startswith('GreenLight')
    summary, _, _ = run(tmp_path, 'long_notes', Reader(reads=1, notes='x' * 2001))
    assert summary['counts']['format_errors'] >= 1 and summary['final_notes'] == ''


def test_character_budget_forces_the_endgame(tmp_path):
    summary, audit, settlement = run(tmp_path, 'budget', Reader(reads=40), max_input_chars_per_campaign=260_000,
                                     max_prompt_chars=30_000)
    counts = summary['counts']
    assert counts['context_exhausted'] and counts['recommended'] and counts['forced_advance']
    assert counts['observe_calls'] < 40
    assert settlement['recommendation'] == POLICIES['policy_b'] and settlement['recommendation_fallback'] is False
    assert any(e.get('cause') == 'context_budget' for e in summary['log'])
    roles = [m['role'] for m in audit[-1]['messages']]
    assert roles == ['system', 'user', 'assistant', 'user']  # only the latest turn and its notes remain


def test_a_result_larger_than_the_prompt_limit_is_truncated(tmp_path):
    summary, audit, _ = run(tmp_path, 'truncate', Reader(reads=1), max_prompt_chars=12_000)
    assert summary['counts']['truncated_results'] >= 1
    assert max(sum(len(m['content']) for m in r['messages']) for r in audit) <= 12_000 + 2_000


def test_day_zero_notes_continue_after_the_branch(tmp_path):
    campaign, folder = launch(tmp_path, 'branch_notes', V7)
    with campaign:
        complete = AuditedCompleter(Reader(reads=0, notes='day-0 plan'), BLINDING, folder / 'a.jsonl')
        prefix = LLMCampaignAgent(campaign.session, Toolbox(campaign.session, 1), complete).run_day_zero_design(max_calls=2)
        assert prefix['notes'] == 'day-0 plan'
        agent = LLMCampaignAgent(campaign.session, Toolbox(campaign.session, 1), complete, history=prefix['history'],
                                 counts=prefix['counts'], notes=prefix['notes'])
        assert 'day-0 plan' in agent.messages('next')[-2]['content']
        campaign.close()
