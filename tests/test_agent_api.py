import copy
import io
import json
import re
from pathlib import Path

import pytest

from slowlab.agent_client import CampaignProcess, InfrastructureFailure, InvalidAction
from slowlab.agent_protocol import PROTOCOL, canonical, public_task_view
from slowlab.executor_server import serve
from slowlab.prompt_firewall import assert_outbound_messages_safe, load_blinding_policy

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'configs/task_contract_v5.json').read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
BLINDING = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
TICK_DAY = 300 / 86400


def small_contract(tmp_path):
    contract = copy.deepcopy(CONTRACT)
    contract['budget'].update(campaign_days=6 * TICK_DAY, crop_days=3 * TICK_DAY,
                              cleanup_days=TICK_DAY, latest_start_day=2 * TICK_DAY)
    path = tmp_path / 'contract.json'
    path.write_text(json.dumps(contract))
    return path


def site_spec(tmp_path, name, *, backend='fake', mode='full', origin='2016-12-31T23:00:00+00:00', soil=None):
    folder = tmp_path / name
    folder.mkdir()
    spec = {'backend': backend, 'contract': str(small_contract(folder)), 'feedback_mode': mode,
            'fallback_policy': POLICIES['policy_a'], 'origin_utc': origin, 'soil_boundary_c': soil,
            'trace': str(folder / 'private_trace.jsonl.gz'), 'settlement_out': str(folder / 'settlement.json'),
            'failure_out': str(folder / 'failure.json')}
    path = folder / 'site.json'
    path.write_text(json.dumps(spec))
    return path, folder


def script(session):
    session.dispatch({'action': 'start', 'unit': 0, 'policy': POLICIES['policy_a']})
    session.dispatch({'action': 'start', 'unit': 1, 'policy': POLICIES['policy_b']})
    session.dispatch({'action': 'advance', 'day': 2 * TICK_DAY})
    if session.task['feedback']['mode'] == 'full':
        session.dispatch({'action': 'observe', 'unit': 0, 'variable': 'air_temperature_c'})
    session.dispatch({'action': 'stop', 'unit': 1})
    session.dispatch({'action': 'advance', 'day': 6 * TICK_DAY})
    session.dispatch({'action': 'recommend', 'policy': POLICIES['policy_b']})


def test_public_task_view_passes_the_blinding_firewall_and_hides_internals():
    for mode in ('full', 'endpoint'):
        view = public_task_view(CONTRACT, mode)
        text = canonical(view)
        assert_outbound_messages_safe([{'role': 'user', 'content': text}], BLINDING)
        for leaked in ('tSpDay', 'co2SpDay', 'rhMax', 'heatDeadZone', 'mcFruitHar', 'tSoOut', 'Cabauw', 'fa502ed'):
            assert leaked not in text
        assert not re.search(r'\b(19|20)\d\d\b', text), 'no calendar year may appear'
        assert set(view['policy']['fields']) == set(CONTRACT['policy']['fields'])
        assert all(set(spec) == {'min', 'max', 'unit'} for spec in view['policy']['fields'].values())
        assert view['protocol'] == PROTOCOL


def test_full_campaign_through_the_process_boundary(tmp_path):
    spec, folder = site_spec(tmp_path, 'a')
    with CampaignProcess(spec, private_log=folder / 'server.log') as campaign:
        session = campaign.session
        assert session.task['compartments'] == [0, 1, 2, 3]
        script(session)
        assert campaign.close() == 0
    settlement = json.loads((folder / 'settlement.json').read_text())
    assert settlement['clock'] == 6 * 300 and settlement['starts'] == 2
    public = canonical(session.transcript)
    assert 'ledger_by_unit' not in public and 'synthetic_margin' not in public
    assert str(tmp_path) not in public and str(tmp_path) not in canonical(session.task)
    assert (folder / 'private_trace.jsonl.gz').exists()


def test_invalid_actions_are_reported_and_the_campaign_continues(tmp_path):
    spec, folder = site_spec(tmp_path, 'a')
    with CampaignProcess(spec, private_log=folder / 'server.log') as campaign:
        session = campaign.session
        with pytest.raises(InvalidAction, match='unknown compartment'):
            session.dispatch({'action': 'start', 'unit': 9, 'policy': POLICIES['policy_a']})
        with pytest.raises(InvalidAction, match='unsupported action'):
            session.dispatch({'action': 'read_truth'})
        assert session.dispatch({'action': 'start', 'unit': 0, 'policy': POLICIES['policy_a']})['event'] == 'start'
        assert [entry['ok'] for entry in session.transcript] == [False, False, True]


def test_endpoint_sessions_cannot_read_running_science(tmp_path):
    spec, folder = site_spec(tmp_path, 'a', mode='endpoint')
    with CampaignProcess(spec, private_log=folder / 'server.log') as campaign:
        session = campaign.session
        session.dispatch({'action': 'start', 'unit': 0, 'policy': POLICIES['policy_a']})
        with pytest.raises(InvalidAction, match='endpoint feedback condition'):
            session.dispatch({'action': 'observe', 'unit': 0, 'variable': 'air_temperature_c'})


def test_public_transcript_does_not_depend_on_private_site_inputs(tmp_path):
    transcripts = []
    for name, origin, soil in (('x', '2016-12-31T23:00:00+00:00', None),
                               ('y', '2013-12-31T23:00:00+00:00', 8.0),
                               ('z', '2019-12-31T23:00:00+00:00', 19.5)):
        spec, folder = site_spec(tmp_path, name, origin=origin, soil=soil)
        with CampaignProcess(spec, private_log=folder / 'server.log') as campaign:
            script(campaign.session)
            transcripts.append((canonical(campaign.session.task), canonical(campaign.session.transcript)))
            campaign.close()
    assert transcripts[0] == transcripts[1] == transcripts[2]


def test_infrastructure_failure_is_generic_publicly_and_detailed_privately(tmp_path):
    spec, folder = site_spec(tmp_path, 'a', backend='fake_failing')
    campaign = CampaignProcess(spec, private_log=folder / 'server.log')
    session = campaign.session
    session.dispatch({'action': 'start', 'unit': 0, 'policy': POLICIES['policy_a']})
    with pytest.raises(InfrastructureFailure) as info:
        session.dispatch({'action': 'advance', 'day': 3 * TICK_DAY})
    assert 'injected' not in str(info.value) and 'Floating' not in str(info.value)
    assert campaign.returncode == 3
    failure = json.loads((folder / 'failure.json').read_text())
    assert 'FloatingPointError: injected numerical fault' in failure['error']
    with pytest.raises(InfrastructureFailure):
        session.dispatch({'action': 'advance', 'day': 4 * TICK_DAY})


def test_malformed_requests_end_the_server_with_a_protocol_error(tmp_path):
    spec, _ = site_spec(tmp_path, 'a')
    out = io.StringIO()
    code = serve(spec, io.StringIO('{"protocol": "other", "id": 1, "op": "hello"}\n'), out)
    assert code == 2 and json.loads(out.getvalue())['error']['kind'] == 'protocol_error'
    out = io.StringIO()
    code = serve(spec, io.StringIO(canonical({'protocol': PROTOCOL, 'id': 2, 'op': 'hello'}) + '\n'), out)
    assert code == 2


def test_sites_must_start_on_1_january(tmp_path):
    from slowlab.agent_client import CampaignProcess
    spec, folder = site_spec(tmp_path, 'a', origin='2016-06-30T23:00:00+00:00')
    with pytest.raises(InfrastructureFailure):
        CampaignProcess(spec, private_log=folder / 'server.log')
    assert '1 January' in (folder / 'server.log').read_text()
