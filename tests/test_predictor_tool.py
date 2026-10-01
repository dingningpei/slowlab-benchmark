import json
from pathlib import Path

import pytest

from slowlab.agent_client import CampaignProcess
from slowlab.agent_protocol import canonical
from slowlab.prompt_firewall import assert_outbound_messages_safe, load_blinding_policy
from slowlab.tools import CATALOG, ToolError, Toolbox

ROOT = Path(__file__).resolve().parents[1]
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
CHANNELS = ('cumulative_harvest_fresh_equivalent', 'canopy_lai_proxy')
RESOURCES = ('heating_energy', 'lighting_energy', 'co2_dosed')


@pytest.fixture
def campaign(tmp_path):
    spec = {'backend': 'fake', 'contract': str(ROOT / 'configs/task_contract_v6.json'), 'feedback_mode': 'full',
            'fallback_policy': POLICIES['policy_a'], 'origin_utc': '2016-12-31T23:00:00+00:00', 'soil_boundary_c': None,
            'trace': None, 'settlement_out': str(tmp_path / 's.json'), 'failure_out': str(tmp_path / 'f.json')}
    (tmp_path / 'site.json').write_text(json.dumps(spec))
    with CampaignProcess(tmp_path / 'site.json', private_log=tmp_path / 'log') as c:
        yield c


def observe_now(session, unit, channels, day):
    for ch in channels:
        session.dispatch({'action': 'observe', 'unit': unit, 'variable': ch, 'start_day': day, 'end_day': day})


def test_prediction_from_harvest_and_canopy_then_all_readings(campaign):
    s = campaign.session
    tools = Toolbox(s, 1)
    s.dispatch({'action': 'start', 'unit': 0, 'policy': POLICIES['policy_a']})
    s.dispatch({'action': 'advance', 'day': 5})
    observe_now(s, 0, CHANNELS, 5)
    with pytest.raises(ToolError, match='10 days'):
        tools.call('predict_crop_outcome', {'unit': 0, 'run_index': 1})
    s.dispatch({'action': 'advance', 'day': 40})
    observe_now(s, 0, CHANNELS[1:], 40)
    with pytest.raises(ToolError, match='within one day'):
        tools.call('predict_crop_outcome', {'unit': 0, 'run_index': 1})
    observe_now(s, 0, CHANNELS[:1], 40)
    two = tools.call('predict_crop_outcome', {'unit': 0, 'run_index': 1})['result']
    assert two['inputs_used'] == 'harvest_lai' and two['reading_day_since_planting'] == 40
    assert set(two['predicted_final_per_m2']) == {'harvest_kg_m2', 'heat_kwh_m2', 'light_kwh_m2', 'co2_kg_m2'}
    assert two['typical_error']['margin_eur_m2'] > 0
    observe_now(s, 0, RESOURCES, 40)
    five = tools.call('predict_crop_outcome', {'unit': 0, 'run_index': 1})['result']
    assert five['inputs_used'] == 'all'
    assert five['predicted_final_per_m2']['harvest_kg_m2'] >= 0
    for bad in ({'unit': 0}, {'unit': 0, 'run_index': 2}, {'unit': 3, 'run_index': 1}):
        with pytest.raises(ToolError):
            tools.call('predict_crop_outcome', bad)


def test_later_planting_needs_a_planting_day_reading_and_closed_crops_are_refused(campaign):
    s = campaign.session
    tools = Toolbox(s, 1)
    s.dispatch({'action': 'start', 'unit': 1, 'policy': POLICIES['policy_b']})
    s.dispatch({'action': 'advance', 'day': 182})
    s.dispatch({'action': 'observe', 'unit': 1, 'run_index': 1})
    s.dispatch({'action': 'start', 'unit': 1, 'policy': POLICIES['policy_a']})
    s.dispatch({'action': 'advance', 'day': 200})
    observe_now(s, 1, CHANNELS, 200)
    with pytest.raises(ToolError, match='planting day'):
        tools.call('predict_crop_outcome', {'unit': 1, 'run_index': 2})
    observe_now(s, 1, CHANNELS[:1], 182)
    result = tools.call('predict_crop_outcome', {'unit': 1, 'run_index': 2})['result']
    assert result['reading_day_since_planting'] == pytest.approx(18) and result['inputs_used'] == 'harvest_lai'
    with pytest.raises(ToolError, match='closed'):
        tools.call('predict_crop_outcome', {'unit': 1, 'run_index': 1})


def test_tool_text_passes_the_blinding_firewall():
    blinding = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
    assert_outbound_messages_safe([{'role': 'user', 'content': canonical(CATALOG['predict_crop_outcome'])}], blinding)
    assert Toolbox.catalog()['version'] == 'analysis-tools-v2'
