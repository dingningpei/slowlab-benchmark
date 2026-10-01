import copy
import json
from pathlib import Path

import pytest

from slowlab.agent_client import CampaignProcess
from slowlab.bo_agent import BOConfig, GPBOAgent

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'configs/task_contract_v6.json').read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
TICK_DAY = 300 / 86400


def run_bo(tmp_path, name, *, mode='full', seed=3, config=BOConfig(), real_calendar=False):
    """Real-calendar structure at 1/15 scale (crop 12 ticks, cleanup 1, latest start 13, campaign 26),
    or the real calendar, which the process predictor needs (readings at least 10 days into a crop)."""
    folder = tmp_path / name
    folder.mkdir()
    contract = copy.deepcopy(CONTRACT)
    if not real_calendar:
        contract['budget'].update(campaign_days=26 * TICK_DAY, crop_days=12 * TICK_DAY,
                                  cleanup_days=1 * TICK_DAY, latest_start_day=13 * TICK_DAY)
    (folder / 'contract.json').write_text(json.dumps(contract))
    spec = {'backend': 'fake', 'contract': str(folder / 'contract.json'), 'feedback_mode': mode,
            'fallback_policy': POLICIES['policy_a'], 'origin_utc': '2016-12-31T23:00:00+00:00',
            'soil_boundary_c': None, 'trace': None, 'settlement_out': str(folder / 'settlement.json'),
            'failure_out': str(folder / 'failure.json')}
    (folder / 'site.json').write_text(json.dumps(spec))
    with CampaignProcess(folder / 'site.json', private_log=folder / 'server.log') as campaign:
        agent = GPBOAgent(campaign.session, seed, config)
        summary = agent.run()
        transcript = campaign.session.transcript
        assert campaign.close() == 0
    settlement = json.loads((folder / 'settlement.json').read_text())
    return summary, transcript, settlement, agent


def requests(transcript, action):
    return [entry['request'] for entry in transcript if entry['request']['action'] == action]


def test_two_waves_completes_eight_crops_within_budget(tmp_path):
    summary, transcript, settlement, agent = run_bo(tmp_path, 'a')
    starts = [e for e in summary['log'] if e['action'] == 'start']
    assert [e['basis'] for e in starts] == ['space_filling'] * 4 + ['expected_improvement'] * 4
    assert summary['completed_crops'] == 8 and settlement['starts'] == 8
    assert settlement['recommendation_fallback'] is False
    assert settlement['recommendation'] == summary['recommendation']
    budget = CONTRACT['budget']
    assert settlement['decision_calls'] <= budget['max_decision_calls']
    assert settlement['tool_calls'] <= budget['max_tool_calls']
    assert all(entry['ok'] for entry in transcript)
    rec = summary['log'][-1]
    assert rec['basis'] == 'max_posterior_scored_mean'
    assert len(rec['predicted_by_planting_day']) == 2
    assert all(e.get('objective') == 'scored_mean_over_plantings' for e in starts[4:])


def test_same_seed_repeats_and_other_seed_differs(tmp_path):
    a = run_bo(tmp_path, 'a', seed=5)[0]
    b = run_bo(tmp_path, 'b', seed=5)[0]
    c = run_bo(tmp_path, 'c', seed=6)[0]
    assert a == b
    assert a['log'][0]['policy'] != c['log'][0]['policy']


def test_two_waves_behaves_identically_under_both_feedback_conditions(tmp_path):
    full, full_transcript, _, _ = run_bo(tmp_path, 'full', mode='full')
    endpoint, endpoint_transcript, _, _ = run_bo(tmp_path, 'endpoint', mode='endpoint')
    assert full['log'] == endpoint['log']
    assert not any('variable' in r for r in requests(endpoint_transcript, 'observe'))


def test_staggered_uses_process_information_only_under_full_feedback(tmp_path):
    config = BOConfig(schedule='staggered', initial_units=2, stagger_fraction=0.5)
    full, full_transcript, full_settlement, _ = run_bo(tmp_path, 'full', mode='full', config=config, real_calendar=True)
    endpoint, endpoint_transcript, _, _ = run_bo(tmp_path, 'endpoint', mode='endpoint', config=config)
    late_full = [e for e in full['log'] if e['action'] == 'start'][2:4]
    late_endpoint = [e for e in endpoint['log'] if e['action'] == 'start'][2:4]
    assert all(e['basis'] == 'expected_improvement' for e in late_full)
    assert all(e['basis'] == 'space_filling' for e in late_endpoint)
    predicted = [e for e in full['log'] if e['action'] == 'predicted_final_margins']
    assert len(predicted) == 1 and len(predicted[0]['values']) == 2 and all(e > 0 for e in predicted[0]['typical_error'])
    read = {r['variable'] for r in requests(full_transcript, 'observe') if 'variable' in r}
    assert read == {'cumulative_harvest_fresh_equivalent', 'heating_energy', 'lighting_energy', 'co2_dosed',
                    'canopy_lai_proxy'}
    assert any('variable' in r for r in requests(full_transcript, 'observe'))
    assert not any('variable' in r for r in requests(endpoint_transcript, 'observe'))
    # 2 early compartments crop twice; 2 late ones once
    assert full_settlement['starts'] == 6 and full['completed_crops'] == 6
    assert full_settlement['decision_calls'] <= CONTRACT['budget']['max_decision_calls']


def test_invalid_configurations_are_rejected():
    for bad in (dict(schedule='random'), dict(stagger_fraction=1.0), dict(pool_size=8), dict(initial_units=0)):
        with pytest.raises(ValueError):
            BOConfig(**bad)
