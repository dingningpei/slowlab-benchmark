import json
from pathlib import Path

from slowlab.agent_client import CampaignProcess
from slowlab.agent_protocol import canonical
from slowlab.bo_agent import BOConfig, GPBOAgent
from slowlab.history_packet import build_packet, gp_reader
from slowlab.leak_check import private_needles, scan
from slowlab.prompt_firewall import load_blinding_policy

ROOT = Path(__file__).resolve().parents[1]
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())


def test_packet_from_a_full_campaign_and_the_gp_reader(tmp_path):
    import copy
    T = 300 / 86400
    c = copy.deepcopy(json.loads((ROOT / 'configs/task_contract_v8.json').read_text()))
    c['budget'].update(campaign_days=26 * T, crop_days=12 * T, cleanup_days=T, latest_start_day=13 * T)
    (tmp_path / 'c.json').write_text(json.dumps(c))
    spec = {'backend': 'fake', 'contract': str(tmp_path / 'c.json'), 'feedback_mode': 'full',
            'fallback_policy': POLICIES['policy_a'], 'origin_utc': '2016-12-31T23:00:00+00:00', 'soil_boundary_c': 13.7,
            'trace': None, 'settlement_out': str(tmp_path / 's.json'), 'failure_out': str(tmp_path / 'f.json')}
    (tmp_path / 'site.json').write_text(json.dumps(spec))
    with CampaignProcess(tmp_path / 'site.json', private_log=tmp_path / 'log') as campaign:
        GPBOAgent(campaign.session, 3, BOConfig()).run()
        task, transcript = campaign.session.task, campaign.session.transcript
        campaign.close()
    packet = build_packet(task, transcript)
    assert packet['format'] == 'history-packet-v1' and len(packet['crops']) == 8
    assert all('contribution_margin_eur_m2' in c for c in packet['crops'] if c['reason'] == 'normal_completion')
    assert packet == build_packet(task, transcript)
    (tmp_path / 'public').mkdir()
    (tmp_path / 'public' / 'packet.json').write_text(canonical(packet))
    policy = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
    assert scan([tmp_path / 'public'], private_needles(spec, policy))['status'] == 'pass'
    config = json.loads((ROOT / 'configs/prior_bo_v1.json').read_text())
    config['anchor_policy'] = json.loads((ROOT / 'configs/fixed_reference_v1.json').read_text())['policy']
    a, b = gp_reader(task, packet, config), gp_reader(task, packet, config)
    assert a == b and a['basis'] == 'max_posterior_scored_mean_tried' and set(a['policy']) == set(task['policy']['fields'])
    assert a['policy'] in [c['policy'] for c in packet['crops']]
    empty = {**packet, 'crops': []}
    assert gp_reader(task, empty, config)['policy'] == config['anchor_policy']
