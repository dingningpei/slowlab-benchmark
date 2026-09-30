import gzip
import json
from pathlib import Path

import pytest

from slowlab.commitment import make_commitment, verify
from slowlab.leak_check import private_needles, scan
from slowlab.prompt_firewall import load_blinding_policy
from slowlab.site_distribution import sample_site

ROOT = Path(__file__).resolve().parents[1]
POLICY = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
DIST = json.loads((ROOT / 'configs/site_distribution_v0.json').read_text())


def test_commitment_round_trip_and_tamper():
    public, opening = make_commitment({'master_seed': 12345, 'sites': [1, 2]}, label='test-v0')
    assert verify(public, opening)
    assert '12345' not in json.dumps(public)
    for change in ({'secret': {'master_seed': 12346, 'sites': [1, 2]}}, {'label': 'other'},
                   {'salt_hex': '00' * 32}):
        assert not verify(public, {**opening, **change})
    a, _ = make_commitment({'s': 1}, label='x')
    b, _ = make_commitment({'s': 1}, label='x')
    assert a['commitment'] != b['commitment']
    with pytest.raises(ValueError):
        make_commitment({'s': 1}, label='x', salt=b'short')


def spec_for(tmp_path):
    s = sample_site(DIST, 99, 0)
    return {'site': s['site'], 'soil_boundary_c': s['soil_boundary_c'], 'origin_utc': '2014-12-31T23:00:00+00:00',
            'sensor_noise': {'seed': s['sensor_noise_seed']}, 'trace': str(tmp_path / 'private' / 'trace.jsonl.gz'),
            'weather': {'kind': 'formal', 'cache': '/data/weather-formal', 'audit': 'results/formal_audit.json'}}, s


def test_clean_public_output_passes(tmp_path):
    spec, _ = spec_for(tmp_path)
    public = tmp_path / 'public'
    public.mkdir()
    (public / 'run.json').write_text(json.dumps({'transcript': [{'day': 12.5, 'margin': 31.27}], 'price': 0.2143}))
    result = scan([public], private_needles(spec, POLICY), [tmp_path / 'private'])
    assert result['status'] == 'pass', result['findings']


@pytest.mark.parametrize('leak, kind', [
    ('the air is from Cabauw', 'data_source'),
    ('GreenLight model', 'blinding_term'),
    ('noise seed {seed}', 'noise_seed'),
    ('rgFruit={rg!r}', 'site_value'),
    ('soil {soil:.6g} C', 'site_value'),
])
def test_each_kind_of_leak_is_caught(tmp_path, leak, kind):
    spec, s = spec_for(tmp_path)
    text = leak.format(seed=s['sensor_noise_seed'], rg=s['site']['unit_parameters']['2']['rgFruit'],
                       soil=s['soil_boundary_c'])
    public = tmp_path / 'public'
    public.mkdir()
    with gzip.open(public / 'audit.jsonl.gz', 'wt') as f:
        f.write(json.dumps({'content': text}) + '\n')
    result = scan([public], private_needles(spec, POLICY))
    assert result['status'] == 'fail' and kind in {f['kind'] for f in result['findings']}


def test_file_names_year_and_layout(tmp_path):
    spec, _ = spec_for(tmp_path)
    public = tmp_path / 'public'
    (public / 'private').mkdir(parents=True)
    (public / 'formal_audit.json').write_text('{}')
    (public / 'notes.txt').write_text('season 2015 looked warm')
    (public / 'figure.png').write_bytes(b'\x89PNG')
    result = scan([public], private_needles(spec, POLICY), [public / 'private'])
    kinds = {(f['kind'], f['level']) for f in result['findings']}
    assert ('private_path', 'error') in kinds and ('weather_year', 'warning') in kinds
    assert ('layout', 'error') in kinds and ('unscanned_file', 'warning') in kinds


def test_a_real_scripted_campaign_leaks_nothing(tmp_path):
    import copy
    from slowlab.agent_client import CampaignProcess
    from slowlab.llm_agent import LLMCampaignAgent
    from slowlab.outbound_audit import AuditedCompleter
    from slowlab.scripted_model import ScriptedModel
    from slowlab.tools import Toolbox
    tick = 300 / 86400
    contract = copy.deepcopy(json.loads((ROOT / 'configs/task_contract_v5.json').read_text()))
    contract['budget'].update(campaign_days=26 * tick, crop_days=12 * tick, cleanup_days=tick, latest_start_day=13 * tick)
    private, public = tmp_path / 'private', tmp_path / 'public'
    private.mkdir()
    public.mkdir()
    (private / 'contract.json').write_text(json.dumps(contract))
    s = sample_site(DIST, 4242, 7)
    spec = {'backend': 'fake', 'contract': str(private / 'contract.json'), 'feedback_mode': 'full',
            'fallback_policy': json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())['policy_a'],
            'origin_utc': '2016-12-31T23:00:00+00:00', 'soil_boundary_c': s['soil_boundary_c'], 'site': s['site'],
            'sensor_noise': {'seed': s['sensor_noise_seed']}, 'trace': None,
            'settlement_out': str(private / 'settlement.json'), 'failure_out': str(private / 'failure.json')}
    (private / 'site.json').write_text(json.dumps(spec))
    with CampaignProcess(private / 'site.json', private_log=private / 'server.log') as campaign:
        complete = AuditedCompleter(ScriptedModel(campaign.session.task), POLICY, public / 'run.audit.jsonl')
        summary = LLMCampaignAgent(campaign.session, Toolbox(campaign.session, 3), complete).run()
        (public / 'run.json').write_text(json.dumps({'agent': summary, 'public_transcript': campaign.session.transcript,
                                                     'task': campaign.session.task}))
        campaign.close()
    assert summary['counts']['recommended']
    result = scan([public], private_needles(spec, POLICY), [private])
    assert result['status'] == 'pass', result['findings'][:3]
    assert json.loads((private / 'settlement.json').read_text())['unit_parameters'] == s['site']['unit_parameters']
