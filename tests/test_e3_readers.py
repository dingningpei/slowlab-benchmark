import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from slowlab.agent_client import CampaignProcess
from slowlab.commitment import make_commitment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import e3_readers  # noqa: E402

POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())


def fake_transcript(tmp_path):
    from slowlab.bo_agent import BOConfig, GPBOAgent
    T = 300 / 86400
    c = copy.deepcopy(json.loads((ROOT / 'configs/task_contract_v8.json').read_text()))
    c['budget'].update(campaign_days=26 * T, crop_days=12 * T, cleanup_days=T, latest_start_day=13 * T)
    (tmp_path / 'c.json').write_text(json.dumps(c))
    spec = {'backend': 'fake', 'contract': str(tmp_path / 'c.json'), 'feedback_mode': 'full',
            'fallback_policy': POLICIES['policy_a'], 'origin_utc': '2016-12-31T23:00:00+00:00', 'soil_boundary_c': None,
            'trace': None, 'settlement_out': str(tmp_path / 's.json'), 'failure_out': str(tmp_path / 'f.json')}
    (tmp_path / 'site.json').write_text(json.dumps(spec))
    with CampaignProcess(tmp_path / 'site.json', private_log=tmp_path / 'log') as campaign:
        GPBOAgent(campaign.session, 3, BOConfig()).run()
        transcript = campaign.session.transcript
        campaign.close()
    return transcript


def test_prepare_and_gp_reader_on_a_committed_sample(tmp_path):
    transcript = fake_transcript(tmp_path)
    jobs, out_dir = [], tmp_path / 'campaigns'
    out_dir.mkdir()
    for s in range(20):
        site = {'distribution': 'configs/site_distribution_v1.json', 'master_seed': 20260930, 'site_index': s}
        jid = f'site{s:02d}_pbo_s0_full'
        jobs.append({'id': jid, 'method': 'bo', 'feedback': 'full', 'site': site, 'year': 2017,
                     'evaluation_years': [2014, 2018, 2020]})
        (out_dir / f'{jid}.json').write_text(json.dumps({'status': 'completed', 'public_transcript': transcript}))
    (tmp_path / 'campaigns.json').write_text(json.dumps({'out_dir': str(out_dir), 'jobs': jobs}))
    public, opening = make_commitment({'seed': 12345}, label='e3-packet-sample-v1')
    (tmp_path / 'opening.json').write_text(json.dumps(opening))
    # a wrong commitment must stop the sample
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/e3_readers.py'), 'prepare', '--campaigns',
                           str(tmp_path / 'campaigns.json'), '--seed-opening', str(tmp_path / 'opening.json'),
                           '--out', str(tmp_path / 'e3')], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode != 0 and 'does not match' in proc.stderr + proc.stdout
    (tmp_path / 'commitment.json').write_text(json.dumps(public))
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/e3_readers.py'), 'prepare', '--campaigns',
                           str(tmp_path / 'campaigns.json'), '--seed-opening', str(tmp_path / 'opening.json'),
                           '--commitment', str(tmp_path / 'commitment.json'), '--out', str(tmp_path / 'e3')],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    sample = json.loads((tmp_path / 'e3' / 'sample.json').read_text())
    assert len(sample['packets']) == 16 and len({p['source_job'] for p in sample['packets']}) == 16
    packet = json.loads((tmp_path / 'e3' / 'packets' / 'p000.json').read_text())
    assert 'pbo' not in json.dumps(packet['packet']) and packet['task']['economics']
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/e3_readers.py'), 'gp', '--out', str(tmp_path / 'e3')],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    gp = json.loads((tmp_path / 'e3' / 'gp_reader.json').read_text())
    assert len(gp) == 16 and all(r['basis'] == 'max_posterior_scored_mean_tried' for r in gp.values())


def test_one_shot_reader_retries_and_validates():
    task = {'policy': json.loads((ROOT / 'configs/task_contract_v8.json').read_text())['policy']}
    good = {k: (s['min'] + s['max']) / 2 for k, s in task['policy']['fields'].items()}
    good['night_temperature_c'] = min(good['night_temperature_c'], good['day_temperature_c'])
    replies = iter(['not json', json.dumps({'type': 'recommendation', 'policy': good, 'reason': 'x'})])
    policy, log = e3_readers.one_shot_reader(task, {'crops': []}, lambda m: next(replies))
    assert policy == good and [e['ok'] for e in log] == [False, True]
    policy, log = e3_readers.one_shot_reader(task, {'crops': []}, lambda m: 'nothing')
    assert policy is None and len(log) == 3
    text = e3_readers.reader_messages(task, {'crops': []})[0]['content']
    assert 'EVIDENCE' in text and 'pbo' not in text
