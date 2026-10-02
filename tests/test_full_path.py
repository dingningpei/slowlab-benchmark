"""Phase 4 full-path test: the pilot entry point (scripts/run_campaign_job.py) with stand-in models on the toy backend."""
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = 300 / 86400


def job(tmp_path, name, model, backend='fake'):
    contract = copy.deepcopy(json.loads((ROOT / 'configs/task_contract_v8.json').read_text()))
    contract['budget'].update(campaign_days=26 * T, crop_days=12 * T, cleanup_days=T, latest_start_day=13 * T)
    (tmp_path / 'c.json').write_text(json.dumps(contract))
    spec = {'method': 'llm', 'llm': {'model': model}, 'tool_seed': 3, 'year': 2017, 'feedback': None,
            'site': {'distribution': 'configs/site_distribution_v1.json', 'master_seed': 20260930, 'site_index': 2},
            'contract': str(tmp_path / 'c.json'), 'backend': backend,
            'private_dir': str(tmp_path / name / 'private'), 'out': str(tmp_path / name / 'out.json')}
    (tmp_path / f'{name}.json').write_text(json.dumps(spec))
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/run_campaign_job.py'), str(tmp_path / f'{name}.json')],
                          capture_output=True, text=True, cwd=ROOT, env={'PYTHONPATH': str(ROOT / 'tests')})
    return proc.returncode, json.loads((tmp_path / name / 'out.json').read_text())


def test_every_action_tool_and_recoverable_mistake(tmp_path):
    code, out = job(tmp_path, 'ok', 'python:full_path_models:Exerciser')
    assert code == 0 and out['status'] == 'completed'
    full, endpoint = (out['branches'][m] for m in ('full', 'endpoint'))
    for branch in (full, endpoint):
        counts = branch['summary']['counts']
        assert counts['recommended'] and counts['format_errors'] >= 1 and counts['invalid_actions'] >= 1
        assert counts['tool_errors'] >= 1 and counts['notes_updates'] >= 1
        actions = {e['request']['action'] for e in branch['public_transcript'] if e['ok']}
        assert actions == {'start', 'advance', 'observe', 'stop', 'recommend'}
        settlement = json.loads(Path(branch['settlement']).read_text())
        assert settlement['recommendation_fallback'] is False and settlement['starts'] == 3
    # Full reads records and daily summaries; Endpoint's science reads are refused (invalid actions)
    assert full['summary']['counts']['records_received'] > 0 and full['summary']['counts']['daily_rows_received'] > 0
    assert endpoint['summary']['counts']['records_received'] == 0
    assert endpoint['summary']['counts']['invalid_actions'] > full['summary']['counts']['invalid_actions']
    # the shared day-0 design is identical in both branches
    shared = [e['request'] for e in full['public_transcript']][:2]
    assert shared == [e['request'] for e in endpoint['public_transcript']][:2]
    assert out['outbound_audit']['model_calls'] > 0 and Path(tmp_path / 'ok' / 'out.audit.jsonl').is_file()


def test_provider_outage_is_an_infrastructure_failure(tmp_path):
    code, out = job(tmp_path, 'outage', 'python:full_path_models:Broken')
    assert code == 3 and out['status'] == 'failed' and 'ProviderError' in out['error']


def test_a_failing_model_backend_is_an_infrastructure_failure(tmp_path):
    code, out = job(tmp_path, 'backend', 'python:full_path_models:Exerciser', backend='fake_failing')
    assert code == 3 and out['status'] == 'failed'


def test_test_models_are_refused_on_the_real_backend(tmp_path):
    code, out = job(tmp_path, 'real', 'python:full_path_models:Exerciser', backend='greenlight')
    assert code == 3 and 'toy backend' in out['error']
