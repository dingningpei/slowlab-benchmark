import copy
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = 300 / 86400


def setup(tmp_path, cap, reserve=None, model='Billed', extra=()):
    contract = copy.deepcopy(json.loads((ROOT / 'configs/task_contract_v8.json').read_text()))
    contract['budget'].update(campaign_days=26 * T, crop_days=12 * T, cleanup_days=T, latest_start_day=13 * T)
    (tmp_path / 'c.json').write_text(json.dumps(contract))
    models = {'models': [{'name': 'billed', 'llm': {'model': f'python:full_path_models:{model}', 'max_tokens': 2048},
                          'price': {'input': 0.001, 'output': 0.001}}], 'budget': {'hard_cap_usd': cap}}
    if reserve is not None:
        models['models'][0]['reserve_usd'] = reserve
    (tmp_path / 'models.json').write_text(json.dumps(models))
    site = {'distribution': 'configs/site_distribution_v1.json', 'master_seed': 20260930, 'site_index': 0}
    jobs = [{'id': f'llm{i}', 'method': 'llm', 'model_name': 'billed', 'llm': models['models'][0]['llm'],
             'tool_seed': i, 'site': site, 'year': 2017, 'feedback': None, 'private_dir': str(tmp_path / 'p' / f'llm{i}')}
            for i in range(2)]
    jobs.append({'id': 'bo0', 'method': 'bo', 'bo': {'schedule': 'two_waves', 'seed': 1}, 'site': site, 'year': 2017,
                 'feedback': 'full', 'private_dir': str(tmp_path / 'p' / 'bo0')})
    batch = {'out_dir': str(tmp_path / 'out'), 'script': 'scripts/run_campaign_job.py',
             'common': {'contract': str(tmp_path / 'c.json'), 'backend': 'fake'}, 'jobs': jobs}
    (tmp_path / 'batch.json').write_text(json.dumps(batch))
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'tests'))
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/run_pilot.py'), str(tmp_path / 'batch.json'),
                           '--models', str(tmp_path / 'models.json'), '--ledger', str(tmp_path / 'ledger.jsonl'),
                           '--workers', '2', *extra], capture_output=True, text=True, cwd=ROOT, env=env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads((tmp_path / 'out' / 'DONE').read_text()), tmp_path / 'ledger.jsonl'


def test_campaigns_run_and_are_booked_within_the_cap(tmp_path):
    done, ledger = setup(tmp_path, cap=0.006)  # one worst-case reserve (~0.0046) at a time
    assert done['status'] == {'llm0': 'completed', 'llm1': 'completed', 'bo0': 'completed'} and not done['skipped_budget']
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [r['job'] for r in rows] == ['llm0', 'llm1'] and all(r['usd'] > 0 and r['calls'] > 0 for r in rows)
    assert done['spent_usd'] == sum(r['usd'] for r in rows) <= 0.006


def test_unaffordable_campaigns_are_skipped_and_bo_still_runs(tmp_path):
    done, ledger = setup(tmp_path, cap=0.004)
    assert sorted(done['skipped_budget']) == ['llm0', 'llm1'] and done['status']['bo0'] == 'completed'
    assert done['spent_usd'] == 0 and not ledger.exists()


def test_costs_use_billed_amounts_cache_hits_or_token_prices():
    from slowlab.spend import call_cost, worst_case_campaign_cost
    deepseek = {'input': 0.30, 'input_cache_hit': 0.006, 'output': 1.20}
    assert call_cost({'usage': {'cost': 0.0123, 'prompt_tokens': 10 ** 6}}, deepseek) == 0.0123
    rec = {'usage': {'prompt_tokens': 1_000_000, 'prompt_cache_hit_tokens': 600_000, 'prompt_cache_miss_tokens': 400_000,
                     'completion_tokens': 100_000}}
    assert abs(call_cost(rec, deepseek) - (0.6 * 0.006 + 0.4 * 0.30 + 0.1 * 1.20)) < 1e-12
    assert abs(call_cost({'usage': {'prompt_tokens': 10 ** 6, 'completion_tokens': 0}}, {'input': 0.1, 'output': 0.5}) - 0.1) < 1e-12
    worst = worst_case_campaign_cost(deepseek, max_input_chars_per_branch=5_000_000, max_llm_calls_per_branch=200,
                                     max_tokens=2048)
    assert 2.4 < worst < 2.6


def test_a_configured_reserve_replaces_the_worst_case(tmp_path):
    # the worst case (~0.0046) does not fit under 0.004, a configured reserve of 0.001 does
    done, ledger = setup(tmp_path, cap=0.004, reserve=0.001)
    assert not done['skipped_budget'] and done['status']['llm0'] == done['status']['llm1'] == 'completed'


def test_failures_above_the_share_pause_the_batch(tmp_path):
    done, _ = setup(tmp_path, cap=1.0, model='Broken', extra=('--pause-failure-share', '0.1', '--pause-min-finished', '1'))
    assert done['paused'] and done['not_started']
    assert sum(s is not None for s in done['status'].values()) < 3
