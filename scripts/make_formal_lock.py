#!/usr/bin/env python3
"""Hash lock for the formal runs (Phase 4): code, configs, prompts, models, tools, commitments.

    make_formal_lock.py --environment ENV.json --out configs/formal_lock_v1.json
    make_formal_lock.py --verify configs/formal_lock_v1.json      (on a deployed tree)

The lock lists the sha256 of every file under slowlab/ and scripts/, of the configs the formal
runs read, and of the rendered prompts (system message for each feedback view and the fixed
reply format, examples and day-0 instruction), plus method versions, model identities, the
seed commitments and the run environment. ``--verify`` recomputes the file and prompt hashes
of the current tree and fails if any locked file or prompt changed or disappeared; files added
after the lock (later E3, reference-search and sensitivity scripts, locked in turn before they
run) are listed but do not fail the check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CONFIGS = ['configs/task_contract_v8.json', 'configs/site_distribution_v1.json', 'configs/sensor_noise_v0.json',
           'configs/fixed_reference_v1.json', 'configs/prior_bo_v1.json', 'configs/formal_models_v1.json',
           'configs/process_predictor_v0.json', 'configs/seed_commitment_test_v1.json',
           'configs/seed_commitment_e3_sample_v1.json', 'configs/site_partition_v1.json',
           'configs/campaign_example_v0.json', 'configs/weather_gapfilled_plan.json',
           'configs/simulation_blinding_v2_2.json', 'pyproject.toml', 'requirements.txt']


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_hashes() -> dict:
    files = sorted(p for d in ('slowlab', 'scripts') for p in (ROOT / d).rglob('*.py') if '__pycache__' not in p.parts)
    out = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in files}
    for c in CONFIGS:
        out[c] = sha((ROOT / c).read_bytes())
    return out


def prompt_hashes() -> dict:
    from slowlab.agent_protocol import public_task_view
    from slowlab import llm_agent
    contract = json.loads((ROOT / 'configs/task_contract_v8.json').read_text())
    out = {f'system_message_{mode or "shared"}': sha(llm_agent.system_message(public_task_view(contract, mode)).encode())
           for mode in (None, 'full', 'endpoint')}
    for name in ('REPLY_FORMAT', 'REPLY_EXAMPLES', 'DAY_ZERO_INSTRUCTION'):
        out[name] = sha(str(getattr(llm_agent, name)).encode())
    return out


def versions() -> dict:
    from slowlab import history_packet, llm_agent, prior_bo, tools
    return {'llm_agent': llm_agent.LLM_AGENT_VERSION, 'tools': tools.TOOLBOX_VERSION,
            'main_baseline': prior_bo.PRIOR_BO_VERSION, 'gp_reader': history_packet.GP_READER_VERSION,
            'history_packet': history_packet.PACKET_FORMAT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--environment', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--verify', type=Path)
    args = ap.parse_args()
    if args.verify:
        lock = json.loads(args.verify.read_text())
        files, prompts = tree_hashes(), prompt_hashes()
        bad = sorted(k for k in lock['files'] if lock['files'][k] != files.get(k))
        added = sorted(set(files) - set(lock['files']))
        bad += sorted(k for k in lock['prompts'] if lock['prompts'][k] != prompts.get(k))
        if versions() != lock['versions']:
            bad.append('versions')
        print(json.dumps({'lock': lock['lock_id'], 'match': not bad, 'differences': bad, 'added_files': added}))
        raise SystemExit(1 if bad else 0)
    models = json.loads((ROOT / 'configs/formal_models_v1.json').read_text())
    lock = {'lock_id': 'slowlab-formal-lock-v1', 'date': '2026-10-04',
            'decision': 'RESEARCH_PLAN decisions 2026-10-04 (N = 48, budget USD 30, pause rules)',
            'scope': ('formal E1/E2 campaigns and their evaluations; E3, best-known reference search and the boundary '
                      'sensitivity reuse these files unchanged and add their own scripts under a later lock'),
            'versions': versions(), 'prompts': prompt_hashes(),
            'models': [{'name': m['name'], 'model': m['llm']['model'],
                        'provider': m['llm'].get('expected_provider', 'DeepSeek official API'),
                        'temperature': m['llm']['temperature'], 'max_tokens': m['llm']['max_tokens'],
                        'price': m['price']} for m in models['models']],
            'commitments': {name: json.loads((ROOT / f'configs/{name}.json').read_text())['commitment']
                            for name in ('seed_commitment_test_v1', 'seed_commitment_e3_sample_v1')},
            'design': {'test_sites': 48, 'repeats': 2, 'baseline_seeds': 2, 'evaluation_years_per_site': 3,
                       'hard_cap_usd': models['budget']['hard_cap_usd']},
            'environment': json.loads(args.environment.read_text()) if args.environment else None,
            'files': tree_hashes()}
    args.out.write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps({'files': len(lock['files']), 'prompts': lock['prompts'], 'versions': lock['versions']}, indent=1))


if __name__ == '__main__':
    main()
