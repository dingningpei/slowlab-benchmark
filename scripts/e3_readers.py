#!/usr/bin/env python3
"""E3: history packets, Readers and their evaluation batch (decisions 2026-10-01 and 2026-10-03).

    e3_readers.py prepare     --campaigns RUN/campaigns.json --seed-opening OPENING.json --out E3
    e3_readers.py gp          --out E3
    e3_readers.py llm         --out E3 --ledger RUN/ledger.jsonl [--env-file KEYS] [--workers 24]
    e3_readers.py evaluations --campaigns RUN/campaigns.json --out E3 --dev-cache C --formal-cache F

prepare: the committed E3 seed (opening verified against configs/seed_commitment_e3_sample_v1.json)
draws 16 of the 48 test sites per method (stream: method name) without replacement; each packet is
the history packet of that site's repeat-0 Full campaign (main baseline: seed 0, Full), built from
the public transcript and the site's public task view. Packets carry no method name.
gp: the GP Reader (gp-reader-v2) on every packet. llm: each formal model as a one-shot Reader on
every packet (prompt below; three attempts; an unusable reply falls back to the frozen default
policy, flagged), spend booked in the formal ledger. evaluations: an evaluation batch for every
Reader recommendation on the packet site's three evaluation years. The source method's own
recommendation is already evaluated in the formal run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.agent_protocol import canonical, public_task_view  # noqa: E402
from slowlab.commitment import verify  # noqa: E402
from slowlab.history_packet import build_packet, gp_reader  # noqa: E402
from slowlab.llm_agent import FormatError, validate_policy_against_view  # noqa: E402

PER_METHOD = 16
READER_PROMPT_VERSION = 'e3-reader-prompt-v1'


def reader_messages(task: dict, packet: dict) -> list:
    system = ('You are a greenhouse grower. Someone else ran one year of experiments in four greenhouse '
              'compartments under the task below; you did not run them. Using only the task description and '
              'the evidence, recommend the management policy you expect to score best under the scoring rule.'
              '\n\nTASK\n' + canonical(task) + '\n\nEVIDENCE\n' + canonical(packet) +
              '\n\nReply with exactly one JSON object: {"type": "recommendation", "policy": {...}, "reason": "..."}')
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': 'Give your recommendation now.'}]


def parse_reader_reply(text: str) -> dict:
    """One JSON object of type "recommendation" with a "policy" object (code fences allowed)."""
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = stripped.strip('`')
        stripped = stripped[stripped.find('\n') + 1:] if '\n' in stripped else stripped
    start, end = stripped.find('{'), stripped.rfind('}')
    if start < 0 or end <= start:
        raise FormatError('no JSON object found: reply with one JSON object only')
    try:
        reply = json.loads(stripped[start:end + 1])
    except json.JSONDecodeError as exc:
        raise FormatError(f'invalid JSON: {exc.msg}') from None
    if not isinstance(reply, dict) or reply.get('type') != 'recommendation' or not isinstance(reply.get('policy'), dict):
        raise FormatError('the reply needs "type": "recommendation" and a "policy" object')
    return reply


def one_shot_reader(task, packet, complete, attempts=3):
    messages = reader_messages(task, packet)
    log = []
    for _ in range(attempts):
        text = complete(messages)
        try:
            reply = parse_reader_reply(text)
            validate_policy_against_view(task, reply['policy'])
        except ValueError as exc:
            log.append({'ok': False, 'error': str(exc)})
            messages = messages[:2] + [{'role': 'assistant', 'content': text[:2000]},
                                       {'role': 'user', 'content': f'That reply was not usable: {exc}. '
                                                                   'Reply again with one JSON object.'}]
            continue
        log.append({'ok': True})
        return dict(reply['policy']), log
    return None, log


def site_task(job, contract):
    from slowlab.private_runs import prepare
    from slowlab.site_parameters import site_contract
    p = prepare({'contract': 'configs/task_contract_v8.json', 'year': job['year'], 'site': job['site'],
                 'backend': 'fake'})
    return public_task_view(site_contract(contract, p['site']), 'full')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('prepare', 'gp', 'llm', 'evaluations'))
    ap.add_argument('--campaigns', type=Path)
    ap.add_argument('--seed-opening', type=Path)
    ap.add_argument('--commitment', type=Path, default=ROOT / 'configs/seed_commitment_e3_sample_v1.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--ledger', type=Path)
    ap.add_argument('--env-file')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--dev-cache')
    ap.add_argument('--formal-cache')
    args = ap.parse_args()
    out = args.out.resolve()
    contract = json.loads((ROOT / 'configs/task_contract_v8.json').read_text())
    if args.stage == 'prepare':
        opening = json.loads(args.seed_opening.read_text())
        if not verify(json.loads(args.commitment.read_text()), opening):
            raise SystemExit('E3 seed opening does not match its commitment')
        seed = int(opening['secret']['seed'])
        batch = json.loads(args.campaigns.read_text())
        by_id = {j['id']: j for j in batch['jobs']}
        sites = sorted({j['site']['site_index'] for j in batch['jobs']})
        models = sorted({j['model_name'] for j in batch['jobs'] if j['method'] == 'llm'})
        packets = []
        for method in models + ['pbo']:
            stream = int.from_bytes(hashlib.sha256(method.encode()).digest()[:8], 'big')
            chosen = sorted(np.random.default_rng([seed, stream]).choice(sites, PER_METHOD, replace=False).tolist())
            for s in chosen:
                job_id = f'site{s:02d}_{method}_r0' if method != 'pbo' else f'site{s:02d}_pbo_s0_full'
                job = by_id[job_id]
                record = json.loads((Path(batch['out_dir']) / f'{job_id}.json').read_text())
                transcript = (record['branches']['full']['public_transcript'] if method != 'pbo'
                              else record['public_transcript'])
                task = site_task(job, contract)
                packets.append({'packet_id': f'p{len(packets):03d}', 'source_method': method, 'source_job': job_id,
                                'site': job['site'], 'evaluation_years': job['evaluation_years'],
                                'task': task, 'packet': build_packet(task, transcript)})
        (out / 'packets').mkdir(parents=True, exist_ok=True)
        for p in packets:
            (out / 'packets' / f"{p['packet_id']}.json").write_text(json.dumps(p, indent=1))
        (out / 'sample.json').write_text(json.dumps({'per_method': PER_METHOD, 'packets': [
            {k: p[k] for k in ('packet_id', 'source_method', 'source_job')} for p in packets]}, indent=1))
        print(json.dumps({'packets': len(packets)}))
        return
    packets = [json.loads(f.read_text()) for f in sorted((out / 'packets').glob('p*.json'))]
    if args.stage == 'gp':
        config = json.loads((ROOT / 'configs/prior_bo_v1.json').read_text())
        config['anchor_policy'] = json.loads((ROOT / 'configs/fixed_reference_v1.json').read_text())['policy']
        recs = {p['packet_id']: gp_reader(p['task'], p['packet'], config) for p in packets}
        (out / 'gp_reader.json').write_text(json.dumps(recs, indent=1))
        print(json.dumps({'gp_recommendations': len(recs)}))
        return
    if args.stage == 'llm':
        import os
        from slowlab.outbound_audit import AuditedCompleter
        from slowlab.prompt_firewall import load_blinding_policy
        from slowlab.providers import load_dotenv, openai_compatible
        from slowlab.spend import records_cost
        if args.env_file:
            os.environ['SLOWLAB_ENV_FILE'] = args.env_file
        load_dotenv()
        models = json.loads((ROOT / 'configs/formal_models_v1.json').read_text())['models']
        default = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())['policy_a']
        blinding = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
        (out / 'llm').mkdir(exist_ok=True)

        def read(job):
            model, p = job
            path = out / 'llm' / f"{model['name']}_{p['packet_id']}.json"
            if path.exists():
                return json.loads(path.read_text())
            llm = model['llm']
            seed = 29000 + int(p['packet_id'][1:])
            provider = openai_compatible(llm['model'], temperature=llm['temperature'], max_tokens=llm['max_tokens'],
                                         generation_seed=seed, provider_only=llm.get('provider_only'),
                                         expected_provider=llm.get('expected_provider'))
            audit = out / 'llm' / f"{model['name']}_{p['packet_id']}.audit.jsonl"
            try:
                policy, log = one_shot_reader(p['task'], p['packet'],
                                              AuditedCompleter(provider, blinding, audit, label='e3_reader'))
                status = 'completed'
            except Exception as exc:  # infrastructure failure: recorded, retried by rerunning the stage
                policy, log, status = None, [{'ok': False, 'error': f'{type(exc).__name__}: {exc}'}], 'failed'
            calls = getattr(provider, 'call_records', [])
            rec = {'status': status, 'reader': model['name'], 'prompt_version': READER_PROMPT_VERSION,
                   'packet_id': p['packet_id'], 'policy': policy if policy is not None else default,
                   'frozen_default': policy is None, 'log': log, 'usd': records_cost(calls, model['price']),
                   'calls': len(calls), 'provider_call_records': calls}
            if status == 'completed':
                path.write_text(json.dumps(rec, indent=1))
            with args.ledger.open('a') as f:
                f.write(json.dumps({'kind': 'e3_reader', 'job': path.stem, 'model': model['name'],
                                    'status': status, 'usd': rec['usd'], 'calls': len(calls)}) + '\n')
            return rec
        jobs = [(m, p) for m in models for p in packets]
        with ThreadPoolExecutor(args.workers) as pool:
            results = list(pool.map(read, jobs))
        print(json.dumps({'readers': len(results), 'failed': sum(r['status'] != 'completed' for r in results),
                          'frozen_default': sum(r['frozen_default'] for r in results),
                          'usd': round(sum(r['usd'] for r in results), 4)}))
        return
    from slowlab.private_runs import weather_spec
    gp = json.loads((out / 'gp_reader.json').read_text())
    jobs = []
    for p in packets:
        recs = [('gp-reader', gp[p['packet_id']]['policy'])]
        for f in sorted((out / 'llm').glob(f"*_{p['packet_id']}.json")):
            r = json.loads(f.read_text())
            recs.append((r['reader'], r['policy']))
        for reader, policy in recs:
            for y in p['evaluation_years']:
                jobs.append({'id': f"{p['packet_id']}_{reader}_y{y}", 'site': p['site'], 'year': y, 'policy': policy,
                             'weather': weather_spec(y, args.dev_cache, args.formal_cache)})
    path = out / 'evaluations.json'
    path.write_text(json.dumps({'out_dir': str(out / 'evaluations'), 'purpose': 'E3 Reader evaluations (private)',
                                'common': {'contract': 'configs/task_contract_v8.json', 'backend': 'greenlight',
                                           'sensor_noise': {'config': 'configs/sensor_noise_v0.json', 'setting': 'main'}},
                                'jobs': jobs}, indent=1))
    path.chmod(0o600)
    print(json.dumps({'evaluation_jobs': len(jobs)}))


if __name__ == '__main__':
    main()
