#!/usr/bin/env python3
"""Run a campaign batch under an API spend cap (pilot: configs/pilot_models_v3.json; formal:
configs/formal_models_v1.json).

An LLM campaign starts only if spent + the reserve of every running LLM campaign + its own
reserve stays within the cap. The reserve is the worst case (the full harness allowance), so
the cap holds by construction, unless the model config gives a per-campaign ``reserve_usd``
(formal runs, decision 2026-10-04: three times the pilot maximum; there the hard cap is held by
provider-side limits that sum to the cap, and the guard only paces the launches).
With ``--pause-failure-share`` the runner stops launching once more than that share of the
finished jobs failed (after ``--pause-min-finished`` jobs) and records the pause. Spend is booked from each
finished job's per-call records (OpenRouter's billed cost, else tokens x price),
including jobs that failed after making calls, and from earlier ledger entries
(connectivity tests). BO jobs cost nothing and run whenever a worker is free.
Jobs that cannot be afforded are recorded as skipped_budget. Resumable.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.llm_agent import LLMConfig  # noqa: E402
from slowlab.spend import records_cost, worst_case_campaign_cost  # noqa: E402


def ledger_total(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(json.loads(line)['usd'] for line in path.read_text().splitlines() if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('batch', type=Path)
    ap.add_argument('--models', default='configs/pilot_models_v3.json')
    ap.add_argument('--ledger', type=Path, required=True)
    ap.add_argument('--env-file', default=None, help='private file with the provider key (SLOWLAB_ENV_FILE)')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--max-hours', type=float, default=3.0)
    ap.add_argument('--max-memory-gb', type=float, default=6.0)
    ap.add_argument('--pause-failure-share', type=float, default=None)
    ap.add_argument('--pause-min-finished', type=int, default=10)
    args = ap.parse_args()
    cfg = json.loads((ROOT / args.models).read_text())
    cap = cfg['budget']['hard_cap_usd']
    prices = {m['name']: m['price'] for m in cfg['models']}
    reserves = {m['name']: m['reserve_usd'] for m in cfg['models'] if 'reserve_usd' in m}
    contract = json.loads((ROOT / 'configs/task_contract_v8.json').read_text())
    per_branch = contract['budget']['context']['max_input_chars_per_campaign']
    batch = json.loads(args.batch.read_text())
    out_dir = Path(batch['out_dir']); (out_dir / 'specs').mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if args.env_file:
        env['SLOWLAB_ENV_FILE'] = args.env_file

    def worst(job):
        if job['model_name'] in reserves:
            return reserves[job['model_name']]
        m = job['llm']
        return worst_case_campaign_cost(prices[job['model_name']], max_input_chars_per_branch=per_branch,
                                        max_llm_calls_per_branch=LLMConfig().max_llm_calls, max_tokens=m['max_tokens'])

    def done_status(job):
        out = out_dir / f"{job['id']}.json"
        return json.loads(out.read_text()).get('status') if out.exists() else None

    def run(job):
        out = out_dir / f"{job['id']}.json"
        spec = {**batch['common'], **{k: v for k, v in job.items() if k not in ('id', 'evaluation_years')}, 'out': str(out)}
        path = out_dir / 'specs' / f"{job['id']}.json"
        path.write_text(json.dumps(spec, indent=1))
        limit = int(args.max_memory_gb * 2 ** 30)

        def cap_mem():
            try:
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except (ValueError, OSError):
                pass
        with (out_dir / f"{job['id']}.log").open('w') as log:
            try:
                subprocess.run([sys.executable, '-B', str(ROOT / batch['script']), str(path)], cwd=ROOT, env=env,
                               stdout=log, stderr=subprocess.STDOUT, timeout=args.max_hours * 3600, preexec_fn=cap_mem)
            except subprocess.TimeoutExpired:
                out.write_text(json.dumps({'status': 'failed', 'error': f'wall-clock limit {args.max_hours} h'}) + '\n')
        return json.loads(out.read_text()) if out.exists() else {'status': 'no_output'}

    pending = [j for j in batch['jobs'] if done_status(j) != 'completed']
    spent = ledger_total(args.ledger)
    running, reserve, skipped = {}, {}, []
    finished_count, failed_count, paused = 0, 0, None
    began = time.monotonic()
    with ThreadPoolExecutor(args.workers) as pool:
        while pending or running:
            launched = False
            for job in list(pending):
                if len(running) >= args.workers or paused:
                    break
                if job['method'] == 'llm':
                    need = worst(job)
                    if spent + sum(reserve.values()) + need > cap:
                        continue
                    reserve[job['id']] = need
                running[pool.submit(run, job)] = job
                pending.remove(job)
                launched = True
            if not running:
                if paused:
                    break
                skipped = [j['id'] for j in pending]
                pending = []
                break
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED, timeout=None if not launched else 1)
            for fut in finished:
                job = running.pop(fut)
                record = fut.result()
                cost = 0.0
                if job['method'] == 'llm':
                    reserve.pop(job['id'], None)
                    cost = records_cost(record.get('provider_call_records') or [], prices[job['model_name']])
                    with args.ledger.open('a') as f:
                        f.write(json.dumps({'kind': 'campaign', 'job': job['id'], 'model': job['model_name'],
                                            'status': record.get('status'), 'usd': cost,
                                            'calls': len(record.get('provider_call_records') or [])}) + '\n')
                    spent += cost
                finished_count += 1
                failed_count += record.get('status') != 'completed'
                if (args.pause_failure_share is not None and not paused and finished_count >= args.pause_min_finished
                        and failed_count > args.pause_failure_share * finished_count):
                    paused = f'{failed_count} of {finished_count} finished jobs failed'
                print(json.dumps({'job': job['id'], 'status': record.get('status'), 'usd': round(cost, 4),
                                  'spent_usd': round(spent, 4), 'reserved_usd': round(sum(reserve.values()), 3),
                                  'elapsed_h': round((time.monotonic() - began) / 3600, 2)}), flush=True)
    summary = {'spent_usd': spent, 'cap_usd': cap, 'skipped_budget': skipped, 'paused': paused,
               'not_started': [j['id'] for j in pending] if paused else [],
               'status': {j['id']: done_status(j) for j in batch['jobs']}}
    (out_dir / 'DONE').write_text(json.dumps(summary, indent=1))
    print(json.dumps({'spent_usd': round(spent, 4), 'skipped_budget': len(skipped), 'paused': paused}))


if __name__ == '__main__':
    main()
