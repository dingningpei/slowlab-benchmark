#!/usr/bin/env python3
"""Run many evaluations, one process each, with a fixed worker count (resumable).

    python3 scripts/run_evaluation_batch.py BATCH.json --workers 12

BATCH: {"out_dir": ..., "script": optional runner (default evaluate_recommendation.py), "common": {spec fields shared by all jobs},
        "jobs": [{"id": ..., <spec fields: site, year, policy, weather, ...>}]}
A job whose output already records status "completed" is skipped; failed jobs
are kept with their identity and rerun only with --retry-failed. Each job runs
under a wall-clock limit and an address-space limit, so a numerically runaway
run fails on its own (with its last simulated day) instead of starving others.
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('batch', type=Path)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--max-hours', type=float, default=2.0, help='wall-clock limit per job')
    parser.add_argument('--max-memory-gb', type=float, default=6.0, help='address-space limit per job')
    args = parser.parse_args()
    batch = json.loads(args.batch.read_text())
    out_dir = Path(batch['out_dir'])
    (out_dir / 'specs').mkdir(parents=True, exist_ok=True)
    ids = [job['id'] for job in batch['jobs']]
    if len(set(ids)) != len(ids):
        raise SystemExit('duplicate job ids')

    def run(job):
        out = out_dir / f"{job['id']}.json"
        if out.exists():
            status = json.loads(out.read_text()).get('status')
            if status == 'completed' or (status == 'failed' and not args.retry_failed):
                return job['id'], status, 0.0
        spec = {**batch['common'], **{k: v for k, v in job.items() if k != 'id'}, 'out': str(out)}
        path = out_dir / 'specs' / f"{job['id']}.json"
        path.write_text(json.dumps(spec, indent=2))
        began = time.monotonic()
        script = ROOT / batch.get('script', 'scripts/evaluate_recommendation.py')
        limit = int(args.max_memory_gb * 2 ** 30)

        def cap():
            try:  # Linux (the run host) enforces this; macOS rejects lowering RLIMIT_AS
                resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
            except (ValueError, OSError):
                pass
        with (out_dir / f"{job['id']}.log").open('w') as log:
            try:
                subprocess.run([sys.executable, '-B', str(script), str(path)], cwd=ROOT, stdout=log,
                               stderr=subprocess.STDOUT, timeout=args.max_hours * 3600, preexec_fn=cap)
            except subprocess.TimeoutExpired:
                progress = out.with_suffix('.progress')
                out.write_text(json.dumps({'status': 'failed', 'error': f'wall-clock limit {args.max_hours} h',
                                           'limit': 'time', 'spec': spec,
                                           'last_progress': progress.read_text() if progress.exists() else None},
                                          indent=2) + '\n')
        status = json.loads(out.read_text()).get('status') if out.exists() else 'no_output'
        return job['id'], status, time.monotonic() - began

    began = time.monotonic()
    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for job_id, status, seconds in pool.map(run, batch['jobs']):
            done += 1
            print(json.dumps({'job': job_id, 'status': status, 'seconds': round(seconds, 1),
                              'done': done, 'of': len(ids), 'elapsed_h': round((time.monotonic() - began) / 3600, 2)}),
                  flush=True)
    summary = {}
    for job_id in ids:
        out = out_dir / f'{job_id}.json'
        summary[job_id] = json.loads(out.read_text()).get('status') if out.exists() else 'no_output'
    (out_dir / 'DONE').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
