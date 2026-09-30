#!/usr/bin/env python3
"""Run many evaluations, one process each, with a fixed worker count (resumable).

    python3 scripts/run_evaluation_batch.py BATCH.json --workers 12

BATCH: {"out_dir": ..., "common": {spec fields shared by all jobs},
        "jobs": [{"id": ..., <spec fields: site, year, policy, weather, ...>}]}
A job whose output already records status "completed" is skipped; failed jobs
are kept with their identity and rerun only with --retry-failed.
"""
from __future__ import annotations

import argparse
import json
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
        with (out_dir / f"{job['id']}.log").open('w') as log:
            subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/evaluate_recommendation.py'), str(path)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
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
