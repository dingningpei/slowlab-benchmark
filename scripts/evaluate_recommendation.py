#!/usr/bin/env python3
"""Evaluate one recommended policy in one evaluation weather year (private process).

    python3 scripts/evaluate_recommendation.py SPEC.json

SPEC fields:
  backend            "greenlight" or "fake"
  contract           contract path
  policy | settlement  the policy, or a private settlement whose recommendation is scored
  site               null (model defaults) or {"distribution", "master_seed", "site_index"}
  year               evaluation weather year
  weather            reader spec as for the executor server (greenlight backend)
  greenlight_source  optional pinned source path
  sensor_noise       null or {"config", "setting", "seed"}; with a site the seed is the
                     site's evaluation draw for this year and "seed" must be omitted
  out                private output path

Writes a private record (status, score, per-crop margins, identities, wall time,
peak RSS). An exception is recorded as status "failed" and exits with code 3.
"""
from __future__ import annotations

import json
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.evaluator import evaluate_year  # noqa: E402
from slowlab.executor_server import _path  # noqa: E402
from slowlab.private_runs import prepare, sha  # noqa: E402


def run(spec: dict) -> dict:
    if 'settlement' in spec:
        policy = json.loads(_path(spec['settlement']).read_text())['recommendation']
    else:
        policy = spec['policy']
    p = prepare(spec)
    result = evaluate_year(p['contract'], policy, site=p['site'], unit_parameters=p['unit_parameters'],
                           weather=p['weather'], source=p['source'], year=p['year'],
                           soil_boundary_c=p['soil_boundary_c'], sensor_noise=p['sensor_noise'],
                           executor_kwargs=p['executor_kwargs'])
    return {'status': 'completed', 'identity': p['identity'], **result}


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: evaluate_recommendation.py SPEC.json')
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text())
    out = _path(spec['out'])
    began = time.monotonic()
    try:
        record = run(spec)
        code = 0
    except Exception as error:  # recorded, never silently dropped
        record = {'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        code = 3
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record.update(spec_sha256=sha(spec_path), backend=spec['backend'], elapsed_seconds=round(time.monotonic() - began, 1),
                  peak_rss_mb=round(rss / (1024 * 1024 if sys.platform == 'darwin' else 1024), 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record.get(k) for k in ('status', 'score_eur_m2', 'elapsed_seconds', 'peak_rss_mb', 'error')}))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
