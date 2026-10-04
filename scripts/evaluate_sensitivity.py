#!/usr/bin/env python3
"""Evaluate one recommendation in one year with a site parameter overridden (boundary sensitivity).

    python3 scripts/evaluate_sensitivity.py SPEC.json

SPEC: as for scripts/evaluate_recommendation.py, plus "site_overrides": {"boundary_ueff_w_m2_k": value}
(the only field allowed; decision 2026-10-03: Ueff 0 and 4 W/m2K). Everything else, including the
site's evaluation draws, is unchanged, so the result pairs with the main evaluation of the same
recommendation and year.
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
from slowlab.private_runs import prepare, progress_writer, sha  # noqa: E402

ALLOWED = {'boundary_ueff_w_m2_k'}


def run(spec: dict) -> dict:
    overrides = spec['site_overrides']
    if set(overrides) - ALLOWED or not overrides:
        raise ValueError('only boundary_ueff_w_m2_k may be overridden')
    policy = (json.loads(_path(spec['settlement']).read_text())['recommendation'] if 'settlement' in spec
              else spec['policy'])
    p = prepare(spec)
    site = {**p['site'], **overrides}
    p['executor_kwargs']['progress_hook'] = progress_writer(_path(spec['out']))
    result = evaluate_year(p['contract'], policy, site=site, unit_parameters=p['unit_parameters'],
                           weather=p['weather'], source=p['source'], year=p['year'],
                           soil_boundary_c=p['soil_boundary_c'], sensor_noise=p['sensor_noise'],
                           executor_kwargs=p['executor_kwargs'])
    return {'status': 'completed', 'identity': p['identity'], 'site_overrides': overrides,
            'site_value_replaced': {k: p['site'].get(k) for k in overrides}, **result}


def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: evaluate_sensitivity.py SPEC.json')
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text())
    out = _path(spec['out'])
    began = time.monotonic()
    try:
        record, code = run(spec), 0
    except Exception as error:
        record = {'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        code = 3
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record.update(spec_sha256=sha(spec_path), backend=spec['backend'], elapsed_seconds=round(time.monotonic() - began, 1),
                  peak_rss_mb=round(rss / (1024 * 1024 if sys.platform == 'darwin' else 1024), 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record.get(k) for k in ('status', 'elapsed_seconds', 'error')}))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
