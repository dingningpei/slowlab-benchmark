#!/usr/bin/env python3
"""Evaluate several policies in one run (reference-search job; private process). SPEC as for
scripts/evaluate_recommendation.py, with "policies" (a list) instead of "policy"."""
from __future__ import annotations

import json
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.evaluator import evaluate_policies_year  # noqa: E402
from slowlab.executor_server import _path  # noqa: E402
from slowlab.private_runs import prepare, sha  # noqa: E402


def main():
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text())
    out = _path(spec['out'])
    began = time.monotonic()
    try:
        p = prepare(spec)
        result = evaluate_policies_year(p['contract'], spec['policies'], site=p['site'],
                                        unit_parameters=p['unit_parameters'], weather=p['weather'], source=p['source'],
                                        year=p['year'], soil_boundary_c=p['soil_boundary_c'],
                                        sensor_noise=p['sensor_noise'], executor_kwargs=p['executor_kwargs'])
        record, code = {'status': 'completed', 'identity': p['identity'], **result}, 0
    except Exception as error:
        record, code = {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                        'traceback': traceback.format_exc()}, 3
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record.update(spec_sha256=sha(spec_path), elapsed_seconds=round(time.monotonic() - began, 1),
                  peak_rss_mb=round(rss / (1024 * 1024 if sys.platform == 'darwin' else 1024), 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record) + '\n')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
