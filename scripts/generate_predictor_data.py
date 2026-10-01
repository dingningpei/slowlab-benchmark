#!/usr/bin/env python3
"""One predictor-data run on a development site (private process).

    python3 scripts/generate_predictor_data.py SPEC.json

SPEC: as for scripts/evaluate_recommendation.py (backend, contract, site, year,
weather, sensor_noise, out) plus "layout" ("two_wave" or "single") and
"layout_seed". Failures are recorded with status "failed" and exit code 3.
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
from slowlab.executor_server import _path  # noqa: E402
from slowlab.predictor_data import collect  # noqa: E402
from slowlab.private_runs import prepare, progress_writer, sha  # noqa: E402


def main():
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text())
    out = _path(spec['out'])
    began = time.monotonic()
    try:
        p = prepare(spec)
        p['executor_kwargs']['progress_hook'] = progress_writer(out)
        result = collect(p['contract'], layout=spec['layout'], layout_seed=int(spec['layout_seed']), year=p['year'],
                         weather=p['weather'], source=p['source'], site_unit_parameters=p['unit_parameters'],
                         soil_boundary_c=p['soil_boundary_c'], sensor_noise=p['sensor_noise'],
                         executor_kwargs=p['executor_kwargs'])
        record, code = {'status': 'completed', 'identity': p['identity'], 'site_prices': (p['site'] or {}).get('prices'),
                        **result}, 0
    except Exception as error:
        record, code = {'status': 'failed', 'error': f'{type(error).__name__}: {error}',
                        'traceback': traceback.format_exc()}, 3
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record.update(spec_sha256=sha(spec_path), elapsed_seconds=round(time.monotonic() - began, 1),
                  peak_rss_mb=round(rss / (1024 * 1024 if sys.platform == 'darwin' else 1024), 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record) + '\n')
    print(json.dumps({k: record.get(k) for k in ('status', 'elapsed_seconds', 'peak_rss_mb', 'error')}))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
