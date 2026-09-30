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

import hashlib
import json
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.evaluator import evaluate_year  # noqa: E402
from slowlab.executor_server import _path, build_weather  # noqa: E402
from slowlab.site_distribution import evaluation_draws, sample_site  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(spec: dict) -> dict:
    contract_path = _path(spec['contract'])
    contract = json.loads(contract_path.read_text())
    if 'settlement' in spec:
        policy = json.loads(_path(spec['settlement']).read_text())['recommendation']
    else:
        policy = spec['policy']
    year = int(spec['year'])
    identity = {'contract_sha256': sha(contract_path), 'year': year}
    site, unit_parameters, soil, noise_seed = None, None, None, None
    if spec.get('site'):
        s = spec['site']
        dist_path = _path(s['distribution'])
        distribution = json.loads(dist_path.read_text())
        sampled = sample_site(distribution, s['master_seed'], s['site_index'], contract['facility']['compartments'])
        drawn = evaluation_draws(distribution, s['master_seed'], s['site_index'], year, contract['facility']['compartments'])
        site = {k: v for k, v in sampled['site'].items() if k != 'unit_parameters'}
        unit_parameters, soil, noise_seed = drawn['unit_parameters'], sampled['soil_boundary_c'], drawn['sensor_noise_seed']
        identity.update(distribution_sha256=sha(dist_path), site_index=s['site_index'])
    noise = None
    if spec.get('sensor_noise'):
        from slowlab.sensor_noise import SensorNoise
        n = spec['sensor_noise']
        if noise_seed is not None and 'seed' in n:
            raise ValueError('with a site, the evaluation noise seed comes from the site draws')
        noise = SensorNoise.from_file(_path(n['config']), noise_seed if noise_seed is not None else n['seed'],
                                      n.get('setting', 'main'))
    if spec['backend'] == 'fake':
        from slowlab import fake_backend
        weather, source = None, None
        kwargs = {'lifecycle_factory': fake_backend.FakeLifecycle, 'sample_endpoint': fake_backend.sample_endpoint}
    elif spec['backend'] == 'greenlight':
        from slowlab.greenlight_source import resolve_greenlight_source
        source = resolve_greenlight_source(_path(spec.get('greenlight_source')), contract)
        weather = build_weather(spec['weather'], datetime(year - 1, 12, 31, 23, tzinfo=timezone.utc))
        kwargs = {}
    else:
        raise ValueError('unknown backend')
    result = evaluate_year(contract, policy, site=site, unit_parameters=unit_parameters, weather=weather,
                           source=source, year=year, soil_boundary_c=soil, sensor_noise=noise, executor_kwargs=kwargs)
    return {'status': 'completed', 'identity': identity, **result}


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
