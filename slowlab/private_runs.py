"""Shared private setup for executor-side runs (evaluation, predictor data).

``prepare(spec)`` turns a private run spec into the contract, site values,
per-compartment parameters, deep-soil temperature, sensor-noise model, weather
reader, model source and executor keyword arguments. With a sampled site, the
compartment differences and the noise seed are the site's draws for the run's
weather year (``evaluation_draws``), separate from its campaign draws.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .executor_server import _path, build_weather
from .site_distribution import evaluation_draws, sample_site


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(spec: dict) -> dict:
    contract_path = _path(spec['contract'])
    contract = json.loads(contract_path.read_text())
    year = int(spec['year'])
    identity = {'contract_sha256': sha(contract_path), 'year': year}
    site, unit_parameters, soil, noise_seed = None, None, None, None
    units = contract['facility']['compartments']
    if spec.get('site'):
        s = spec['site']
        dist_path = _path(s['distribution'])
        distribution = json.loads(dist_path.read_text())
        sampled = sample_site(distribution, s['master_seed'], s['site_index'], units)
        drawn = evaluation_draws(distribution, s['master_seed'], s['site_index'], year, units)
        site = {k: v for k, v in sampled['site'].items() if k != 'unit_parameters'}
        unit_parameters, soil, noise_seed = drawn['unit_parameters'], sampled['soil_boundary_c'], drawn['sensor_noise_seed']
        identity.update(distribution_sha256=sha(dist_path), site_index=s['site_index'])
    noise = None
    if spec.get('sensor_noise'):
        from .sensor_noise import SensorNoise
        n = spec['sensor_noise']
        if noise_seed is not None and 'seed' in n:
            raise ValueError('with a site, the noise seed comes from the site draws')
        noise = SensorNoise.from_file(_path(n['config']), noise_seed if noise_seed is not None else n['seed'],
                                      n.get('setting', 'main'))
    if spec['backend'] == 'fake':
        from . import fake_backend
        weather, source = None, None
        kwargs = {'lifecycle_factory': fake_backend.FakeLifecycle, 'sample_endpoint': fake_backend.sample_endpoint}
    elif spec['backend'] == 'greenlight':
        from .greenlight_source import resolve_greenlight_source
        source = resolve_greenlight_source(_path(spec.get('greenlight_source')), contract)
        weather = build_weather(spec['weather'], datetime(year - 1, 12, 31, 23, tzinfo=timezone.utc))
        kwargs = {}
    else:
        raise ValueError('unknown backend')
    return {'contract': contract, 'year': year, 'site': site, 'unit_parameters': unit_parameters,
            'soil_boundary_c': soil, 'sensor_noise': noise, 'weather': weather, 'source': source,
            'executor_kwargs': kwargs, 'identity': identity}


def progress_writer(out: Path):
    """Executor progress hook: write the simulated day and wall time beside the output, once per day."""
    import time
    path = Path(out).with_suffix('.progress')
    began = time.monotonic()

    def hook(executor):
        path.write_text(json.dumps({'day': executor.clock / 86400, 'wall_seconds': round(time.monotonic() - began, 1)}))
    return hook
