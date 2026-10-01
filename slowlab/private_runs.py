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


DEVELOPMENT_YEARS = (2014, 2017, 2018, 2019, 2020)
FORMAL_AUDIT = 'results/weather_formal_year_audit_v0.json'


def weather_spec(year: int, dev_cache: str, formal_cache: str | None = None) -> dict:
    """Weather reader spec for a campaign or evaluation year (development or audited formal year)."""
    year = int(year)
    if year == 2014:
        return {'kind': 'development_expanded', 'cache': dev_cache}
    if year in (2017, 2018, 2019, 2020):
        return {'cache': dev_cache, 'plan': 'configs/weather_gapfilled_plan.json'}
    if formal_cache is None:
        raise ValueError(f'year {year} needs the formal weather cache')
    return {'kind': 'formal', 'cache': formal_cache, 'audit': FORMAL_AUDIT, 'year': year, 'dev_cache': dev_cache}


def campaign_spec(*, site: dict, year: int, contract: str, feedback_mode, fallback_policy: dict, private_dir: Path,
                  dev_cache: str, formal_cache: str | None = None, noise_config: str = 'configs/sensor_noise_v0.json',
                  backend: str = 'greenlight', trace: bool = True) -> dict:
    """Executor-server spec for a campaign on a sampled site: the site's campaign draws (not its evaluation draws)."""
    from .site_distribution import sample_site
    from .executor_server import _path
    distribution = json.loads(_path(site['distribution']).read_text())
    units = json.loads(_path(contract).read_text())['facility']['compartments']
    sampled = sample_site(distribution, site['master_seed'], site['site_index'], units)
    private_dir = Path(private_dir)
    spec = {'backend': backend, 'contract': contract, 'feedback_mode': feedback_mode,
            'fallback_policy': fallback_policy, 'origin_utc': f'{int(year) - 1}-12-31T23:00:00+00:00',
            'soil_boundary_c': sampled['soil_boundary_c'], 'site': sampled['site'],
            'trace': str(private_dir / 'trace.jsonl.gz') if trace else None,
            'settlement_out': str(private_dir / 'settlement.json'), 'failure_out': str(private_dir / 'failure.json')}
    if backend == 'greenlight':
        spec['weather'] = weather_spec(year, dev_cache, formal_cache)
        spec['sensor_noise'] = {'config': noise_config, 'seed': sampled['sensor_noise_seed'], 'setting': 'main'}
    return spec
