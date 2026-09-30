"""Sample hidden site configurations from a site-distribution config.

Every component of a site (each site-level parameter, each price, each
compartment effect, the sensor-noise seed) is drawn from its own random
stream, seeded by SHA-256 over the distribution id, the master seed, the site
index and the component name. Adding or removing a component therefore leaves
every other draw unchanged, and site ``i`` does not depend on how many sites
are generated. Nothing here looks at outcomes: sites are neither selected nor
reweighted.

The result feeds the executor's site spec: ``site`` (public prices, side-wall
exchange, per-compartment model parameters), ``soil_boundary_c`` and the
sensor-noise seed. Weather years are assigned separately, once the pools are
fixed.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np

from .site_parameters import MODEL_PARAMETERS, PRICE_FIELDS, site_contract, validate_model_parameters

CONTRACT_TARGETS = {'boundary_ueff_w_m2_k', 'soil_boundary_c'}


def component_rng(distribution_id: str, master_seed: int, site_index: int, component: str) -> np.random.Generator:
    key = '|'.join((distribution_id, str(int(master_seed)), str(int(site_index)), component)).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'big'))


def draw(spec: dict, rng: np.random.Generator):
    kind = spec['type']
    if kind == 'uniform':
        return float(rng.uniform(spec['low'], spec['high']))
    if kind == 'loguniform':
        return float(math.exp(rng.uniform(math.log(spec['low']), math.log(spec['high']))))
    if kind == 'integer':
        return int(rng.integers(spec['low'], spec['high'], endpoint=True))
    raise ValueError(f'unknown distribution type: {kind}')


def _apply(rule: dict, x: float, site_value: float | None = None) -> float:
    op = rule['op']
    if op == 'value':
        return x
    if op == 'times':
        return rule['base'] * x
    if op == 'plus':
        return rule['base'] + x
    if op == 'times_site_value':
        if site_value is None:
            raise ValueError('compartment effect needs a site-level value for its parameter')
        return site_value * x
    raise ValueError(f'unknown rule: {op}')


def sample_site(distribution: dict, master_seed: int, site_index: int, n_units: int = 4) -> dict:
    """One site: raw draws plus the executor-facing values they imply."""
    did = distribution['distribution_id']

    def x_of(component, spec):
        return draw(spec['distribution'], component_rng(did, master_seed, site_index, component))

    draws, model, prices, extra = {}, {}, {}, {}
    for name, spec in distribution['site_level'].items():
        x = draws[name] = x_of('site/' + name, spec)
        for target, rule in spec['applies_to'].items():
            value = _apply(rule, x)
            if target in MODEL_PARAMETERS:
                model[target] = value
            elif target in CONTRACT_TARGETS:
                extra[target] = value
            else:
                raise ValueError(f'unknown site-level target: {target}')
    for name, spec in distribution['prices_public'].items():
        if 'fixed' in spec:
            prices[name] = float(spec['fixed'])
            continue
        x = draws[name] = x_of('price/' + name, spec)
        for target, rule in spec['applies_to'].items():
            if target not in PRICE_FIELDS:
                raise ValueError(f'unknown price target: {target}')
            prices[target] = _apply(rule, x)
    units = {}
    for unit in range(n_units):
        values = dict(model)
        for name, spec in distribution['compartment_level'].items():
            x = draws[f'unit{unit}/{name}'] = x_of(f'unit{unit}/{name}', spec)
            for target, rule in spec['applies_to'].items():
                values[target] = _apply(rule, x, model.get(target))
        units[str(unit)] = validate_model_parameters(values)
    noise = draw(distribution['sensor_noise_seed']['distribution'],
                 component_rng(did, master_seed, site_index, 'sensor_noise_seed'))
    site = {'prices': prices, 'unit_parameters': units}
    if 'boundary_ueff_w_m2_k' in extra:
        site['boundary_ueff_w_m2_k'] = extra['boundary_ueff_w_m2_k']
    return {'distribution_id': did, 'site_index': int(site_index), 'draws': draws, 'site': site,
            'soil_boundary_c': extra.get('soil_boundary_c'), 'sensor_noise_seed': noise}


def check_site(contract: dict, sampled: dict) -> dict:
    """Validate a sampled site against the contract; returns the site's contract copy."""
    for values in sampled['site']['unit_parameters'].values():
        validate_model_parameters(values)
    return site_contract(contract, sampled['site'])
