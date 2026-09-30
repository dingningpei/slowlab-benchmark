"""Hidden site parameters: which model constants a site may change, and how.

``MODEL_PARAMETERS`` lists the pinned-model constants a site (or one
compartment of it) may override, with hard plausibility bounds used only to
reject malformed input; sampling distributions live in the site-distribution
config. Contract-level site values (public prices, the side-wall exchange
coefficient) are applied by ``site_contract``, which returns a copy of the
contract for that site.
"""
from __future__ import annotations

import copy
import math

MODEL_PARAMETERS = {
    'j25LeafMax': (50.0, 500.0),
    'rgFruit': (0.05, 1.0),
    'tEndSum': (500.0, 2000.0),
    'laiMax': (1.0, 6.0),
    'tCan24Min': (5.0, 25.0),
    'tCan24Max': (15.0, 35.0),
    'tauRfPar': (0.4, 0.95),
    'tauRfNir': (0.4, 0.95),
    'cLeakage': (1e-5, 1e-3),
    'cDgh': (0.3, 1.2),
    'cWgh': (0.02, 0.3),
    'etaLampPar': (0.2, 0.8),
}
PRICE_FIELDS = {
    'price_eur_per_kg_fresh_equivalent': (0.5, 6.0),
    'electricity_eur_per_kwh': (0.02, 0.8),
    'delivered_heat_eur_per_kwh': (0.005, 0.3),
    'co2_eur_per_kg': (0.02, 1.0),
}
UEFF_BOUNDS = (0.0, 10.0)


def _finite_within(name, value, bounds):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'site value {name} must be a finite number')
    low, high = bounds
    if not low <= value <= high:
        raise ValueError(f'site value {name}={value} outside plausibility bounds [{low}, {high}]')
    return float(value)


def validate_model_parameters(values: dict | None) -> dict:
    if not values:
        return {}
    unknown = set(values) - set(MODEL_PARAMETERS)
    if unknown:
        raise ValueError('not an overridable model parameter: ' + ', '.join(sorted(unknown)))
    out = {name: _finite_within(name, value, MODEL_PARAMETERS[name]) for name, value in values.items()}
    low = out.get('tCan24Min', 15.0)
    high = out.get('tCan24Max', 24.5)
    if not low < high:
        raise ValueError('tCan24Min must be below tCan24Max')
    return out


def site_contract(contract: dict, site: dict | None) -> dict:
    """Contract copy with the site's public prices and side-wall exchange coefficient."""
    if not site:
        return contract
    unknown = set(site) - {'prices', 'boundary_ueff_w_m2_k', 'unit_parameters'}
    if unknown:
        raise ValueError('unknown site fields: ' + ', '.join(sorted(unknown)))
    out = copy.deepcopy(contract)
    for name, value in (site.get('prices') or {}).items():
        if name not in PRICE_FIELDS:
            raise ValueError(f'not a site price field: {name}')
        out['economics'][name] = _finite_within(name, value, PRICE_FIELDS[name])
    if site.get('boundary_ueff_w_m2_k') is not None:
        out['facility']['boundary']['ueff_w_m2_k'] = _finite_within(
            'boundary_ueff_w_m2_k', site['boundary_ueff_w_m2_k'], UEFF_BOUNDS)
    return out
