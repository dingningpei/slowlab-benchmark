"""Training data for the process predictor (private, development sites only).

One data run is one facility year on a development site with randomly drawn
feasible policies:

* ``two_wave``: every compartment plants on day 0 and again on day 182 (eight crops);
* ``single``: every compartment plants once on its own random day in [0, 183] (four crops).

For every crop the run records what an agent in the Full condition could have
read on each whole day since planting (the latest public reading of each
predictor channel at that time; the public store is causal and immutable, so
reading it after the run gives the same values) and the crop's final accrued
totals. Policies are uniform in the public box with night <= day.
"""
from __future__ import annotations

import numpy as np

from .campaign_executor import CampaignExecutor
from datetime import datetime, timezone

CHANNELS = ('cumulative_harvest_fresh_equivalent', 'heating_energy', 'lighting_energy', 'co2_dosed',
            'canopy_lai_proxy')
FORMAT = 'slowlab-predictor-data-v1'


def random_policy(fields: dict, rng: np.random.Generator) -> dict:
    while True:
        policy = {name: float(rng.uniform(spec['min'], spec['max'])) for name, spec in fields.items()}
        if policy['night_temperature_c'] <= policy['day_temperature_c']:
            return {k: round(v, 3) for k, v in policy.items()}


def layout_plan(contract: dict, layout: str, rng: np.random.Generator) -> list[tuple[float, int]]:
    budget = contract['budget']
    units = contract['facility']['compartments']
    crop, cleanup, latest = budget['crop_days'], budget['cleanup_days'], budget['latest_start_day']
    if layout == 'two_wave':
        second = crop + cleanup
        if second > latest or second + crop > budget['campaign_days']:
            raise ValueError('two waves do not fit the calendar')
        return [(0.0, u) for u in range(units)] + [(float(second), u) for u in range(units)]
    if layout == 'single':
        days = rng.integers(0, int(latest) + 1, size=units) if latest >= 1 else np.zeros(units)
        return sorted((float(d), u) for u, d in enumerate(days))
    raise ValueError('unknown layout')


def collect(contract: dict, *, layout: str, layout_seed: int, year: int, weather, source, site_unit_parameters=None,
            soil_boundary_c=None, sensor_noise=None, executor_kwargs=None, snapshot_step_days: float = 1.0) -> dict:
    rng = np.random.default_rng(layout_seed)
    plan = layout_plan(contract, layout, rng)
    fields = contract['policy']['fields']
    policies = [random_policy(fields, rng) for _ in plan]
    crop_days = contract['budget']['crop_days']
    x = CampaignExecutor(contract, source, weather, feedback_mode='full', fallback_policy=policies[0],
                         origin_utc=datetime(int(year) - 1, 12, 31, 23, tzinfo=timezone.utc),
                         soil_boundary_c=soil_boundary_c, unit_parameters=site_unit_parameters,
                         sensor_noise=sensor_noise, **(executor_kwargs or {}))
    starts = []
    for (day, unit), policy in zip(plan, policies):
        if day * 86400 > x.clock:
            x.dispatch({'action': 'advance', 'day': day})
        result = x.dispatch({'action': 'start', 'unit': unit, 'policy': policy})
        starts.append({'unit': unit, 'run_index': result['run_index'], 'planting_day': day, 'policy': policy})
    x.dispatch({'action': 'advance', 'day': max(d for d, _ in plan) + crop_days})
    totals = {(int(r['unit']), int(r['run_index'])): r for r in x.private_crop_totals()}
    steps = int(round(crop_days / snapshot_step_days))
    crops = []
    for s in starts:
        t0 = s['planting_day'] * 86400
        times = [t0 + k * snapshot_step_days * 86400 for k in range(steps + 1)]
        snaps = {}
        for channel in CHANNELS:
            values = []
            for t in times:
                record = x._public.latest(str(s['unit']), channel, as_of=t)
                values.append(None if record is None else float(record['value']))
            snaps[channel] = values
        final = totals[(s['unit'], s['run_index'])]
        crops.append({**s, 'snapshot_days': [k * snapshot_step_days for k in range(steps + 1)], 'snapshots': snaps,
                      'final_reason': final['reason'], 'final_accrued': final['accrued'],
                      'final_event_cost_eur_m2': final['event_cost_eur_m2']})
    return {'format': FORMAT, 'layout': layout, 'layout_seed': layout_seed, 'year': int(year), 'crops': crops,
            'unit_parameters': x.unit_parameters, 'soil_boundary_c': x.soil_boundary_c}
