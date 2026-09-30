"""Phase-0 contract checks and scheduling dry-run, without crop simulation.

Not the V2.2 executor: no harvest, sensor, energy or economic performance is
invented here. These checks make the task contract executable for review.
"""
from __future__ import annotations
import math
from numbers import Real


def finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def validate_policy(contract, policy):
    fields = contract['policy']['fields']
    if not isinstance(policy, dict) or set(policy) != set(fields):
        raise ValueError('policy must contain exactly the six declared fields')
    for name, bounds in fields.items():
        value = policy[name]
        if not finite(value) or not bounds['min'] <= value <= bounds['max']:
            raise ValueError(f'policy field outside domain: {name}')
    if policy['night_temperature_c'] > policy['day_temperature_c']:
        raise ValueError('night temperature must not exceed day temperature')


def validate_contract(c):
    f, b = c['facility'], c['budget']
    g = f['geometry']
    if not math.isclose(g['width_m'] * g['length_m'], f['floor_area_m2']):
        raise ValueError('floor area mismatch')
    if not 0 < g['screen_height_m'] < g['eave_height_m'] < g['mean_height_m']:
        raise ValueError('invalid vertical geometry')
    roof = f['floor_area_m2'] / math.cos(math.radians(g['roof_slope_deg']))
    if not math.isclose(roof, g['roof_area_m2']):
        raise ValueError('roof area mismatch')
    total = roof + sum(g['wall_lower_area_m2'].values()) + sum(g['wall_upper_area_m2'].values())
    if not math.isclose(total, g['reference_cover_area_m2']):
        raise ValueError('cover area mismatch')
    cap = f['capacity']
    if not math.isclose(cap['heating_w_m2']*f['floor_area_m2'], cap['heating_w']):
        raise ValueError('heating unit mismatch')
    if not math.isclose(cap['co2_g_m2_h']*f['floor_area_m2']*1000/3600, cap['co2_mg_s']):
        raise ValueError('CO2 unit mismatch')
    supply = f.get('central_supply')
    if supply is not None:
        if supply.get('model') != 'adequately_sized_no_cross_compartment_contention':
            raise ValueError('unsupported central supply model')
        minimum = {'heating_w_min': cap['heating_w'] * f['compartments'],
                   'co2_mg_s_min': cap['co2_mg_s'] * f['compartments'],
                   'lamp_electric_w_min': cap['lamp_electric_w_m2'] * f['floor_area_m2'] * f['compartments']}
        for name, required in minimum.items():
            value = supply.get(name)
            if not finite(value) or value + 1e-8 < required:
                raise ValueError('central supply cannot meet all simultaneous per-compartment maxima: ' + name)
    if b['latest_start_day'] + b['crop_days'] + b['cleanup_days'] > b['campaign_days']:
        raise ValueError('latest start cannot finish crop and cleanup')
    if c['evaluation']['horizon_days'] != b['crop_days']:
        raise ValueError('deployment and crop horizons differ')
    if not 0 < c['economics']['fruit_dry_matter_fraction'] < 1:
        raise ValueError('invalid dry matter fraction')


def dry_run(c, example):
    """Validate one authored action trace and report only logical resource use."""
    validate_contract(c)
    if example['contract_id'] != c['contract_id']:
        raise ValueError('contract identity mismatch')
    b, f = c['budget'], c['facility']
    clock, runs, trace, final_policy = 0., [], [], None
    actions = example['actions']
    if len(actions) > b['max_tool_calls']:
        raise ValueError('tool action budget exceeded')
    for action in actions:
        kind = action['action']
        if kind not in c['events']['allowed']:
            raise ValueError('unsupported action')
        if kind == 'advance':
            day = action['day']
            if not finite(day) or not clock <= day <= b['campaign_days']:
                raise ValueError('invalid clock advance')
            clock = float(day)
        elif kind in ('start', 'stop', 'observe'):
            unit = action['unit']
            if type(unit) is not int or not 0 <= unit < f['compartments']:
                raise ValueError('unknown compartment')
            existing = [run for run in runs if run['unit'] == unit]
            last = existing[-1] if existing else None
            if kind == 'start':
                if clock > b['latest_start_day'] or len(runs) >= b['max_starts']:
                    raise ValueError('start budget/deadline exceeded')
                if last and clock < last['end'] + b['cleanup_days']:
                    raise ValueError('compartment is active or cleaning')
                policy = example[action['policy']]
                validate_policy(c, policy)
                runs.append({'unit': unit, 'start': clock, 'end': clock+b['crop_days'],
                             'reason': 'normal_completion'})
            elif kind == 'stop':
                if not last or not last['start'] <= clock < last['end']:
                    raise ValueError('no active crop to stop')
                last.update(end=clock, reason='early_stop')
            else:
                when = action.get('as_of_day', clock)
                if not finite(when) or not 0 <= when <= clock:
                    raise ValueError('cannot observe future')
                # No synthetic observation value is fabricated by this dry-run.
        else:
            if clock != b['campaign_days']:
                raise ValueError('final recommendation requires campaign deadline')
            final_policy = example[action['policy']]
            validate_policy(c, final_policy)
        trace.append({'action': kind, 'day': clock})
    if clock != b['campaign_days'] or final_policy is None:
        raise ValueError('example must reach deadline and recommend')
    active = sum(min(run['end'], clock)-run['start'] for run in runs)
    cleanup = sum(max(0, min(clock, run['end']+b['cleanup_days'])-min(clock,run['end'])) for run in runs)
    e, area = c['economics'], f['floor_area_m2']
    return {'scope':'logical scheduling only; no simulated greenhouse results',
            'starts':len(runs), 'early_stops':sum(run['reason']=='early_stop' for run in runs),
            'normal_completions':sum(run['reason']=='normal_completion' and run['end']<=clock for run in runs),
            'active_compartment_days':active, 'cleanup_compartment_days':cleanup,
            'occupied_m2_days':(active+cleanup)*area,
            'known_fixed_cost_eur':area*(len(runs)*(e['planting_eur_per_m2']+e['cleanup_eur_per_m2'])+active*e['background_service_eur_per_m2_day']),
            'harvest_energy_and_margin':'not evaluated', 'trace':trace}
