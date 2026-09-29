"""Bounded Phase-1 causal-control smoke; not a complete campaign executor."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path


def _nodes(obj):
    found = {}
    for key, value in obj.items():
        if isinstance(value, dict):
            if 'type' in value:
                found[key] = value
            found.update(_nodes(value))
    return found


def model_override(contract, source: Path):
    """Check pinned assets and create the declared 96 m2 development override."""
    definitions = source/'greenlight/models/katzin_2021/definition'
    nodes = {}
    for name, expected in contract['model']['files'].items():
        path = definitions/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'pinned model hash mismatch: {name}')
        if name != contract['model']['entrypoint']:
            nodes.update(_nodes(json.loads(path.read_text())))
    f = contract['facility']; g = f['geometry']; cap = f['capacity']
    constants = {'aFlr': f['floor_area_m2'], 'aCov': g['reference_cover_area_m2'],
                 'hAir':g['screen_height_m'], 'hGh':g['mean_height_m'],
                 'psi':g['roof_slope_deg'], 'aRoof':cap['roof_vent_area_m2'],
                 'pBoil':cap['heating_w'], 'phiExtCo2':cap['co2_mg_s']}
    override = {k:{'definition':repr(float(v))} for k,v in constants.items()}
    boundary = f['boundary']
    for state, key in [('tAir','wall_lower_area_m2'),('tTop','wall_upper_area_m2')]:
        terms = [f"{boundary['ueff_w_m2_k']*g[key][wall]/f['floor_area_m2']}*({boundary[wall+'_temperature_c']}-tOut)"
                 for wall in ('north','east','west')]
        correction = '+'.join(terms)
        capacity = 'capAir' if state == 'tAir' else 'capTop'
        override[state] = {'definition':f"({nodes[state]['definition']})+({correction})/{capacity}"}
    return definitions, override


def held_commands(policy, observed, local_hour):
    """Native smooth bands applied only to current sampled climate."""
    def control(x, point, band):
        z = 2/band*math.log(100)*(x-point-band/2)
        return 1/(1+math.exp(-max(-700,min(700,z))))
    day = 6 <= local_hour < 22
    target = policy['day_temperature_c'] if day else policy['night_temperature_c']
    t,rh,co2,solar = (observed[k] for k in ('temperature','rh','co2','solar'))
    lamp = float(6 <= local_hour < 6+policy['supplemental_light_hours'] and solar < 400 and t < 35)
    heat = control(t,target,-1)
    # ventCold in the native model has min_val=1, max_val=0.
    vent = min(1-control(t,target-1,-1), max(control(t,target+policy['vent_temperature_offset_c'],4),
                                                    control(rh,policy['vent_rh_threshold_pct'],50)))
    dose = control(co2,policy['co2_target_ppm'],-100) if solar>20 or lamp else 0.
    return {'uBoil':heat,'uRoof':vent,'uExtCo2':dose,'uLamp':lamp}
