"""Deterministic virtual public measurements from reached GreenLight endpoints.

This is a named, limited sensor transform, not permission to expose model state.
"""
from __future__ import annotations
import math


def public_endpoint_measurements(engine, ledger, indoor):
    if engine.clock != ledger.clock:
        raise ValueError('sensor and resource clocks differ')
    if set(indoor) != {'air_temperature_c','relative_humidity_pct','co2_ppm'}:
        raise ValueError('incomplete indoor virtual sensor triplet')
    sla=float(engine.model.consts['sla'])
    lai=sla*float(engine.state['cLeaf'])
    totals=ledger.summary()['per_m2']
    result={**indoor,
            'canopy_lai_proxy':lai,
            'cumulative_harvest_fresh_equivalent':totals['harvest_kg_m2'],
            'heating_energy':totals['heat_kwh_m2'],
            'lighting_energy':totals['light_kwh_m2'],
            'co2_dosed':totals['co2_kg_m2']}
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in result.values()):
        raise ValueError('nonfinite public sensor value')
    if any(result[k]<-1e-12 for k in result if k not in ('air_temperature_c',)):
        raise ValueError('negative public cumulative/crop sensor value')
    return result
