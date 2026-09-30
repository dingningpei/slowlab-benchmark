"""Shared standby controller and physical per-floor-area resource accounting.

Development primitives; no agent exposure, weather driver or campaign scheduling.
"""
from __future__ import annotations
import math
from numbers import Real


def standby_commands(contract, available_temperature):
    if isinstance(available_temperature, bool) or not isinstance(available_temperature, Real) or not math.isfinite(available_temperature):
        raise ValueError('standby requires available finite temperature')
    s=contract['controller']['standby']
    def response(target, band):
        z=2/band*math.log(100)*(available_temperature-target-band/2)
        return 1/(1+math.exp(-max(-700,min(700,z))))
    return {'uBoil':response(s['heating_target_c'],s['heating_band_c']),
            'uRoof':response(s['ventilation_target_c'],s['ventilation_band_c']),
            'uExtCo2':0.,'uLamp':0.}


class ResourceLedger:
    """Integrate solver samples, rejecting gaps/duplicates and invalid fluxes.

    Totals are per m2 floor. Heating is delivered heat, not fuel; transpiration
    is positive water vapour production, not irrigation. No standing-fruit sale.
    """
    KEYS=('heat_kwh_m2','light_kwh_m2','co2_kg_m2','harvest_kg_m2','transpiration_kg_m2','days')
    def __init__(self, contract, start):
        self.contract=contract
        self.clock=float(start)
        self.totals={phase:{k:0. for k in self.KEYS} for phase in ('active','cleanup','idle')}
        self.event_cost_eur_m2=0.
        self.events=[]

    def record_event(self, kind, time):
        if kind not in ('plant','stop') or time!=self.clock:
            raise ValueError('invalid accounting event or clock')
        if self.events and self.events[-1]['kind']==kind:
            raise ValueError('duplicate lifecycle event')
        if not self.events and kind!='plant':
            raise ValueError('cannot stop before planting')
        rate='planting_eur_per_m2' if kind=='plant' else 'cleanup_eur_per_m2'
        cost=self.contract['economics'][rate]
        self.event_cost_eur_m2+=cost
        self.events.append({'kind':kind,'time':time,'cost_eur_m2':cost})

    def add_segment(self, data, start, end, phase):
        import numpy as np
        if phase not in self.totals or start!=self.clock or not math.isfinite(end) or end<=start:
            raise ValueError('noncontiguous or invalid ledger segment')
        t=np.asarray(data['Time'],dtype=float)
        if len(t)<2 or not np.isfinite(t).all() or t[0]!=start or t[-1]!=end or not (np.diff(t)>0).all():
            raise ValueError('invalid quadrature times')
        def integrate(name,positive=False):
            y=np.asarray(data[name],dtype=float)
            if y.shape!=t.shape or not np.isfinite(y).all():
                raise ValueError('invalid resource flux: '+name)
            if positive:y=np.maximum(y,0.)
            elif (y < -1e-10).any():raise ValueError('negative resource flux: '+name)
            else:y=np.maximum(y,0.)
            return float(np.sum((y[:-1]+y[1:])*0.5*np.diff(t)))
        inc={'heat_kwh_m2':integrate('hBoilPipe')/3.6e6,
             'light_kwh_m2':integrate('qLampIn')/3.6e6,
             'co2_kg_m2':integrate('mcExtAir')*1e-6,
             'harvest_kg_m2':integrate('mcFruitHar')*1e-6/self.contract['economics']['fruit_dry_matter_fraction'],
             'transpiration_kg_m2':integrate('mvCanAir',positive=True),'days':(end-start)/86400}
        if phase!='active' and (inc['harvest_kg_m2']>1e-12 or inc['transpiration_kg_m2']>1e-12):
            raise ValueError('empty greenhouse has crop output')
        for k,v in inc.items():self.totals[phase][k]+=v
        self.clock=end
        return inc

    def summary(self):
        e=self.contract['economics'];area=self.contract['facility']['floor_area_m2']
        total={k:sum(p[k] for p in self.totals.values()) for k in self.KEYS}
        costs=(total['heat_kwh_m2']*e['delivered_heat_eur_per_kwh']+
               total['light_kwh_m2']*e['electricity_eur_per_kwh']+total['co2_kg_m2']*e['co2_eur_per_kg']+
               self.totals['active']['days']*e['background_service_eur_per_m2_day']+self.event_cost_eur_m2)
        net=total['harvest_kg_m2']*(e['price_eur_per_kg_fresh_equivalent']-e['harvest_handling_eur_per_kg'])-costs
        return {'per_m2':total,'by_phase_per_m2':{k:dict(v) for k,v in self.totals.items()},
                'event_cost_eur_m2':self.event_cost_eur_m2,'synthetic_margin_eur':net*area,
                'occupied_m2_days':area*(self.totals['active']['days']+self.totals['cleanup']['days']),
                'scope':'single-compartment accrued ledger; not final campaign settlement'}


# Model outputs integrated by ResourceLedger.add_segment.
LEDGER_FLUXES = ('hBoilPipe', 'qLampIn', 'mcExtAir', 'mcFruitHar', 'mvCanAir')


def realise_independent_commands(contract, requested):
    """Validate one synchronous tick of every compartment under explicitly adequate supply.

    Copies exact per-unit fractional requests. GreenLight scales each fraction by
    its own pinned equipment capacity; no unsupported central curtailment exists.
    A future central bottleneck must use a new contract and allocation rule.
    """
    units = tuple(str(i) for i in range(contract['facility']['compartments']))
    if not isinstance(requested, dict) or set(requested) != set(units):
        raise ValueError('one command map required for every declared compartment')
    supply = contract['facility'].get('central_supply', {})
    if supply.get('model') != 'adequately_sized_no_cross_compartment_contention':
        raise ValueError('independent realisation requires adequate central supply contract')
    keys = {'uBoil','uRoof','uExtCo2','uLamp','uThScr'}
    realised = {}
    for unit in units:
        commands = requested[unit]
        if not isinstance(commands, dict) or set(commands) != keys:
            raise ValueError('invalid per-compartment command set: ' + unit)
        if any(not isinstance(x, Real) or isinstance(x, bool) or not math.isfinite(x) or not 0 <= x <= 1 for x in commands.values()):
            raise ValueError('per-compartment command outside [0,1]: ' + unit)
        realised[unit] = {k: float(commands[k]) for k in sorted(keys)}
    return realised
