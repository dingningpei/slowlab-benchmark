"""Read-only Full/Endpoint feedback filter over an executor-owned sensor store.

A Phase-1 permission primitive; a future agent API must expose only this view,
never its underlying store or mutator methods.
"""
from __future__ import annotations
import copy


class FeedbackView:
    def __init__(self, mode, observations, units):
        if mode not in ('full','endpoint'):
            raise ValueError('unknown feedback mode')
        self._mode=mode
        self._observations=observations
        self._units=frozenset(units)
        if not self._units or any(not isinstance(x,str) or not x for x in self._units):
            raise ValueError('invalid compartment identifiers')
        self._status={unit:{'unit':unit,'phase':'idle','clock':0.0,'event':None,'run_index':0} for unit in self._units}
        self._released={}
        self._baseline={}

    def _require_unit(self,unit):
        if unit not in self._units:raise ValueError('unknown compartment')

    def operational_status(self,unit):
        self._require_unit(unit)
        return dict(self._status[unit])

    def executor_set_status(self,unit,phase,event=None,ledger=None):
        """Executor-owned mutation, never included in the agent tool surface."""
        self._require_unit(unit)
        if phase not in ('idle','active','cleanup') or event not in (None,'start','stop','normal_completion','safety_stop'):
            raise ValueError('invalid operational status')
        previous=self._status[unit]
        if event=='start' and (phase!='active' or previous['phase']=='active'):
            raise ValueError('start requires a free or cleaned compartment')
        run_index=previous['run_index']+1 if event=='start' else previous['run_index']
        if event=='start':
            if run_index>1 and ledger is None:
                raise ValueError('replant requires resource baseline')
            if ledger is not None:
                if ledger.clock!=self._observations.clock:raise ValueError('resource baseline clock mismatch')
                summary=ledger.summary()
                just_planted=(ledger.events and ledger.events[-1]['kind']=='plant'
                              and ledger.events[-1]['time']==ledger.clock)
                plant_cost=ledger.events[-1]['cost_eur_m2'] if just_planted else 0.
                self._baseline[(unit,run_index)]={'per_m2':dict(summary['per_m2']),
                                                   'event_cost_eur_m2':summary['event_cost_eur_m2']-plant_cost}
        self._status[unit]={'unit':unit,'phase':phase,'clock':self._observations.clock,
                            'event':event,'run_index':run_index}

    def history(self,unit,variable,*,as_of=None,start=0.,end=None):
        self._require_unit(unit)
        if self._mode!='full':
            raise PermissionError('endpoint feedback cannot read running science records')
        return self._observations.history(unit,variable,as_of=as_of,start=start,end=end)

    def executor_release_final(self,unit,reason,ledger):
        """Release accrued final aggregates only after an actual stop/completion."""
        self._require_unit(unit)
        status=self._status[unit]
        if status['phase']!='cleanup' or status['event']!=reason or reason not in ('stop','normal_completion','safety_stop'):
            raise ValueError('no corresponding closed crop')
        key=(unit,status['run_index'])
        if key in self._released:raise ValueError('final aggregate already released')
        totals=ledger.summary()
        if ledger.clock!=self._observations.clock:
            raise ValueError('final ledger and observation clocks differ')
        allowed=('harvest_kg_m2','heat_kwh_m2','light_kwh_m2','co2_kg_m2','days')
        baseline=self._baseline.get(key,{'per_m2':{name:0. for name in allowed},'event_cost_eur_m2':0.})
        self._released[key]={'unit':unit,'run_index':status['run_index'],'reason':reason,
                              'closed_at':self._observations.clock,
                              'accrued':{name:totals['per_m2'][name]-baseline['per_m2'][name] for name in allowed},
                              'event_cost_eur_m2':totals['event_cost_eur_m2']-baseline['event_cost_eur_m2']}

    def final_aggregate(self,unit,run_index=None):
        self._require_unit(unit)
        if run_index is None:
            candidates=[index for name,index in self._released if name==unit]
            run_index=max(candidates) if candidates else None
        payload=self._released.get((unit,run_index))
        return copy.deepcopy(payload) if payload is not None else None
