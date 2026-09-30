"""Reuse one parsed GreenLight model; each step holds only present inputs.

A short-run development adapter, not yet the complete V2.2 campaign executor.
"""
from __future__ import annotations
import contextlib
import csv
import json
import tempfile
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

from .greenlight_adapter import greenlight_raw_endpoint
from .greenlight_source import resolve_greenlight_source
from .greenlight_smoke import model_override, _nodes


def require_dynamic_inputs(model, required):
    """Fail closed if GreenLight treated declared inputs as fixed auxiliaries."""
    missing = set(required) - set(model.inputs)
    if missing:
        raise RuntimeError("dynamic control inputs were not registered: " + ", ".join(sorted(missing)))


def require_live_canopy(state):
    """This active-crop adapter cannot yet advance an empty compartment.

    Reject before native capCan=capLeaf*sla*cLeaf becomes singular. Do not
    silently add ghost leaves or enable NaN replacement as a lifecycle fix.
    """
    leaf = state.get('cLeaf')
    if leaf is None or not math.isfinite(leaf) or leaf <= 0:
        raise RuntimeError('crop-absent mode is not implemented: positive cLeaf required')


CROP_POOLS = ('cBuf', 'cFruit', 'cLeaf', 'cStem')
CROP_STATES = (*CROP_POOLS, 'tCanSum', 'tCan24', 'tCan')


def empty_crop_override(definitions, contract):
    """Remove crop processes, retaining native zero-LAI radiation transmission.

    Source hashes were verified by model_override. Never zero general greenhouse
    CO2 flows (mcExtAir/mcAirOut/etc.). Only crop-file mc* flows are disabled.
    """
    crop = _nodes(json.loads((definitions / 'vanthoor_2011/crop_vanthoor_2011_chapter_9_simplified.json').read_text()))
    flows = tuple(k for k in crop if k.startswith('mc'))
    override = {k: {'definition': '0'} for k in (*CROP_STATES, *flows)}
    override.update({k: {'definition': '0', 'init': '0'} for k in (*CROP_POOLS, 'tCanSum')})
    # Ground heating is disabled in this contract. Its native canopy view factor
    # does not vanish with LAI, so explicitly remove that unused canopy exchange.
    override['rGroPipeCan'] = {'definition': '0'}
    # Compensation point is undefined without leaves; zero is an inactive
    # placeholder, not a physiological estimate. Disable its consumers too.
    for key in ('gamma', 'p', 'r'):
        override[key] = {'definition': '0'}
    return override, flows


class ReusableGreenLight:
    COMMANDS = {'uBoil':'cmdHeat','uRoof':'cmdVent','uExtCo2':'cmdCo2','uLamp':'cmdLamp'}

    def __init__(self, contract, source: Path | None, start=21600., mode="active", cached_solver=False, native_rhs=False, *, weather=None, weather_origin_utc: datetime | None = None, soil_boundary_c: float | None = None, local_clock_offset_seconds: float = 0,
                 array_output: bool = False):
        if mode not in ("active", "empty"):
            raise ValueError("mode must be active or empty")
        if native_rhs and not cached_solver:
            raise ValueError("native RHS requires cached solver")
        if array_output and not cached_solver:
            raise ValueError("array output requires cached solver")
        if (weather is None) != (weather_origin_utc is None):
            raise ValueError("weather and weather_origin_utc must be supplied together")
        if weather_origin_utc is not None and (weather_origin_utc.tzinfo is None or weather_origin_utc.utcoffset() != timedelta(0)):
            raise ValueError("weather_origin_utc must be timezone-aware UTC")
        if soil_boundary_c is not None and (isinstance(soil_boundary_c, bool) or not isinstance(soil_boundary_c, (int, float)) or not math.isfinite(soil_boundary_c) or not -20 <= soil_boundary_c <= 40):
            raise ValueError('soil_boundary_c must be a finite deep-soil scenario in [-20, 40] °C')
        self.soil_boundary_c = None if soil_boundary_c is None else float(soil_boundary_c)
        if isinstance(local_clock_offset_seconds,bool) or not isinstance(local_clock_offset_seconds,(int,float)) or not math.isfinite(local_clock_offset_seconds) or not 0<=local_clock_offset_seconds<86400:
            raise ValueError('invalid local clock offset')
        self.local_clock_offset_seconds = float(local_clock_offset_seconds)
        self._weather = weather
        self._weather_origin_utc = weather_origin_utc
        self.mode = mode
        self.COMMANDS = dict(type(self).COMMANDS)
        sampled_screen = contract['controller'].get('thermal_screen', {}).get('implementation') == 'external_sample_and_hold'
        if sampled_screen:
            self.COMMANDS['uThScr'] = 'cmdScreen'
        source = resolve_greenlight_source(source, contract)
        from greenlight import GreenLight
        definitions, override = model_override(contract, source)
        self.crop_flows = ()
        if mode == "empty":
            empty, self.crop_flows = empty_crop_override(definitions, contract)
            override.update(empty)
        bootstrap_weather = (weather.at_utc(weather_origin_utc + timedelta(seconds=start)).greenlight_inputs()
                             if weather is not None else {})
        for name,value in {'tOut':10,'vpOut':1000,'co2Out':760,'wind':2,'tSky':0,'iGlob':100,
                           'dayRadSum':0,'isDay':1,'isDaySmooth':1,'uSide':0,'uBoilGro':0,
                           'uIntLamp':0,'heatCorrection':0}.items():
            override[name]={'definition':str(value)}
        if self.soil_boundary_c is not None:
            override['tSoOut']={'type':'input','definition':repr(self.soil_boundary_c),
                                'description':'Explicit deep-soil boundary scenario, not a measured greenhouse value'}
        for name,value in bootstrap_weather.items():
            override[name]={'type':'input','definition':repr(value),'description':'Current private-executor weather forcing'}
        for command,channel in self.COMMANDS.items():
            override[channel]={'type':'input','definition':'0','unit':'-','description':'Current held command'}
            override[command]={'definition':channel}
        override['cmdDay']={'type':'input','definition':'1','unit':'-','description':'Current clock day indicator'}
        override.update({'lampNoCons':{'definition':'cmdLamp'},'smoothLamp':{'definition':'cmdLamp'},
                         'isDayInside':{'definition':'cmdDay'},'time_state':{'init':str(start)}})
        if sampled_screen:
            override['isDay'] = {'definition': 'cmdDay'}
            override['isDaySmooth'] = {'definition': 'cmdDay'}
        override['options']={'t_start':str(start),'t_end':str(start+300),'solver':'LSODA','max_step':'30',
                             'output_step':'300','clip_large_nums':'False','nans_to_zeros':'False',
                             'interpolation':'left'}
        # The official loader registers true inputs only when data is loaded.
        # Supply one CURRENT bootstrap row before expression compilation; no
        # future command/weather row is needed or retained.
        self.bootstrap_inputs = {'Time': float(start),
                                 **{channel: 0.0 for channel in self.COMMANDS.values()},
                                 'cmdDay': float(6 <= ((start+self.local_clock_offset_seconds)/3600) % 24 < 22),
                                 **bootstrap_weather,
                                 **({'tSoOut': self.soil_boundary_c} if self.soil_boundary_c is not None else {})}
        with tempfile.TemporaryDirectory(prefix='slowlab-v22-inputs-') as temporary:
            input_path = Path(temporary) / 'current_controls.csv'
            with input_path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.bootstrap_inputs))
                writer.writeheader()
                writer.writerow(self.bootstrap_inputs)
            with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):
                self.model=GreenLight(base_path=str(definitions),input_prompt=[
                    str(definitions/contract['model']['entrypoint']), override, str(input_path)])
                self.model.load()
        require_dynamic_inputs(self.model, [*self.COMMANDS.values(), 'cmdDay', *bootstrap_weather,
                                            *(['tSoOut'] if self.soil_boundary_c is not None else [])])
        self.names=tuple(self.model.states)
        self.state={k:float(self.model.init[k]) for k in self.names}
        self.clock=float(start)
        # Store native constant input defaults; no future forcing rows accepted.
        self._inputs=self.model.input_data.iloc[0].to_dict()
        self.load_count = 1
        self._load_log=self.model.log
        self.segment_log=''
        self._cached_solver = None
        if cached_solver:
            from .cached_solver import CachedGreenLightSolver
            self._cached_solver = CachedGreenLightSolver(self.model,array_output=array_output)
            if native_rhs:
                from .native_rhs import NativeRHS
                self._cached_solver.rhs = NativeRHS(self.model)

    def step(self, commands, end_time):
        import pandas as pd
        if set(commands)!=set(self.COMMANDS) or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=1 for v in commands.values()):
            raise ValueError('commands must be finite fractions with exact keys')
        if not math.isfinite(end_time) or end_time <= self.clock or end_time-self.clock > 300:
            raise ValueError('step must end within the next 300 seconds')
        if self.mode == "active":
            require_live_canopy(self.state)
        elif any(self.state[k] != 0 for k in CROP_POOLS):
            raise ValueError("empty mode requires zero crop pools")
        mdl=self.model
        row=dict(self._inputs);row['Time']=self.clock
        row.update({self.COMMANDS[k]:v for k,v in commands.items()})
        row['cmdDay']=float(6 <= ((self.clock+self.local_clock_offset_seconds)/3600)%24 < 22)
        if self.soil_boundary_c is not None:
            row['tSoOut']=self.soil_boundary_c
        if self._weather is not None:
            # Historical interval means are hidden simulator forcing, never
            # agent-visible observations at the interval's left edge.
            row.update(self._weather.at_utc(self._weather_origin_utc + timedelta(seconds=self.clock)).greenlight_inputs())
        mdl.input_data=pd.DataFrame([row])
        mdl.options.update(t_start=str(self.clock),t_end=str(end_time))
        mdl.init.update(self.state)
        mdl.full_sol=pd.DataFrame(columns=["Time"])
        mdl.log=''  # Retain this segment separately, not an ever-growing log.
        before=dict(self.state)
        with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):
            if self._cached_solver is None:
                mdl.solve()
            else:
                self._cached_solver.solve()
        self.segment_log=mdl.log
        after=greenlight_raw_endpoint(mdl.states_sol,self.names,self.names,expected_time=end_time)
        if any(float(mdl.states_sol.y[i][0]) != before[name] for i,name in enumerate(self.names)):
            raise ValueError('solver did not continue from prior state')
        self.state=after;self.clock=float(end_time)
        return dict(after)


class CropLifecycle:
    """Single-unit physical transition primitive; not the campaign scheduler.

    Both compiled modes are cached. Cleanup blocks planting for the contracted
    duration; callers still advance all 300s physical steps during that interval.
    Costs and harvested output must be settled by the campaign ledger separately.
    """
    def __init__(self, contract, source, start=21600., cached_solver=False, native_rhs=False, *,
                 weather=None, weather_origin_utc=None, soil_boundary_c=None, local_clock_offset_seconds=0, array_output=False, initially_empty=False):
        self.engines = {m: ReusableGreenLight(contract, source, start, m, cached_solver=cached_solver,
                                              native_rhs=native_rhs, weather=weather,
                                              weather_origin_utc=weather_origin_utc,
                                              soil_boundary_c=soil_boundary_c,
                                              local_clock_offset_seconds=local_clock_offset_seconds,
                                              array_output=array_output)
                        for m in ('active', 'empty')}
        self.mode = 'empty' if initially_empty else 'active'
        self.crop_initial = {k: self.engines['active'].state[k] for k in CROP_STATES}
        self.cleanup_seconds = contract['budget']['cleanup_days'] * 86400
        self.ready_at = float(start) if initially_empty else None
        self.events = []

    @property
    def engine(self):
        return self.engines[self.mode]

    def step(self, commands, end_time):
        return self.engine.step(commands, end_time)

    def stop(self):
        if self.mode != 'active':
            raise ValueError('no active crop to stop')
        state = dict(self.engine.state)
        now = self.engine.clock
        self.events.append({'event': 'stop', 'time': now,
                            'removed_crop': {k: state[k] for k in CROP_STATES},
                            'standing_crop_revenue': 0.0})
        state.update({k: 0.0 for k in (*CROP_POOLS, 'tCanSum')})
        self.engines['empty'].state = state
        self.engines['empty'].clock = now
        self.mode = 'empty'
        self.ready_at = now + self.cleanup_seconds

    def replant(self):
        if self.mode != 'empty' or self.engine.clock < self.ready_at:
            raise ValueError('replant requires completed cleanup in empty mode')
        state, now = dict(self.engine.state), self.engine.clock
        state.update(self.crop_initial)
        state['tCan'] = state['tCan24'] = state['tAir']
        self.engines['active'].state = state
        self.engines['active'].clock = now
        self.mode = 'active'
        self.ready_at = None
        self.events.append({'event': 'replant' if self.events else 'plant', 'time': now})
