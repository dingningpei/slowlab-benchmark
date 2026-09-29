#!/usr/bin/env python3
"""Two 5-minute steps, raw endpoints, single compartment, synthetic weather."""
import argparse
import contextlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from slowlab.greenlight_adapter import greenlight_raw_endpoint
from slowlab.online_observations import OnlineObservations
from slowlab.v22_greenlight_smoke import model_override, held_commands


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    sys.path.insert(0,str(args.source))
    from greenlight import GreenLight
    import numpy, scipy, pandas, numexpr
    c=json.loads((ROOT/'configs/v22_task_contract_v0.json').read_text())
    policy=json.loads((ROOT/'configs/v22_campaign_example_v0.json').read_text())['policy_a']
    definitions, facility=model_override(c,args.source)
    log=OnlineObservations({'tAir':'degC'}); events=[]; previous=None
    observed={'temperature':16.5,'rh':90.,'co2':450.,'solar':100.}
    started=time.monotonic()
    result={'scope':'10-minute development smoke, not full Phase1 or validation','events':events, 'versions':{m.__name__:m.__version__ for m in (numpy,scipy,pandas,numexpr)}}
    try:
        for tick in range(2):
            start=21600+300*tick; end=start+300
            commands=held_commands(policy,observed,(start/3600)%24)
            override=dict(facility)
            # Fixed weather is declared smoke forcing, never a historical replay.
            for name,value in {'tOut':10,'vpOut':1000,'co2Out':760,'wind':2,'tSky':0,'iGlob':100,
                               'dayRadSum':0,'isDay':1,'isDaySmooth':1,'uSide':0,'uBoilGro':0,
                               'uIntLamp':0,**commands}.items():
                override[name]={'definition':str(value)}
            override.update({'lampNoCons':{'definition':str(commands['uLamp'])},
                             'smoothLamp':{'definition':str(commands['uLamp'])},
                             'isDayInside':{'definition':'1'}, 'heatCorrection':{'definition':'0'}})
            if previous:
                for name,value in previous.items():
                    override.setdefault(name,{})['init']=repr(value)
            else:
                override['time_state']={'init':str(start)}
            override['options']={'t_start':str(start),'t_end':str(end),'solver':'LSODA','max_step':'30',
                                 'output_step':'300','clip_large_nums':'False','nans_to_zeros':'False'}
            load_started=time.monotonic()
            with open(os.devnull,'w') as sink, contextlib.redirect_stdout(sink):
                sim=GreenLight(base_path=str(definitions),input_prompt=[str(definitions/c['model']['entrypoint']),override])
                sim.load()
                # First controls must use actual initialized state, never guessed sensors.
                if previous is None:
                    init={name:float(value) for name,value in sim.init.items()}
                    observed=observe(init)
                    commands=held_commands(policy,observed,6)
                    for name,value in commands.items(): override[name]={'definition':str(value)}
                    override['lampNoCons']={'definition':str(commands['uLamp'])}
                    override['smoothLamp']={'definition':str(commands['uLamp'])}
                    sim=GreenLight(base_path=str(definitions),input_prompt=[str(definitions/c['model']['entrypoint']),override]);sim.load()
                loaded_at=time.monotonic()
                sim.solve()  # Deliberately do not call save()/run().
            solved_at=time.monotonic()
            previous=greenlight_raw_endpoint(sim.states_sol,list(sim.states),list(sim.states),expected_time=end)
            if any(not math.isfinite(v) for v in previous.values()): raise ValueError('nonfinite state')
            observed=observe(previous)
            if not -20 < observed['temperature'] < 60 or not 0 <= observed['rh'] <= 105:
                raise ValueError('physical smoke envelope exceeded')
            log.advance_to(end); log.record('0','tAir',measurement_time=end,available_at=end,value=observed['temperature'])
            events.append({'start_s':start,'end_s':end,'commands':commands,'climate':observed,'public_temperature':log.latest('0','tAir'),
                           'solver_nfev':sim.states_sol.nfev, 'load_seconds':loaded_at-load_started, 'solve_seconds':solved_at-loaded_at})
        result['status']='passed_bounded_smoke'
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      thread_environment={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS']})
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


def observe(state):
    t=state['tAir'];vp=state['vpAir'];rho=state['co2Air']
    return {'temperature':t,'rh':100*vp/(610.78*math.exp(17.2694*t/(t+238.3))),
            'co2':8.3144598*(t+273.15)*rho/(101325*44.01e-3),'solar':100.}

if __name__=='__main__':main()
