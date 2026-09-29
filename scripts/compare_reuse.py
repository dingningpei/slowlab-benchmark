#!/usr/bin/env python3
"""Bounded six-step comparison: rebuilt models versus one parsed model."""
import argparse,contextlib,hashlib,json,math,os,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from slowlab.greenlight_smoke import model_override,held_commands
from slowlab.greenlight_adapter import greenlight_raw_endpoint
from slowlab.greenlight_reuse import ReusableGreenLight


def observe(s):
    t=s['tAir'];return {'temperature':t,'rh':100*s['vpAir']/(610.78*math.exp(17.2694*t/(t+238.3))),
                      'co2':8.3144598*(t+273.15)*s['co2Air']/(101325*44.01e-3),'solar':100.}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    sys.path.insert(0,str(args.source))
    from greenlight import GreenLight
    import numpy as np
    contract=json.loads((ROOT/'configs/task_contract_v0.json').read_text())
    policy=json.loads((ROOT/'configs/campaign_example_v0.json').read_text())['policy_a']
    definitions,facility=model_override(contract,args.source)
    t0=time.monotonic();engine=ReusableGreenLight(contract,args.source)
    load_s=time.monotonic()-t0
    reference=dict(engine.state);steps=[]
    result={'scope':'six 300s steps with synthetic weather; not whole Phase1','load_once_seconds':load_s,'steps':steps}
    try:
        for tick in range(6):
            start=21600+300*tick;end=start+300
            commands=held_commands(policy,observe(reference),(start/3600)%24)
            # Deliberately exercise command changes, including zero commands.
            if tick==2:commands={'uBoil':0.,'uRoof':0.4,'uExtCo2':0.,'uLamp':0.}
            if tick==3:commands={'uBoil':0.6,'uRoof':0.1,'uExtCo2':0.2,'uLamp':1.}
            override=dict(facility)
            for name,value in {'tOut':10,'vpOut':1000,'co2Out':760,'wind':2,'tSky':0,'iGlob':100,
                               'dayRadSum':0,'isDay':1,'isDaySmooth':1,'uSide':0,'uBoilGro':0,'uIntLamp':0,
                               'isDayInside':1,'heatCorrection':0,'lampNoCons':commands['uLamp'],
                               'smoothLamp':commands['uLamp'],**commands}.items():override[name]={'definition':str(value)}
            for name,value in reference.items():override.setdefault(name,{})['init']=repr(value)
            override['options']={'t_start':str(start),'t_end':str(end),'solver':'LSODA','max_step':'30','output_step':'300','clip_large_nums':'False','nans_to_zeros':'False'}
            begun=time.monotonic()
            with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):
                sim=GreenLight(base_path=str(definitions),input_prompt=[str(definitions/contract['model']['entrypoint']),override]);sim.load();sim.solve()
            reload_s=time.monotonic()-begun
            reference=greenlight_raw_endpoint(sim.states_sol,list(sim.states),list(sim.states),expected_time=end)
            begun=time.monotonic();actual=engine.step(commands,end);reuse_s=time.monotonic()-begun
            names=list(reference);expected=np.array([reference[k] for k in names]);got=np.array([actual[k] for k in names])
            close=np.allclose(got,expected,rtol=1e-6,atol=1e-3)
            if len(engine.model.input_data) != 1 or float(engine.model.input_data.iloc[0]['Time']) != start:
                raise ValueError('input table includes non-current records')
            steps.append({'end_seconds':end,'commands':commands,'state_count':len(names),
                          'max_abs_state_difference':float(np.max(np.abs(got-expected))),
                          'matches_at_solver_tolerance':bool(close),'current_input_rows':len(engine.model.input_data),'reload_seconds':reload_s,'reuse_seconds':reuse_s,
                          'climate':observe(actual)})
            if not close:raise ValueError('reused model differs beyond solver tolerance')
        result['status']='passed_comparison'
        result['model_load_count']=engine.load_count
        result['registered_inputs']=list(engine.model.inputs)
        result['bootstrap_inputs']=engine.bootstrap_inputs
        result['speedup_excluding_initialization']=sum(s['reload_seconds'] for s in steps)/sum(s['reuse_seconds'] for s in steps)
    except Exception as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        result['wall_seconds']=time.monotonic()-t0;result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
