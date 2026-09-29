"""Bounded paired lifecycle validation with sampled thermal-screen control."""
import argparse,json,sys,time,hashlib,math,traceback
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.source))
from slowlab.greenlight_reuse import CropLifecycle,CROP_STATES
from slowlab.controller import commands_from_observations
from slowlab.resources import ResourceLedger
from slowlab.online_observations import OnlineObservations
c=json.loads((r/'configs/task_contract_v2.json').read_text());policy=json.loads((r/'configs/campaign_example_v0.json').read_text())['policy_a']
startwall=time.monotonic();result={'status':'running','scope':'one compartment, independent sensor-controller branches, fixed weather; reference/native comparison','control_ticks':[]}
try:
    engines=[CropLifecycle(c,a.source,start=78900.,cached_solver=v,native_rhs=v) for v in (False,True)]
    stores=[OnlineObservations(c['observations']['controller_channels']) for _ in engines]
    ledgers=[ResourceLedger(c,78900.) for _ in engines]
    events=[[],[]]
    def observe(x,s):
        t=x.engine.state['tAir'];v={'air_temperature_c':t,'relative_humidity_pct':100*x.engine.state['vpAir']/(610.78*math.exp(17.2694*t/(t+238.3))),'co2_ppm':8.3144598*(t+273.15)*x.engine.state['co2Air']/(101325*44.01e-3),'outdoor_temperature_c':10.,'solar_radiation_w_m2':100.}
        # Initial deterministic virtual readings; later use exact current raw endpoint auxiliaries.
        df=x.engine.model.full_sol
        if not df.empty and float(df['Time'].iloc[-1])==x.engine.clock:
            v['relative_humidity_pct']=float(df['rhIn'].iloc[-1]);v['co2_ppm']=float(df['co2InPpm'].iloc[-1])
        s.advance_to(x.engine.clock)
        for k,value in v.items():s.record('a',k,measurement_time=x.engine.clock,available_at=x.engine.clock,value=value)
    for x,s,l in zip(engines,stores,ledgers):observe(x,s);l.record_event('plant',78900.)
    maxerror=0.;steps=0;range_screen=[1.,0.]
    def advance(phase):
        global maxerror,steps
        if time.monotonic()-startwall>120:raise RuntimeError('120-second compute limit')
        requests=[]
        for j,(x,s,l) in enumerate(zip(engines,stores,ledgers)):
            now=x.engine.clock;command,event=commands_from_observations(c,s,'a',phase=phase,policy=policy)
            requests.append(command);x.step(command,now+300);df=x.engine.model.full_sol
            for actuator,channel in x.engine.COMMANDS.items():
                assert channel in x.engine.model.inputs
                assert np.all(df[actuator].to_numpy()==command[actuator]),('unheld actuator',actuator)
            assert np.all(df['isDay'].to_numpy()==float(6<=(now/3600)%24<22))
            assert np.isfinite(df.to_numpy(dtype=float)).all()
            event['end']=x.engine.clock;event['realised']=dict(command);events[j].append(event)
            l.add_segment(df,now,x.engine.clock,phase);observe(x,s)
        assert requests[0]==requests[1],('controller divergence',steps)
        range_screen[0]=min(range_screen[0],requests[0]['uThScr']);range_screen[1]=max(range_screen[1],requests[0]['uThScr'])
        left,right=[np.array(list(x.engine.state.values())) for x in engines]
        maxerror=max(maxerror,float(np.max(abs(left-right))))
        assert np.allclose(left,right,rtol=1e-6,atol=1e-3)
        for k,v in ledgers[0].summary()['per_m2'].items():assert np.isclose(v,ledgers[1].summary()['per_m2'][k],rtol=1e-5,atol=1e-9)
        steps+=1
        if steps%96==0:print(json.dumps({'steps':steps,'elapsed_seconds':time.monotonic()-startwall}),flush=True)
    for _ in range(12):advance('active')
    for x,l in zip(engines,ledgers):
        before=dict(x.engine.state);x.stop();l.record_event('stop',x.engine.clock)
        assert all(x.engine.state[k]==v for k,v in before.items() if k not in CROP_STATES)
        try:x.replant()
        except ValueError:pass
        else:raise AssertionError('early replant accepted')
    for _ in range(576):advance('cleanup')
    advance('idle')
    for x,l in zip(engines,ledgers):
        before=dict(x.engine.state);x.replant();l.record_event('plant',x.engine.clock)
        assert all(x.engine.state[k]==v for k,v in before.items() if k not in CROP_STATES)
    for _ in range(12):advance('active')
    result.update(status='passed',steps=steps,max_abs_state_difference=maxerror,screen_command_range=range_screen,ledger=ledgers[1].summary(),control_ticks=[events[1][0],events[1][1],events[1][12],events[1][-1]],all_controls_held=True,full_cleanup_replant=True)
except Exception as exc:
    result.update(status='failed',error=repr(exc));traceback.print_exc()
finally:
    result['elapsed_seconds']=time.monotonic()-startwall
    result['hashes']={f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ['configs/task_contract_v2.json','slowlab/controller.py','slowlab/greenlight_reuse.py','scripts/check_screen_controller.py']}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('control_ticks','hashes')}),flush=True)
if result['status']!='passed':sys.exit(1)
