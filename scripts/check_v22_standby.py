"""Single-thread, wall-time-bounded two-day cleanup/replant integration gate."""
import argparse,json,sys,time,hashlib,traceback
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.source))
from slowlab.v22_greenlight_reuse import CropLifecycle,CROP_STATES
from slowlab.v22_resources import ResourceLedger,standby_commands
from slowlab.online_observations import OnlineObservations
c=json.loads((r/'configs/v22_task_contract_v1.json').read_text())
started=time.monotonic();result={'status':'running','contract_id':c['contract_id'],'scope':'one compartment, fixed-weather fixture, two-day cleanup + idle + replant; not weather validation','threads':1}
ledger=None
try:
    x=CropLifecycle(c,a.source)
    ledger=ResourceLedger(c,x.engine.clock);ledger.record_event('plant',x.engine.clock)
    obs=OnlineObservations({'air_temperature_c':'degC'})
    def observe():
        now=x.engine.clock;obs.advance_to(now)
        obs.record('unit0','air_temperature_c',measurement_time=now,available_at=now,value=x.engine.state['tAir'])
    observe();temps=[x.engine.state['tAir']];steps=0;heat_exact=0.;max_zero=0.
    def advance(command,phase):
        global steps,heat_exact,max_zero
        if time.monotonic()-started>120:raise RuntimeError('120-second compute bound reached; resume required')
        start=x.engine.clock
        x.step(command,start+300)
        df=x.engine.model.full_sol
        bad=[k for k in df if not np.isfinite(df[k].to_numpy(dtype=float)).all()]
        if bad:raise RuntimeError('nonfinite outputs: '+str(bad))
        ledger.add_segment(df,start,x.engine.clock,phase)
        heat_exact+=command['uBoil']*c['facility']['capacity']['heating_w_m2']*300/3.6e6
        if phase!='active':
            zero=[*x.engine.crop_flows,'mvCanAir','hCanAir','lCanAir','rParSunCan','rNirSunCan','rCanFlr','rPipeCan','rFirLampCan']
            error=float(np.abs(df[zero].to_numpy(dtype=float)).max());max_zero=max(max_zero,error)
            if error>1e-10:raise RuntimeError('nonzero empty crop flux')
        observe();temps.append(x.engine.state['tAir']);steps+=1
        if steps%96==0:print(json.dumps({'steps':steps,'elapsed_seconds':time.monotonic()-started,'temperature_c':temps[-1]}),flush=True)
    advance({'uBoil':.4,'uRoof':.05,'uExtCo2':.3,'uLamp':1.},'active')
    before=dict(x.engine.state);x.stop();ledger.record_event('stop',x.engine.clock)
    assert all(x.engine.state[k]==v for k,v in before.items() if k not in CROP_STATES)
    ready=x.ready_at
    try:x.replant()
    except ValueError:pass
    else:raise AssertionError('premature replant accepted')
    while x.engine.clock<ready:
        sample=obs.latest('unit0','air_temperature_c')
        assert sample['available_at']<=x.engine.clock
        advance(standby_commands(c,sample['value']),'cleanup')
    assert ledger.totals['cleanup']['days']==2. or abs(ledger.totals['cleanup']['days']-2.)<1e-12
    advance(standby_commands(c,obs.latest('unit0','air_temperature_c')['value']),'idle')
    before=dict(x.engine.state);now=x.engine.clock
    x.replant();ledger.record_event('plant',now)
    assert all(x.engine.state[k]==v for k,v in before.items() if k not in CROP_STATES)
    assert x.engine.state['tCan']==x.engine.state['tCan24']==before['tAir']
    advance({'uBoil':.4,'uRoof':.05,'uExtCo2':.3,'uLamp':1.},'active')
    assert abs(ledger.summary()['per_m2']['heat_kwh_m2']-heat_exact)<1e-10
    result.update(status='passed',steps=steps,temperature_range_c=[min(temps),max(temps)],empty_flux_max_abs=max_zero,heat_quadrature_error=abs(ledger.summary()['per_m2']['heat_kwh_m2']-heat_exact),facility_continuity=True,replant_after_full_cleanup=True)
except Exception as exc:
    result.update(status='failed',error=type(exc).__name__+': '+str(exc));traceback.print_exc()
finally:
    result['elapsed_seconds']=time.monotonic()-started
    if ledger:result['ledger']=ledger.summary()
    result['hashes']={f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ['configs/v22_task_contract_v1.json','slowlab/v22_resources.py','slowlab/v22_greenlight_reuse.py','scripts/check_v22_standby.py']}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if result['status']!='passed':sys.exit(1)
