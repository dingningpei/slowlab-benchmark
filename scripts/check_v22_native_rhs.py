"""Bounded native/reference derivative and multiday trajectory regression."""
import argparse,hashlib,json,sys,time,traceback
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--max-seconds',type=int,default=120);a=p.parse_args();sys.path.insert(0,str(a.source))
from slowlab.v22_greenlight_reuse import ReusableGreenLight,CROP_STATES,CROP_POOLS
from slowlab.v22_cached_solver import CachedGreenLightSolver
from slowlab.v22_resources import ResourceLedger,standby_commands
c=json.loads((r/'configs/v22_task_contract_v1.json').read_text())
started=time.monotonic();result={'status':'running','scope':'fixed-weather development fixture, not annual or empirical validation','thresholds':{'derivative_rtol':1e-10,'derivative_atol':1e-10,'state_rtol':1e-6,'state_atol':1e-3,'ledger_rtol':1e-5,'ledger_atol':1e-9},'comparisons':[]}
try:
 engines={mode:[ReusableGreenLight(c,a.source,78900.,mode,cached_solver=v,native_rhs=v) for v in (False,True)] for mode in ('active','empty')}
 result['derivatives']=[]
 for mode,pair in engines.items():
    native=pair[1]._cached_solver.rhs;ref=CachedGreenLightSolver(pair[0].model).rhs
    m=pair[0].model;cols=['Time']+[k for k in m.inputs if k!='Time'];d=m.input_data[cols].to_numpy();base=np.array([pair[0].state[k] for k in m.states])
    maxerr=0.
    for i,tair in enumerate((5.,10.,20.,30.,35.)):
        y=base.copy();y[list(m.states).index('tAir')]=tair
        for name in ('cmdHeat','cmdVent','cmdCo2','cmdLamp'):d[0,cols.index(name)]=float(i%2)
        left=ref(78900.,y,d);right=native(78900.,y,d)
        assert np.isfinite(left).all()
        assert np.allclose(left,right,rtol=1e-10,atol=1e-10),(mode,tair,float(np.max(abs(left-right))))
        maxerr=max(maxerr,float(np.max(abs(left-right))))
    result['derivatives'].append({'mode':mode,'cases':5,'max_abs_difference':maxerr,'source_sha256':native.source_sha256,'compiler':native.compiler_version,'flags':native.compiler_flags})
 ledgers=[ResourceLedger(c,78900.) for _ in range(2)]
 for l in ledgers:l.record_event('plant',78900.)
 totals=[0.,0.];worst=0.;steps=0
 def advance(pair,commands,phase):
    global steps,worst
    if time.monotonic()-started>a.max_seconds:raise RuntimeError(f'{a.max_seconds}-second test budget reached')
    for j,(x,l) in enumerate(zip(pair,ledgers)):
        begin=x.clock;timer=time.perf_counter();x.step(commands,begin+300);totals[j]+=time.perf_counter()-timer
        assert np.isfinite(x.model.full_sol.to_numpy(dtype=float)).all()
        l.add_segment(x.model.full_sol,begin,x.clock,phase)
    left=np.array(list(pair[0].state.values()));right=np.array(list(pair[1].state.values()))
    err=float(np.max(abs(left-right)));worst=max(worst,err)
    assert np.allclose(left,right,rtol=1e-6,atol=1e-3),('state',steps,err)
    for key,value in ledgers[0].summary()['per_m2'].items():
        assert np.isclose(value,ledgers[1].summary()['per_m2'][key],rtol=1e-5,atol=1e-9),('ledger',key)
    if phase!='active':
        for x in pair:
            assert all(x.state[k]==0 for k in CROP_POOLS)
            assert float(abs(x.model.full_sol[list(x.crop_flows)+['mvCanAir']]).to_numpy().max())<1e-10
    steps+=1
    if steps%96==0:print(json.dumps({'steps':steps,'elapsed':time.monotonic()-started,'max_abs_state_difference':worst}),flush=True)
 active=engines['active'];empty=engines['empty'];initial=[{k:x.state[k] for k in CROP_STATES} for x in active]
 for i in range(12):advance(active,{'uBoil':.4,'uRoof':.05+.1*(i%2),'uExtCo2':.3,'uLamp':float(i%2)},'active')
 for x,e,l in zip(active,empty,ledgers):
    e.state=dict(x.state);e.state.update({k:0. for k in (*CROP_POOLS,'tCanSum')});e.clock=x.clock;l.record_event('stop',x.clock)
 for i in range(576):
    # Same commands isolate solver error; production sensor/controller chain is separate.
    u=standby_commands(c,empty[0].state['tAir']);advance(empty,u,'cleanup')
 for x,e,init,l in zip(active,empty,initial,ledgers):
    x.state=dict(e.state);x.state.update(init);x.state['tCan']=x.state['tCan24']=x.state['tAir'];x.clock=e.clock;l.record_event('plant',x.clock)
 for i in range(12):advance(active,{'uBoil':.4,'uRoof':.05,'uExtCo2':.3,'uLamp':float(i%2)},'active')
 result.update(status='passed',steps=steps,max_abs_state_difference=worst,reference_step_seconds=totals[0],native_step_seconds=totals[1],speedup=totals[0]/totals[1],ledgers=[l.summary() for l in ledgers])
except Exception as exc:
 result.update(status='failed',error=repr(exc));traceback.print_exc()
finally:
 result['elapsed_seconds']=time.monotonic()-started
 result['hashes']={f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ['slowlab/v22_native_rhs.py','slowlab/v22_greenlight_reuse.py','scripts/check_v22_native_rhs.py','configs/v22_task_contract_v1.json']}
 a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if result['status']!='passed':sys.exit(1)
