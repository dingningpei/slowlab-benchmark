"""Bounded all-output comparison, both modes and day-boundary control changes."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parents[2];sys.path.insert(0,str(r))
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.source))
from slowlab.v22.greenlight_reuse import ReusableGreenLight
from slowlab.v22.resources import ResourceLedger
c=json.loads((r/'configs/v22/task_contract_v1.json').read_text());rows=[];result={'status':'running','scope':'12 steps each active/empty across 22:00 with held-command changes, all outputs compared','steps':rows}
try:
 for mode in ('active','empty'):
    engines=[ReusableGreenLight(c,a.source,78900.,mode,cached_solver=v) for v in (False,True)]
    ledgers=[ResourceLedger(c,78900.) for _ in engines]
    for i in range(12):
        commands={'uBoil':(.4 if i%2 else .1),'uRoof':(.05 if i%3 else .3),'uExtCo2':(.3 if mode=='active' else 0.),'uLamp':(float(i%2) if mode=='active' else 0.)}
        timings=[]
        for x,l in zip(engines,ledgers):
            start=x.clock;t=time.perf_counter();x.step(commands,start+300);timings.append(time.perf_counter()-t)
            l.add_segment(x.model.full_sol,start,x.clock,'active' if mode=='active' else 'cleanup')
        left,right=[x.model.full_sol for x in engines]
        assert list(left)==list(right)
        assert left.shape==right.shape, (left.shape,right.shape)
        error=float(np.max(np.abs(left.to_numpy()-right.to_numpy())))
        assert np.array_equal(left.to_numpy(),right.to_numpy()),error
        assert ledgers[0].summary()==ledgers[1].summary()
        rows.append({'mode':mode,'tick':i,'max_abs_all_output_difference':error,'reference_seconds':timings[0],'cached_seconds':timings[1]})
 result['status']='passed';result['speedup']=sum(x['reference_seconds'] for x in rows)/sum(x['cached_seconds'] for x in rows)
except Exception as exc:
 result.update(status='failed',error=repr(exc));raise
finally:
 result['hashes']={f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ['slowlab/v22/cached_solver.py','slowlab/v22/greenlight_reuse.py','scripts/v22/check_cached_solver.py']}
 a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
