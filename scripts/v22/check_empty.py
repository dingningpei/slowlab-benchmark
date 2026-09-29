"""Bounded physical empty-mode check; does not skip the two-day cleanup gate."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
r=Path(__file__).resolve().parents[2];sys.path.insert(0,str(r))
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.source))
from slowlab.v22.greenlight_reuse import CropLifecycle,CROP_STATES,CROP_POOLS
c=json.loads((r/'configs/v22/task_contract_v0.json').read_text());start=time.monotonic()
x=CropLifecycle(c,a.source)
u={'uBoil':.4,'uRoof':.05,'uExtCo2':0.,'uLamp':0.}
x.step(u,x.engine.clock+300)
before=dict(x.engine.state);now=x.engine.clock
x.stop()
assert all(x.engine.state[k]==v for k,v in before.items() if k not in CROP_STATES)
assert x.engine.clock==now
try:x.replant()
except ValueError:pass
else:raise AssertionError('cleanup bypass')
checks=[]
for _ in range(2):
    x.step(u,x.engine.clock+300)
    df=x.engine.model.full_sol
    zero=list(x.engine.crop_flows)+['lai','aCan','mvCanAir','hCanAir','lCanAir','rParSunCan','rNirSunCan','rParLampCan','rNirLampCan','rFirLampCan','rPipeCan','rCanFlr','rCanCovIn','rCanSky','rCanThScr','rCanBlScr','rGroPipeCan']
    values=df[zero].to_numpy(dtype=float)
    assert np.isfinite(df.to_numpy(dtype=float)).all(), [k for k in df if not np.isfinite(df[k].to_numpy(dtype=float)).all()]
    maxima={k:float(df[k].abs().max()) for k in zero}
    assert all(v<=1e-10 for v in maxima.values()), maxima
    assert all(x.engine.state[k]==0 for k in CROP_POOLS)
    checks.append({'clock':x.engine.clock,'zero_crop_flux_max':max(maxima.values()),'delivered_heat_w_m2':float(df['hBoilPipe'].mean())})
result={'status':'passed','scope':'active 300s then empty 600s; fixed weather; no full cleanup simulated','checks':checks,'facility_continuity_at_stop':True,'premature_replant_rejected':True,'elapsed_seconds':time.monotonic()-start,'source_sha256':hashlib.sha256((r/'slowlab/v22/greenlight_reuse.py').read_bytes()).hexdigest(),'limitations':['Replant transition tested separately with synthetic boundary state, not a two-day physical rollout.','Resource ledger and multicompartment scheduler not implemented.']}
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
