"""Bounded reference or cached solver profiling; no model API calls."""
import argparse,cProfile,io,json,pstats,sys
from pathlib import Path
r=Path(__file__).resolve().parents[2];sys.path.insert(0,str(r))
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cached',action='store_true');a=p.parse_args();sys.path.insert(0,str(a.source))
from slowlab.v22.greenlight_reuse import ReusableGreenLight
c=json.loads((r/'configs/v22/task_contract_v1.json').read_text())
x=ReusableGreenLight(c,a.source,cached_solver=a.cached)
profile=cProfile.Profile();profile.enable()
for _ in range(12):x.step({'uBoil':.4,'uRoof':.05,'uExtCo2':.3,'uLamp':1.},x.clock+300)
profile.disable();stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumtime').print_stats(25)
result={'scope':'12 active steps, fixed commands/weather, excludes model load; cProfile adds overhead','cached':a.cached,'profile':stream.getvalue()}
a.out.write_text(json.dumps(result,indent=2)+'\n');print(stream.getvalue())
