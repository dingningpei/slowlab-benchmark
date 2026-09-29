"""Bounded independent-replica prefix check; no future commands sent to model."""
import argparse,json,sys
from pathlib import Path
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
sys.path.insert(0,str(args.source))
from slowlab.v22_greenlight_reuse import ReusableGreenLight
c=json.loads((r/'configs/v22_task_contract_v0.json').read_text())
src=args.source
prefix=[{'uBoil':.8,'uRoof':.01,'uExtCo2':.5,'uLamp':1.},{'uBoil':.4,'uRoof':.1,'uExtCo2':.1,'uLamp':0.}]
branches=[]
for future in ({'uBoil':0.,'uRoof':0.,'uExtCo2':0.,'uLamp':0.},{'uBoil':1.,'uRoof':1.,'uExtCo2':1.,'uLamp':1.}):
    engine=ReusableGreenLight(c,src)
    schedule=[*prefix,future]
    states=[]
    for i,command in enumerate(schedule[:2]):
        states.append(engine.step(command,21600+300*(i+1)))
    branches.append(states)
assert branches[0]==branches[1]
result={'scope':'two independent replicas, same executed prefix, different unexecuted future command plans','state_count':len(branches[0][0]),'prefix_steps':2,'exactly_equal':True,'limitation':'adapter boundary only; not the complete future weather/agent pipeline'}
args.out.parent.mkdir(parents=True,exist_ok=True)
args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
