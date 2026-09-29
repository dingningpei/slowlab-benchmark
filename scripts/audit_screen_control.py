"""Audit whether the native screen obeys the common sample-and-hold contract."""
import argparse,hashlib,json,sys
from pathlib import Path
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.source))
from slowlab.greenlight_reuse import ReusableGreenLight
from slowlab.greenlight_smoke import _nodes
c=json.loads((r/'configs/task_contract_v1.json').read_text())
x=ReusableGreenLight(c,a.source,cached_solver=True,native_rhs=True)
x.step({'uBoil':.4,'uRoof':.05,'uExtCo2':.3,'uLamp':1.},x.clock+300)
df=x.model.full_sol
path=a.source/'greenlight/models/katzin_2021/definition/extension_greenhouse_katzin_2021_vanthoor_2011.json'
nodes=_nodes(json.loads(path.read_text()))
result={'status':'contract_gap','scope':'native screen actuator within one fixed-command 300s segment','contract_rule':c['controller']['time_information'],'registered_commands':x.COMMANDS,'screen_is_registered_held_input':'uThScr' in x.COMMANDS,'uThScr_min':float(df.uThScr.min()),'uThScr_max':float(df.uThScr.max()),'uThScr_range':float(df.uThScr.max()-df.uThScr.min()),'equations':{k:nodes[k]['definition'] for k in ['uThScr','thScrHeat','thScrRh','thScrCold']},'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'meaning':'current internal climate feedback bypasses declared sampled sensor/controller interface; not evidence of future-weather leakage','next':'Expose held screen command from shared controller, use available T/RH/outdoor measurements, specify common active/standby screen settings, then rerun equivalence and lifecycle checks.'}
assert not result['screen_is_registered_held_input']
result['note']='At this fixed-weather operating point the cold-screen minimum masks downstream variation; the contract gap is established by the verified expression dependencies and absent command channel, not by claimed nonzero actuator motion.'
result['internal_feedback_ranges']={k:float(df[k].max()-df[k].min()) for k in ('thScrHeat','thScrRh','rhIn','tAir')}
assert result['source_sha256']==c['model']['files'][path.name]
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
