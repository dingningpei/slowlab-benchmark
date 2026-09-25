#!/usr/bin/env python3
"""Execute the frozen one-shot AGC v8 temporal holdout validation."""
from __future__ import annotations
import argparse,contextlib,csv,hashlib,json,math,os,sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date,datetime,timedelta
from functools import partial
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.audit_agc2019_climate_replay import error_metrics,interpolate
from scripts.run_agc2019_calibration_sequences import aggregate,first_numeric_row,observed_start,sequence_metrics
from slowlab.agc_temperature_residual import TemperatureResidualModel,feature_vector
from slowlab.greenlight_adapter import assert_greenlight_solution_complete,greenlight_boundary_temperature_override,greenlight_initial_climate_override,greenlight_initial_pipe_override

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def corrected_tair(path,obs_path,start,days,model):
 with Path(path).open(newline='',encoding='utf-8-sig') as f:r=csv.DictReader(f);next(r);next(r);sim=list(r)
 end=start+timedelta(days=days)
 with Path(obs_path).open(newline='',encoding='utf-8-sig') as f:obs=[x for x in csv.DictReader(f) if start<=datetime.fromisoformat(x['timestamp'])<end]
 ot=[(datetime.fromisoformat(x['timestamp'])-start).total_seconds() for x in obs];ov=[float(x['air_temperature_c']) for x in obs]
 pred=[];actual=[]
 for row in sim:
  t=float(row['Time']);stamp=start+timedelta(seconds=t);base=float(row['tAir']);pred.append(base+model.predict(row,stamp));actual.append(interpolate(ot,ov,t))
 return error_metrics(pred,actual),sum(x<=-273.15 for x in pred)
def run_one(record,candidate,model,observations_root,definitions,led_extension,pipe_extension,out_dir,period_end):
 from greenlight import GreenLight
 start_day=date.fromisoformat(record['start_day']);days=int(record['day_count'])
 if start_day<date(2020,4,1) or start_day+timedelta(days=days)>date.fromisoformat(period_end):raise ValueError('sequence outside frozen holdout')
 obs_path=observations_root/record['compartment']/'climate_observations.csv';stamp=start_day.isoformat()+'T00:00:00';observed=observed_start(obs_path,stamp)
 control=Path(record['controls']['path']);first=first_numeric_row(control);weather=Path(record['weather']['path']);first_weather=first_numeric_row(weather)
 initials=greenlight_initial_climate_override(observed['air_temperature_c'],observed['relative_humidity_pct'],observed['co2_ppm']);initials.update(greenlight_boundary_temperature_override(observed['air_temperature_c'],float(first_weather['tOut'])));initials.update(greenlight_initial_pipe_override(observed['air_temperature_c'],float(first['pipeLowObserved']),float(first['pipeGrowObserved'])));initials['cLeaf']={'init':repr(float(candidate['initial_lai'])/float(candidate['sla_m2_per_mg_ch2o']))}
 params={**candidate['facility_parameters'],**candidate['calibration_parameters']};override={'Parameters':{'AGC calibration candidate':{k:{'type':'const','definition':repr(float(v))} for k,v in params.items()}},'Initial state':initials,'options':{'t_end':repr(days*86400),'solver':candidate['solver']}}
 safe=record['sequence_id'].replace(':','_');output=out_dir/f'{safe}.csv';output.parent.mkdir(parents=True,exist_ok=True)
 sim=GreenLight(base_path=str(definitions.resolve()),input_prompt=[str((definitions/'main_katzin_2021.json').resolve()),str((definitions/'lamp_hps_katzin_2020.json').resolve()),str(led_extension.resolve()),str(pipe_extension.resolve()),str(weather.resolve()),str(control.resolve()),override],output_path=str(output))
 with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):sim.run()
 assert_greenlight_solution_complete(sim.states_sol,days*86400);m=sequence_metrics(output,obs_path,start_day,days);m['metrics']['tAir'],m['physical_violations']['tAir']=corrected_tair(output,obs_path,datetime.combine(start_day,datetime.min.time()),days,model)
 return {'sequence_id':record['sequence_id'],'compartment':record['compartment'],'start_day':record['start_day'],'day_count':days,'solver_complete':True,**m}
def main():
 p=argparse.ArgumentParser();
 for name in ('sequences','candidate','model','execution','observations-root','definitions','led-extension','pipe-extension','out-dir','out'):p.add_argument('--'+name,required=True,type=Path)
 p.add_argument('--workers',type=int,default=1);a=p.parse_args();execution=json.loads(a.execution.read_text());seq=json.loads(a.sequences.read_text());candidate_payload=json.loads(a.candidate.read_text());candidate=candidate_payload.get('candidate',candidate_payload);payload=json.loads(a.model.read_text());model=TemperatureResidualModel.from_dict(payload['model'])
 if digest(a.sequences)!=execution['sequence_manifest_sha256'] or digest(a.model)!=execution['temperature_residual_model_sha256']:raise ValueError('frozen artifact hash mismatch')
 identities=[(x['compartment'],x['start_day'],x['day_count']) for x in seq['sequences']]
 if identities!=[tuple(x) for x in execution['sequences']]:raise ValueError('frozen sequence identities mismatch')
 task=partial(run_one,candidate=candidate,model=model,observations_root=a.observations_root,definitions=a.definitions,led_extension=a.led_extension,pipe_extension=a.pipe_extension,out_dir=a.out_dir,period_end=execution['period_end_exclusive'])
 with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(task,seq['sequences']))
 result={'status':'one-shot temporal holdout evaluated','protocol_id':payload['protocol_id'],'sequences':results,'sequence_count':len(results),'aggregate':aggregate(results),'execution_sha256':digest(a.execution),'model_sha256':digest(a.model)};a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps({'sequence_count':len(results),'aggregate':result['aggregate']},indent=2))
if __name__=='__main__':main()
