#!/usr/bin/env python3
"""Audit AGC 2024 replay inputs without accessing indoor climate outcomes."""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, pathlib, zipfile
from collections import defaultdict
from datetime import datetime

TEAM_FILES={"agrifusion","ideas","mugrow","reference","tomatonuts","trigger"}
OUTCOME_TAILS=("air_temperature","relative_humidity","co2_concentration","humidity_deficit","leaf_temperature")
REQUIRED_TEAM={
 "pipe_temperature":"compartment/heating_lower_circuit/pipe_temperature",
 "screen_energy":"compartment/screen_energy/screen_position",
 "screen_blackout":"compartment/screen_blackout/screen_position",
 "window_lee":"compartment/window_position_lee_side",
 "window_wind":"compartment/window_position_wind_side",
 "lamp_state":"compartment/lamps_activation_percentage",
 "co2_state":"compartment/co2_actuation_state",
 "co2_minutes":"compartment/co2_dosage_minutes_cumulative",
 "plant_density":"dwarf_tomato/plant_density",
}
REQUIRED_WEATHER={
 "outside_temperature":"weather/air_temperature.outside",
 "outside_rh":"weather/relative_humidity.outside",
 "solar_radiation":"weather/radiation_global",
 "wind_speed":"weather/wind_speed",
 "wind_direction":"weather/wind_direction.registration",
 "rain_state":"weather/rain_state",
 "sky_longwave_proxy":"weather/heat_emission",
}

def digest(path:pathlib.Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def forbidden(name:str)->bool:
 if not name.startswith('compartment/'): return False
 tail=name.split('/',1)[1]
 return any(tail==x or tail.startswith(x+'.') for x in OUTCOME_TAILS)

def observation_only(name:str)->bool:
 return name in {"compartment/par","compartment/mass.plant"} or "fluorescence" in name or "vivent" in name

def derived_only(name:str)->bool:
 return name.startswith('economics/') or name.startswith('energy/')

def parse_time(value:str):
 v=value.strip()
 if not v:return None
 if v.endswith('Z'):v=v[:-1]+'+00:00'
 try:return datetime.fromisoformat(v)
 except ValueError:
  for fmt in ('%Y-%m-%d %H:%M:%S','%Y-%m-%d %H:%M:%S%z'):
   try:return datetime.strptime(v,fmt)
   except ValueError:pass
 raise ValueError(f'unsupported timestamp {value!r}')

def number(value:str):
 v=value.strip()
 if not v:return None
 try:
  x=float(v)
  return x if math.isfinite(x) else None
 except ValueError:return None

def audit_table(z,name,required):
 with z.open(name) as raw:
  text=io.TextIOWrapper(raw,encoding='utf-8-sig',newline='')
  reader=csv.reader(text); header=next(reader)
  if len(set(header))!=len(header):raise ValueError(f'duplicate columns in {name}')
  forbidden_idx={i for i,c in enumerate(header) if forbidden(c)}
  observation_idx={i for i,c in enumerate(header) if observation_only(c)}
  derived_idx={i for i,c in enumerate(header) if derived_only(c)}
  allowed_idx=[i for i in range(len(header)) if i not in forbidden_idx|observation_idx|derived_idx]
  essential_cols=[c for c in required.values() if c!='dwarf_tomato/plant_density']
  essential_idx=[header.index(c) for c in essential_cols if c in header]
  stats={header[i]:{'nonmissing':0,'numeric':0,'min':None,'max':None,'changes':0} for i in allowed_idx if header[i]!='time'}
  prior={}; timestamps=[]; rows=0; width_errors=0; incomplete_input_times=[]; complete_inputs=0
  for row in reader:
   rows+=1
   if len(row)!=len(header):width_errors+=1;continue
   if essential_idx:
    if len(essential_idx)==len(essential_cols) and all(row[i].strip() for i in essential_idx):complete_inputs+=1
    else:incomplete_input_times.append(row[header.index('time')])
   # Only these indices are accessed. Forbidden indoor outcome indices are never indexed.
   for i in allowed_idx:
    col=header[i]; value=row[i]
    if col=='time':
     t=parse_time(value)
     if t is not None:timestamps.append(t)
     continue
    s=stats[col]
    if value.strip():s['nonmissing']+=1
    x=number(value)
    if x is not None:
     s['numeric']+=1;s['min']=x if s['min'] is None else min(s['min'],x);s['max']=x if s['max'] is None else max(s['max'],x)
     if col in prior and prior[col]!=x:s['changes']+=1
     prior[col]=x
  for s in stats.values():
   s['coverage']=s['nonmissing']/rows if rows else 0
  deltas=[]
  for a,b in zip(timestamps,timestamps[1:]):deltas.append((b-a).total_seconds())
  positive=[x for x in deltas if x>0]
  cadence=min(positive,key=lambda x:abs(x-300)) if positive else None
  required_report={}
  for label,col in required.items():
   required_report[label]={"column":col,"present":col in header,"stats":stats.get(col)}
  return {
   'rows':rows,'columns':len(header),'row_width_errors':width_errors,
   'timestamp_count':len(timestamps),'timestamp_start':min(timestamps).isoformat() if timestamps else None,
   'timestamp_end':max(timestamps).isoformat() if timestamps else None,
   'nonincreasing_timestamp_steps':sum(x<=0 for x in deltas),
   'steps_not_300_seconds':sum(x!=300 for x in deltas),
   'forbidden_outcome_columns_named_only':[header[i] for i in sorted(forbidden_idx)],
   'forbidden_outcome_values_accessed':False,
   'observation_only_columns_named_only':[header[i] for i in sorted(observation_idx)],
   'observation_only_values_accessed':False,
   'derived_resource_columns_named_only':[header[i] for i in sorted(derived_idx)],
   'derived_resource_values_accessed':False,
   'essential_input_complete_rows':complete_inputs,
   'essential_input_incomplete_timestamps':incomplete_input_times,
   'required_inputs':required_report,
   'allowed_field_stats':stats,
  }

def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=pathlib.Path,required=True);p.add_argument('--protocol',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args()
 protocol=json.loads(a.protocol.read_text())
 if protocol.get('status')!='frozen before archive download or timeseries value access':raise ValueError('unexpected protocol status')
 result={'audit_id':'agc2024-dwarf-tomato-input-feasibility-v0','protocol_id':protocol['protocol_id'],'archive_sha256':digest(a.archive),'protocol_sha256':digest(a.protocol),'direct_outcome_values_accessed':False,'selection_made':False,'tables':{}}
 with zipfile.ZipFile(a.archive) as z:
  names=[n for n in z.namelist() if '/timeseries/' in n and n.endswith('.csv') and not n.startswith('__MACOSX/')]
  for name in sorted(names):
   stem=pathlib.PurePosixPath(name).stem
   req=REQUIRED_WEATHER if stem=='weather' else (REQUIRED_TEAM if stem in TEAM_FILES else {})
   result['tables'][stem]=audit_table(z,name,req)
 result['audit_disclosure']="An initial implementation summarized resource-field ranges because the frozen protocol allowed resource inputs. The accompanying paper was then found to define heating energy from pipe temperature minus indoor air temperature. No identity or period had been selected. The final auditor excludes all energy/economics values, records their names only, and does not use them for eligibility."
 result['known_unmeasured_or_unpublished']=[
  'heating-water mass flow and separate supply/return temperatures per circuit',
  'direct natural-ventilation air mass flow or a facility-specific discharge calibration',
  'fogging or humidification realised state and water rate',
  'CO2 instantaneous mass flow; public channels expose actuation state, cumulative dosing minutes and aggregate dosage',
  'screen heat/moisture flux and material exchange coefficients',
  'continuous canopy transpiration or LAI state',
  'compartment geometry, boundary temperatures, thermal masses and equipment calibration parameters in the archive'
 ]
 result['source_grounded_facility_evidence']={
  'paper':'Maree et al. 2025, Sensors 25:4321, doi:10.3390/s25144321',
  'floor_plan':'10 m x 9.6 m (96 m2), with figure A1 cross-section and table layout',
  'boundary_topology':'each compartment has one exterior side wall and one side wall shared with another compartment',
  'heating':'single floor pipe-rail circuit; 120 W m-2 peak capacity',
  'ventilation':'continuous roof ventilation; opening area 0.3 m2 per m2 floor; 0.40 x 0.45 mm anti-insect net',
  'screens':['LUXOUS 1547 D FR energy screen','OBSCURA 9950 FR W blackout screen'],
  'lighting':'8 Fluence VYPR 4i B9F WB dimmable LED fixtures; 20-200 micromol m-2 s-1; assumed efficacy 3.2 micromol J-1',
  'fogging_capacity_g_m2_h':220,
  'co2_capacity_g_m2_h':7.5,
  'crop':'Pick-&-Joy Red Cherry dwarf tomato; 13 cm pots on three 1.62 x 6.12 m, 0.8 m-high tables; initial density 56 plants m-2',
  'development_asset_not_in_public_archive':'Teams had access to Kaspro parameterized for these compartments and a dwarf-tomato crop model; the public timeseries archive does not include either model.'
 }
 result['derived_field_semantics']={
  'energy/energy_use.heating':'not an independent heat-meter measurement; paper estimates 5-minute heat from 2*max(0, pipe temperature - indoor air temperature) W m-2',
  'energy/electricity_use.lighting':'calculated from lamp activation, 200 micromol m-2 s-1 maximum and assumed 3.2 micromol J-1 efficacy',
  'energy/co2_dosage':'calculated from positive changes in cumulative dosing minutes times 0.125 g m-2 min-1'
 }
 result['provisional_decision']='B for prospective observed-action climate replay after source-grounded fogging/controller and crop-state reconstruction with frozen uncertainty bounds; C for direct counterfactual actuator validation because the public archive alone does not identify ventilation, screen, crop-transpiration, or heating-flux responses.'
 a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'direct_outcome_values_accessed':False,'archive_sha256':result['archive_sha256'],'tables':{k:{'rows':v['rows'],'start':v['timestamp_start'],'end':v['timestamp_end'],'steps_not_300_seconds':v['steps_not_300_seconds'],'essential_input_complete_rows':v['essential_input_complete_rows'],'required':{x:{'present':q['present'],'coverage':q['stats']['coverage'] if q['stats'] else None} for x,q in v['required_inputs'].items()}} for k,v in result['tables'].items()}},indent=2))
if __name__=='__main__':main()
