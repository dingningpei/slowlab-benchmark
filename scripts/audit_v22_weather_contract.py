"""Read-only coverage audit of existing local audit manifests; no downloads."""
import hashlib,json
from datetime import datetime
from pathlib import Path
r=Path(__file__).resolve().parents[1]
def read(name):return json.loads((r/'configs'/name).read_text())
c=read('v22_task_contract_v2.json');original=read('greenlight_original_validation_preflight_v0.json');agc23=read('agc2023_development_summary_v0.json');agc24=read('agc2024_input_feasibility_v0.json')
rows=[]
for name,start,end in [('original GreenLight',original['declared_experiment_horizon']['start'],original['declared_experiment_horizon']['end']),('AGC2023',agc23['climate']['start'],agc23['climate']['end']),('AGC2024 weather',agc24['tables']['weather']['timestamp_start'],agc24['tables']['weather']['timestamp_end'])]:
 rows.append({'source':name,'start':start,'end':end,'span_days':(datetime.fromisoformat(end)-datetime.fromisoformat(start)).total_seconds()/86400})
result={'status':'requires_weather_protocol','campaign_days':c['budget']['campaign_days'],'contract_has_weather_section':'weather' in c,'audited_manifest_coverage':rows,'scope':'Existing audit metadata only; not an exhaustive dataset search or verification of raw files currently on disk. No previously reserved outcomes inspected.','unfrozen':['source years/season and start calendar','aligned temperature/RH/solar/wind/sky-temperature/CO2 channels and units','missing data, timezone and downsampling conventions','development/pilot/test partition and independent evaluation weather','joint temporal/weather structure and blinding/trajectory-reuse audit'],'prohibited_shortcuts':['repeat short seasonal records to manufacture a year','use constant smoke weather as annual evidence','resample individual weather channels independently'],'hashes':{n:hashlib.sha256((r/'configs'/n).read_bytes()).hexdigest() for n in ['v22_task_contract_v2.json','greenlight_original_validation_preflight_v0.json','agc2023_development_summary_v0.json','agc2024_input_feasibility_v0.json']}}
(r/'configs/v22_phase1_weather_contract_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
