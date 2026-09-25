#!/usr/bin/env python3
"""Select AGC holdout days from input fields without inspecting climate outcomes."""
from __future__ import annotations
import argparse, json
from datetime import date, timedelta
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.audit_agc2019_replay_coverage import (calibration_rows, classify, digest, longest_consecutive_run)

def audit(source: Path, manifest_path: Path) -> dict:
    manifest=json.loads(manifest_path.read_text()); start=date.fromisoformat(manifest["holdout_start"]); stop=date.fromisoformat(manifest["root_observation_end_exclusive_source_clock"][:10])+timedelta(days=1)
    weather_path=source/"Weather"/"Weather.csv"
    if digest(weather_path)!=manifest["weather_sha256"]: raise ValueError("weather hash differs from frozen manifest")
    weather=calibration_rows(weather_path,start,stop); days=[start+timedelta(days=i) for i in range((stop-start).days)]
    details={};summary={}
    for compartment,entry in manifest["compartments"].items():
        path=source/compartment/"GreenhouseClimate.csv"
        if digest(path)!=entry["source_sha256"]["GreenhouseClimate.csv"]: raise ValueError(f"climate hash differs: {compartment}")
        climate=calibration_rows(path,start,stop); details[compartment]={}
        for day in days:
            nxt=day+timedelta(days=1); wr=list(weather.get(day,[])); cr=list(climate.get(day,[]))
            if weather.get(nxt): wr.append(weather[nxt][0])
            if climate.get(nxt): cr.append(climate[nxt][0])
            details[compartment][day.isoformat()]=classify(wr,cr,inspect_observations=False)
        eligible=[day for day,item in details[compartment].items() if item["input_eligible"]]
        summary[compartment]={"period_days":len(days),"input_eligible_days":len(eligible),"input_eligible_dates":eligible,"longest_consecutive_input_eligible_days":longest_consecutive_run(eligible)}
    return {"status":"holdout_input_inventory_outcomes_uninspected","holdout_start":manifest["holdout_start"],"period_start":start.isoformat(),"period_stop_exclusive":stop.isoformat(),"outcome_fields_inspected":False,"selection_rule":"input completeness only; Tair, Rhair, and CO2air are never accessed","summary":summary,"daily":details,"manifest_sha256":digest(manifest_path)}

def main():
    p=argparse.ArgumentParser();p.add_argument("--source",required=True,type=Path);p.add_argument("--manifest",required=True,type=Path);p.add_argument("--out",required=True,type=Path);a=p.parse_args();r=audit(a.source,a.manifest);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n");print(json.dumps(r["summary"],indent=2,sort_keys=True))
if __name__=="__main__":main()
