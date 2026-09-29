#!/usr/bin/env python3
"""Run one frozen parameter candidate on continuous AGC calibration sequences."""
from __future__ import annotations
import argparse, contextlib, csv
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timedelta
from functools import partial
import json, math, os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.agc.audit_agc2019_climate_replay import error_metrics, interpolate  # noqa: E402
from slowlab.greenlight_adapter import (assert_greenlight_solution_complete,
    greenlight_boundary_temperature_override, greenlight_initial_climate_override,
    greenlight_initial_pipe_override)  # noqa: E402


def first_numeric_row(path: Path) -> dict[str,str]:
    with path.open(newline="",encoding="utf-8-sig") as f:
        r=csv.DictReader(f);next(r);next(r);return next(r)


def observed_start(path: Path, stamp: str) -> dict[str,float]:
    with path.open(newline="",encoding="utf-8-sig") as f:
        row=next((x for x in csv.DictReader(f) if x["timestamp"]==stamp),None)
    if row is None: raise ValueError(f"missing observed initial state {stamp}")
    return {k:float(row[k]) for k in ("air_temperature_c","relative_humidity_pct","co2_ppm")}


def sequence_metrics(simulation: Path, observations: Path, start_day: date,
                     day_count: int) -> dict:
    with simulation.open(newline="",encoding="utf-8-sig") as f:
        r=csv.DictReader(f);next(r);next(r);sim=list(r)
    start=datetime.combine(start_day,datetime.min.time()); end=start+timedelta(days=day_count)
    with observations.open(newline="",encoding="utf-8-sig") as f:
        obs=[x for x in csv.DictReader(f) if start <= datetime.fromisoformat(x["timestamp"]) < end]
    ot=[(datetime.fromisoformat(x["timestamp"])-start).total_seconds() for x in obs]
    if not sim:
        raise ValueError("empty GreenLight trajectory")
    times=[float(x["Time"]) for x in sim]
    expected_end=day_count*86400
    if (abs(times[0]) > 1e-6 or times[-1] < expected_end-3600-1e-6
            or any(b <= a or b-a > 3600+1e-6 for a,b in zip(times,times[1:]))):
        raise ValueError("GreenLight output does not continuously cover sequence")
    result={}; violations={}
    for model_name,obs_name in (("tAir","air_temperature_c"),("rhIn","relative_humidity_pct"),("co2InPpm","co2_ppm")):
        model=[float(x[model_name]) for x in sim]
        actual=[interpolate(ot,[float(x[obs_name]) for x in obs],t) for t in times]
        if any(not math.isfinite(v) for v in model+actual):
            raise ValueError(f"non-finite value in {model_name} comparison")
        result[model_name]=error_metrics(model,actual)
        if model_name == "tAir": violations[model_name]=sum(v <= -273.15 for v in model)
        elif model_name == "rhIn": violations[model_name]=sum(v < -1e-3 or v > 100.001 for v in model)
        else: violations[model_name]=sum(v < 0 for v in model)
    return {"samples":len(sim),"metrics":result,"physical_violations":violations}


def aggregate(results: list[dict]) -> dict:
    """Pool errors exactly from sufficient statistics implied by each sequence."""
    if not results:
        raise ValueError("cannot aggregate zero sequences")
    groups={"pooled":results}
    for compartment in sorted({x["compartment"] for x in results}):
        groups[compartment]=[x for x in results if x["compartment"] == compartment]
    output={}
    for name, records in groups.items():
        metrics={}
        for variable in ("tAir","rhIn","co2InPpm"):
            n=sum(x["samples"] for x in records)
            delta_n=sum(max(0,x["samples"]-1) for x in records)
            metrics[variable]={
                "mae":sum(x["metrics"][variable]["mae"]*x["samples"] for x in records)/n,
                "rmse":math.sqrt(sum(x["metrics"][variable]["rmse"]**2*x["samples"] for x in records)/n),
                "bias":sum(x["metrics"][variable]["bias"]*x["samples"] for x in records)/n,
                "hourly_change_rmse":math.sqrt(sum(x["metrics"][variable]["hourly_change_rmse"]**2*max(0,x["samples"]-1) for x in records)/delta_n),
            }
        output[name]={
            "sequence_count":len(records),"samples":sum(x["samples"] for x in records),
            "solver_completion_fraction":sum(bool(x["solver_complete"]) for x in records)/len(records),
            "physical_violations":{v:sum(x["physical_violations"][v] for x in records) for v in ("tAir","rhIn","co2InPpm")},
            "metrics":metrics,
        }
    return output


def run_one(record: dict, candidate: dict, observations_root: Path, definitions: Path,
            led_extension: Path, pipe_extension: Path, out_dir: Path) -> dict:
    from greenlight import GreenLight
    start_day=date.fromisoformat(record["start_day"]); days=int(record["day_count"])
    if start_day+timedelta(days=days)>date.fromisoformat(candidate["holdout_start"]):
        raise ValueError("sequence crosses holdout boundary")
    obs_path=observations_root/record["compartment"]/("climate_observations.csv")
    observed=observed_start(obs_path,start_day.isoformat()+"T00:00:00")
    control=Path(record["controls"]["path"]); first=first_numeric_row(control)
    weather=Path(record["weather"]["path"]); first_weather=first_numeric_row(weather)
    initials=greenlight_initial_climate_override(observed["air_temperature_c"],observed["relative_humidity_pct"],observed["co2_ppm"])
    initials.update(greenlight_boundary_temperature_override(
        observed["air_temperature_c"], float(first_weather["tOut"])))
    initials.update(greenlight_initial_pipe_override(observed["air_temperature_c"],float(first["pipeLowObserved"]),float(first["pipeGrowObserved"])))
    initials["cLeaf"]={"init":repr(float(candidate["initial_lai"])/float(candidate["sla_m2_per_mg_ch2o"]))}
    params={**candidate["facility_parameters"],**candidate["calibration_parameters"]}
    override={"Parameters":{"AGC calibration candidate":{k:{"type":"const","definition":repr(float(v))} for k,v in params.items()}},"Initial state":initials,"options":{"t_end":repr(days*86400),"solver":candidate["solver"]}}
    safe=record["sequence_id"].replace(":","_"); output=out_dir/f"{safe}.csv";output.parent.mkdir(parents=True,exist_ok=True)
    sim=GreenLight(base_path=str(definitions.resolve()),input_prompt=[str((definitions/'main_katzin_2021.json').resolve()),str((definitions/'lamp_hps_katzin_2020.json').resolve()),str(led_extension.resolve()),str(pipe_extension.resolve()),str(weather.resolve()),str(Path(record["controls"]["path"]).resolve()),override],output_path=str(output))
    with open(os.devnull,"w") as sink,contextlib.redirect_stdout(sink):sim.run()
    assert_greenlight_solution_complete(sim.states_sol,days*86400)
    metrics=sequence_metrics(output,obs_path,start_day,days)
    return {"sequence_id":record["sequence_id"],"compartment":record["compartment"],"start_day":record["start_day"],"day_count":days,"solver_complete":True,**metrics}


def main():
    p=argparse.ArgumentParser();p.add_argument("--sequences",required=True,type=Path);p.add_argument("--candidate",required=True,type=Path);p.add_argument("--observations-root",required=True,type=Path);p.add_argument("--definitions",required=True,type=Path);p.add_argument("--led-extension",required=True,type=Path);p.add_argument("--pipe-extension",required=True,type=Path);p.add_argument("--out-dir",required=True,type=Path);p.add_argument("--out",required=True,type=Path);p.add_argument("--limit",type=int);p.add_argument("--workers",type=int,default=1);a=p.parse_args()
    seq=json.loads(a.sequences.read_text());candidate=json.loads(a.candidate.read_text());records=seq["sequences"][:a.limit]
    if a.workers < 1: raise ValueError("workers must be positive")
    task=partial(run_one,candidate=candidate,observations_root=a.observations_root,
                 definitions=a.definitions,led_extension=a.led_extension,
                 pipe_extension=a.pipe_extension,out_dir=a.out_dir)
    if a.workers == 1: results=[task(x) for x in records]
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool: results=list(pool.map(task,records))
    payload={"status":"calibration_only_no_holdout","candidate":candidate["candidate_id"],"sequences":results,"sequence_count":len(results),"aggregate":aggregate(results)};a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps({"sequence_count":len(results),"pooled":payload["aggregate"]["pooled"]},indent=2))


if __name__=="__main__":main()
