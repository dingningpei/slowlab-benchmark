#!/usr/bin/env python3
"""Prepare holdout replay inputs selected without climate outcomes."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from slowlab.agc_greenlight_weather import WeatherAssumptions
from scripts.prepare_agc2019_calibration_matrix import prepare

def main():
    p=argparse.ArgumentParser();p.add_argument("--source",required=True,type=Path);p.add_argument("--manifest",required=True,type=Path);p.add_argument("--coverage",required=True,type=Path);p.add_argument("--out-dir",required=True,type=Path);p.add_argument("--matrix",required=True,type=Path);a=p.parse_args()
    cov=json.loads(a.coverage.read_text())
    if cov.get("outcome_fields_inspected") is not False: raise ValueError("holdout coverage must certify outcomes uninspected")
    r=prepare(a.source,a.manifest,a.coverage,a.out_dir,WeatherAssumptions(410,10,0,0),period="holdout");r["status"]="holdout_inputs_only_outcomes_uninspected";r["outcome_fields_inspected"]=False;a.matrix.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n");print(json.dumps({"identity_count":r["identity_count"],"sequence_count":r["sequence_count"]},indent=2))
if __name__=="__main__":main()
