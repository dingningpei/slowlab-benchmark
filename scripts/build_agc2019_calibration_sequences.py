#!/usr/bin/env python3
"""Concatenate frozen AGC calibration days into state-continuous inputs."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from slowlab.greenlight_sequence import concatenate_greenlight_inputs  # noqa: E402


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(matrix_path: Path, out: Path) -> dict:
    matrix=json.loads(matrix_path.read_text()); records=[]
    for sequence in matrix["continuous_sequences"]:
        safe=sequence["sequence_id"].replace(":","_")
        weather=out/f"{safe}_weather.csv"; controls=out/f"{safe}_controls.csv"
        days=sequence["days"]
        weather_rows=concatenate_greenlight_inputs([Path(day["weather"]["path"]) for day in days],weather)
        control_rows=concatenate_greenlight_inputs([Path(day["driver"]["path"]) for day in days],controls)
        records.append({"sequence_id":sequence["sequence_id"],"compartment":sequence["compartment"],
                        "start_day":days[0]["day"],"day_count":len(days),
                        "weather":{"path":str(weather),"sha256":digest(weather),"rows":weather_rows},
                        "controls":{"path":str(controls),"sha256":digest(controls),"rows":control_rows}})
    return {"status":"continuous_calibration_inputs_no_outcomes_read","sequences":records,
            "sequence_count":len(records),"source_matrix_sha256":digest(matrix_path),
            "holdout_start_not_read":matrix["holdout_start_not_read"]}


def main():
    p=argparse.ArgumentParser();p.add_argument("--matrix",required=True,type=Path);p.add_argument("--out-dir",required=True,type=Path);p.add_argument("--out",required=True,type=Path);a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);r=build(a.matrix,a.out_dir);a.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n");print(json.dumps({"sequence_count":r["sequence_count"]},indent=2))


if __name__=="__main__":main()
