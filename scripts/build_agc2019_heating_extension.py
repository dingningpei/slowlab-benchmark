#!/usr/bin/env python3
"""Build bounded pipe-temperature tracking for AGC trajectory replay."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def v(unit, kind, definition, description):
    return {"unit": unit, "type": kind, "definition": definition, "description": description}


def build_extension():
    return {
        "Info": {
            "About": "Calibration-only bounded tracking of observed positive AGC pipe temperatures",
            "Status": "diagnostic aggregate actuator model; tracking times require calibration",
            "Semantics": "positive source values are temperature targets; zero is an off/status flag; no source value is treated as heat power",
        },
        "AGC pipe tracking heat inputs": {
            "hBoilPipe": v("W m**-2", "aux", "pipeLowActive * min(railPipePeak, max(0, capPipe/railTrackTau * (pipeLowTarget-tPipe)))", "Capacity-bounded heat input used to track positive observed rail-pipe temperature"),
            "hBoilGroPipe": v("W m**-2", "aux", "pipeGrowActive * min(growPipePeak, max(0, capGroPipe/growTrackTau * (pipeGrowTarget-tGroPipe)))", "Capacity-bounded heat input used to track positive observed grow-pipe temperature"),
        },
        "AGC pipe tracking parameters": {
            "railPipePeak": v("W m**-2", "const", "180", "Published AGC rail-pipe peak capacity"),
            "growPipePeak": v("W m**-2", "const", "30", "Published AGC grow-pipe peak capacity"),
            "railTrackTau": v("s", "const", "300", "Diagnostic rail-pipe tracking time; calibration parameter"),
            "growTrackTau": v("s", "const", "300", "Diagnostic grow-pipe tracking time; calibration parameter"),
        },
    }


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",required=True,type=Path); args=parser.parse_args(); args.out.write_text(json.dumps(build_extension(),indent=2)+"\n")


if __name__ == "__main__": main()
