#!/usr/bin/env python3
"""Add bounded heat-loss feed-forward to AGC observed-pipe replay."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def build(base: Path, out: Path) -> dict:
    data = json.loads(base.read_text())
    data["Info"]["Status"] = "trajectory-replay boundary condition with bounded loss-compensated state continuity; not a counterfactual actuator model"
    data["Info"]["State continuity correction"] = "During active replay, internal pipe-state input includes the implied heat loss plus proportional tracking; this prevents an artificially cold state at transition to off-coded intervals."
    tracking = data["Internal pipe state tracking"]
    tracking["hBoilPipe"].update({
        "definition": "pipeLowActive * min(railPipePeak, max(0, hObservedRailPipeDemand + capPipe/railTrackTau * (pipeLowObserved-tPipe)))",
        "description": "Capacity-bounded heat-loss feed-forward plus proportional tracking, used only for internal-state continuity into off-coded intervals",
    })
    tracking["hBoilGroPipe"].update({
        "definition": "pipeGrowActive * min(growPipePeak, max(0, hObservedGrowPipeDemand + capGroPipe/growTrackTau * (pipeGrowObserved-tGroPipe)))",
        "description": "Capacity-bounded heat-loss feed-forward plus proportional tracking, used only for internal-state continuity into off-coded intervals",
    })
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n")
    return data


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--base",required=True,type=Path);p.add_argument("--out",required=True,type=Path);a=p.parse_args();build(a.base,a.out)

if __name__ == "__main__": main()
