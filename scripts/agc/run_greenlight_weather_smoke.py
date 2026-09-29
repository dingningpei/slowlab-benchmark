#!/usr/bin/env python3
"""Reproduce a weather-only GreenLight interface run, never a replay score."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowlab.greenlight_adapter import assert_greenlight_solution_complete  # noqa: E402


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--weather", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    from greenlight import GreenLight  # Optional external BSD-licensed dependency.

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sim = GreenLight(base_path=str(args.model.parent),
                     input_prompt=[str(args.model), str(args.weather),
                                   {"options": {"t_end": "86400"}}],
                     output_path=str(args.out))
    # GreenLight emits one progress update per solver step. Its simulation log
    # remains on disk, and a compact audit is printed here.
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        sim.run()
    result = {
        "status": "weather_only_default_facility_smoke",
        "model_sha256": sha256(args.model),
        "weather_sha256": sha256(args.weather),
        "output_sha256": sha256(args.out),
        "solver_success": bool(sim.states_sol.success),
        "end_seconds": float(sim.states_sol.t[-1]),
        "warning": "Uses GreenLight's default facility and controls; no AGC action replay or validation error.",
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    assert_greenlight_solution_complete(sim.states_sol, 86400)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
