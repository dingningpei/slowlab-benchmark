#!/usr/bin/env python3
"""Replay one predictor-data run with a cap on RHS evaluations per control step.

    python3 scripts/diagnose_solver_stall.py SPEC.json --cap 200000 --out diag.json

Every compartment solver counts its right-hand-side evaluations per 300 s step
(counting does not change results). Each simulated day the busiest step so far
is logged per compartment. If a step exceeds the cap, the run stops and the
diagnostic records the compartment, its policy and parameters, the time, and the
states with the largest values and the largest relative derivatives.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.cached_solver import SolverStall  # noqa: E402
from slowlab.predictor_data import collect  # noqa: E402
from slowlab.private_runs import prepare  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=Path)
    parser.add_argument('--cap', type=int, default=200_000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    p = prepare(spec)
    daily, state = [], {'x': None}
    began = time.monotonic()

    def hook(x):
        state['x'] = x
        row = {'day': x.clock / 86400, 'wall_s': round(time.monotonic() - began, 1)}
        for unit, life in x._lifecycles.items():
            for mode, engine in life.engines.items():
                solver = getattr(engine, '_cached_solver', None)
                if solver is None:
                    continue
                solver.max_rhs_calls_per_step = args.cap
                solver.stall_label = f'unit {unit} {mode}'
                row[f'{unit}/{mode}'] = getattr(solver, 'max_rhs_calls_seen', 0)
        daily.append(row)
        args.out.with_suffix('.daily.json').write_text(json.dumps(daily))
    p['executor_kwargs']['progress_hook'] = hook
    record = {'spec': spec, 'cap': args.cap}
    try:
        collect(p['contract'], layout=spec['layout'], layout_seed=int(spec['layout_seed']), year=p['year'],
                weather=p['weather'], source=p['source'], site_unit_parameters=p['unit_parameters'],
                soil_boundary_c=p['soil_boundary_c'], sensor_noise=p['sensor_noise'], executor_kwargs=p['executor_kwargs'])
        record['status'] = 'completed_without_stall'
    except SolverStall as stall:
        x = state['x']
        unit = stall.label.split()[1] if stall.label else None
        y, dy = stall.y, stall.dy
        rel = np.abs(dy) / np.maximum(np.abs(y), 1e-9)
        order_rel = np.argsort(-rel)[:12]
        order_abs = np.argsort(-np.abs(y))[:8]
        record.update(status='stall', label=stall.label, calls=stall.calls, step=[stall.t0, stall.t1], t=stall.t,
                      day=stall.t / 86400, policy=(x._policies.get(unit).as_dict() if x and unit in x._policies else None),
                      unit_parameters=(x.unit_parameters.get(unit) if x else None),
                      largest_relative_derivative=[{'state': stall.states[i], 'y': float(y[i]), 'dy': float(dy[i])}
                                                   for i in order_rel],
                      largest_states=[{'state': stall.states[i], 'y': float(y[i])} for i in order_abs])
    except Exception as error:
        record.update(status='other_error', error=f'{type(error).__name__}: {error}')
    record['wall_seconds'] = round(time.monotonic() - began, 1)
    record['busiest_step_by_solver'] = daily[-1] if daily else None
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: record.get(k) for k in ('status', 'label', 'day', 'calls', 'wall_seconds')}))


if __name__ == '__main__':
    main()
