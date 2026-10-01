#!/usr/bin/env python3
"""Replay one predictor-data run and record each compartment's daily minimum of chosen crop states.

    python3 scripts/monitor_state_minima.py SPEC.json --states cBuf --cap 200000 --out minima.json

Observation only: the solver records the minimum over its step points and the
results of the run are unchanged. A step that exceeds the RHS-evaluation cap
stops the run (recorded as a stall). The output lists every crop with its
planting day and the minimum of each state over its active days.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.cached_solver import SolverStall  # noqa: E402
from slowlab.predictor_data import collect  # noqa: E402
from slowlab.private_runs import prepare  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('spec', type=Path)
    parser.add_argument('--states', default='cBuf')
    parser.add_argument('--cap', type=int, default=200_000)
    parser.add_argument('--out', type=Path, default=None, help='default: the spec\'s "out"')
    args = parser.parse_args()
    states = args.states.split(',')
    spec = json.loads(args.spec.read_text())
    args.out = args.out or Path(spec['out'])
    p = prepare(spec)
    daily = []
    began = time.monotonic()

    def hook(x):
        row = {'day': x.clock / 86400}
        for unit, life in x._lifecycles.items():
            solver = getattr(life.engines['active'], '_cached_solver', None)
            if solver is None:
                continue
            solver.max_rhs_calls_per_step = args.cap
            solver.stall_label = f'unit {unit} active'
            solver.track_state_minima = states
            minima = getattr(solver, 'state_minima', {})
            row[unit] = {'mode': life.mode, **minima}  # only active-crop solves populate the minima
            solver.state_minima = {}
        daily.append(row)
    p['executor_kwargs']['progress_hook'] = hook
    record = {'spec': spec, 'states': states, 'cap': args.cap}
    crops = None
    try:
        result = collect(p['contract'], layout=spec['layout'], layout_seed=int(spec['layout_seed']), year=p['year'],
                         weather=p['weather'], source=p['source'], site_unit_parameters=p['unit_parameters'],
                         soil_boundary_c=p['soil_boundary_c'], sensor_noise=p['sensor_noise'],
                         executor_kwargs=p['executor_kwargs'])
        record['status'] = 'completed'
        crops = [{k: c[k] for k in ('unit', 'run_index', 'planting_day', 'policy', 'final_reason')} for c in result['crops']]
    except SolverStall as stall:
        record.update(status='stall', label=stall.label, day=stall.t / 86400,
                      stalled_states={n: float(stall.y[stall.states.index(n)]) for n in states})
    record['wall_seconds'] = round(time.monotonic() - began, 1)
    if crops is not None:
        crop_days = p['contract']['budget']['crop_days']
        for c in crops:
            lo, hi = c['planting_day'], c['planting_day'] + crop_days
            # The hook reports the minimum over the day that just ended.
            days = [r for r in daily if lo < r['day'] <= hi + 1e-9]
            c['state_minima'] = {n: min((r[str(c['unit'])][n] for r in days if n in r[str(c['unit'])]), default=None)
                                 for n in states}
        record['crops'] = crops
    record['daily'] = daily
    args.out.write_text(json.dumps(record) + '\n')
    print(json.dumps({'status': record['status'], 'wall_seconds': record['wall_seconds'],
                      'crop_minima': [c['state_minima'] for c in crops] if crops else record.get('stalled_states')}))


if __name__ == '__main__':
    main()
