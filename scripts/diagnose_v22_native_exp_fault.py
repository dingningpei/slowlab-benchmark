#!/usr/bin/env python3
"""Identify first overflowing verified model command at preserved RHS trial state."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.v22_cabauw_weather import CabauwLc1Weather
from slowlab.v22_greenlight_reuse import ReusableGreenLight
from slowlab.v22_native_rhs import CEmitter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import numpy as np
    contract = json.loads((ROOT / 'configs/v22_task_contract_v3.json').read_text())
    weather = CabauwLc1Weather(args.cache, ROOT / 'configs/v22_weather_gapfilled_plan.json')
    engine = ReusableGreenLight(contract, args.source, start=0, mode='active',
                                weather=weather,
                                weather_origin_utc=datetime(2016,12,31,23,tzinfo=timezone.utc),
                                soil_boundary_c=20., array_output=False)
    snapshot = json.loads(args.snapshot.read_text())['failure_context']
    fault = snapshot['native_fault']
    if list(engine.model.states) != snapshot['state_names']:
        raise ValueError('state order differs from failed run')
    input_names = ['Time', *(name for name in engine.model.inputs if name != 'Time')]
    if input_names != snapshot['input_names'] or len(fault['inputs']) != len(input_names):
        raise ValueError('input order differs from failed run')
    commands = engine.model.commands
    CEmitter({'y':len(fault['state']), 'd':len(fault['inputs']),
              'a':len(engine.model.solving_order), 'dy':len(fault['state'])})
    a = np.zeros(len(engine.model.solving_order))
    dy = np.zeros(len(fault['state']))
    scope = {'a':a,'dy':dy,'y':np.asarray(fault['state']),
             'd':np.asarray(fault['inputs'])}
    events = []
    for index, command in enumerate(commands):
        parsed = ast.parse(command)
        if len(parsed.body) != 1:
            raise ValueError('expected one assignment per model command')
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                exec(compile(parsed,'<verified-model-command>','exec'),
                     {'np':np,'__builtins__':{}},scope)
        except FloatingPointError as exc:
            events.append({'index':index,'command':command,
                           'error':type(exc).__name__+': '+str(exc)})
            # Evaluate in the same warning mode as reference to permit later
            # assignments and find whether this is the sole overflow site.
            with np.errstate(over='ignore',invalid='warn',divide='warn'):
                exec(compile(parsed,'<verified-model-command>','exec'),
                     {'np':np,'__builtins__':{}},scope)
    report = {'status':'identified' if events else 'no_python_overflow_reproduced',
              'scope':'exact saved LSODA trial state and inputs; model definitions hash-verified',
              'unit':snapshot['unit'],'time':fault['time'],
              'overflow_events':events,
              'all_derivatives_finite':bool(np.isfinite(dy).all()),
              'max_abs_derivative_difference_vs_saved_reference':float(np.max(np.abs(
                  dy-np.asarray(snapshot['reference_rhs']['derivative']))))}
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'overflow_commands':len(events),
                      'max_abs_derivative_difference':report['max_abs_derivative_difference_vs_saved_reference']}))


if __name__=='__main__':
    main()
