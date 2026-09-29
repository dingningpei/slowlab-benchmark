#!/usr/bin/env python3
"""Replay exact failed LSODA RHS evaluation with stable native logistic."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from slowlab.v22.cabauw_weather import CabauwLc1Weather
from slowlab.v22.greenlight_reuse import ReusableGreenLight
from slowlab.v22.native_rhs import NativeRHS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source))
    import numpy as np
    contract = json.loads((ROOT / 'configs/v22/task_contract_v3.json').read_text())
    weather = CabauwLc1Weather(args.cache, ROOT / 'configs/v22/weather_gapfilled_plan.json')
    engine = ReusableGreenLight(contract, args.source, start=0, mode='active',
                                weather=weather,
                                weather_origin_utc=datetime(2016,12,31,23,tzinfo=timezone.utc),
                                soil_boundary_c=20.)
    snapshot = json.loads(args.snapshot.read_text())['failure_context']
    fault = snapshot['native_fault']
    if list(engine.model.states) != snapshot['state_names']:
        raise ValueError('state order changed')
    if ['Time', *(name for name in engine.model.inputs if name != 'Time')] != snapshot['input_names']:
        raise ValueError('input order changed')
    y = np.asarray(fault['state'])
    d = np.asarray(fault['inputs'])[None,:]
    native = NativeRHS(engine.model)
    output = native(fault['time'], y, d)
    old_native = np.asarray(fault['derivative'])
    reference = np.asarray(snapshot['reference_rhs']['derivative'])
    max_abs = float(np.max(np.abs(output-reference)))
    max_vs_old = float(np.max(np.abs(output-old_native)))
    report = {'status':'passed_exact_fault_replay' if max_abs < 1e-9 else 'failed_derivative_mismatch',
              'scope':'single saved LSODA trial evaluation, not a full campaign or annual stability pass',
              'unit':snapshot['unit'],'time':fault['time'],
              'old_flags':fault['flags'],'old_native_derivative_all_finite':fault['derivative_all_finite'],
              'new_native_all_finite':bool(np.isfinite(output).all()),
              'max_abs_difference_vs_reference':max_abs,
              'max_abs_difference_vs_old_native':max_vs_old,
              'native_source_sha256':native.source_sha256,
              'native_compiler':native.compiler_version}
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('status','max_abs_difference_vs_reference','max_abs_difference_vs_old_native')}))
    if report['status'] != 'passed_exact_fault_replay':
        raise SystemExit(1)


if __name__=='__main__':
    main()
