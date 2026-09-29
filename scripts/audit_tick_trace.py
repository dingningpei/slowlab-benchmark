#!/usr/bin/env python3
"""Audit causal command and observation order in a private short-run tick trace."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def _finite(value) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def audit(trace_path: Path, result_path: Path, contract_path: Path) -> dict:
    result = json.loads(result_path.read_text())
    contract = json.loads(contract_path.read_text())
    assert result['status'] == 'passed_four_unit_runtime_window'
    expected_steps = result['completed_steps']
    units = tuple(str(i) for i in range(contract['facility']['compartments']))
    public_channels = set(contract['observations']['public_channels'])
    controller_channels = set(contract['observations']['controller_channels'])
    command_keys = {'uBoil', 'uRoof', 'uExtCo2', 'uLamp', 'uThScr'}
    resource_keys = {'heat_kwh_m2', 'light_kwh_m2', 'co2_kg_m2',
                     'harvest_kg_m2', 'transpiration_kg_m2', 'days'}
    totals = {unit: {key: 0.0 for key in resource_keys} for unit in units}
    observed_steps = 0
    with gzip.open(trace_path, 'rt') as stream:
        for tick, line in enumerate(stream):
            observed_steps += 1
            frame = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            start = tick * contract['controller']['tick_seconds']
            end = start + contract['controller']['tick_seconds']
            assert frame['tick'] == tick and frame['start'] == start and frame['end'] == end
            assert set(frame['units']) == set(units)
            for unit in units:
                row = frame['units'][unit]
                assert row['phase'] == 'active' and row['model_input_rows'] == 1
                assert set(row['controller_records']) == controller_channels
                for name, record in row['controller_records'].items():
                    assert record['compartment'] == unit and record['variable'] == name
                    assert 0 <= record['measurement_time'] <= record['available_at'] <= start
                    assert _finite(record['value'])
                assert set(row['requested']) == set(row['realised']) == command_keys
                assert row['requested'] == row['realised']
                assert all(_finite(value) and 0 <= value <= 1 for value in row['requested'].values())
                assert set(row['ledger_increment']) == resource_keys
                for key, value in row['ledger_increment'].items():
                    assert _finite(value) and value >= 0
                    totals[unit][key] += value
                assert abs(row['ledger_increment']['days'] -
                           contract['controller']['tick_seconds'] / 86400) < 1e-12
                assert set(row['public_endpoint']) == public_channels
                for name, record in row['public_endpoint'].items():
                    assert record['compartment'] == unit and record['variable'] == name
                    assert record['measurement_time'] == record['available_at'] == end
                    assert _finite(record['value'])
                assert abs(row['public_endpoint']['cumulative_harvest_fresh_equivalent']['value']
                           - totals[unit]['harvest_kg_m2']) < 1e-8
        assert observed_steps == expected_steps
    for unit in units:
        ledger = result['ledgers'][unit]['per_m2']
        for key in resource_keys:
            assert abs(totals[unit][key] - ledger[key]) < max(1e-8, abs(ledger[key]) * 1e-8)
    return {'status': 'passed_private_tick_trace_audit',
            'scope': 'recorded order, causal timestamps, exact independent command realisation and resource totals; cannot prove underlying GreenLight physics',
            'steps': expected_steps,
            'unit_steps': expected_steps * len(units),
            'trace_bytes': trace_path.stat().st_size,
            'trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            'result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
            'contract_sha256': hashlib.sha256(contract_path.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.trace, args.result, args.contract)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'steps': report['steps'],
                      'trace_bytes': report['trace_bytes']}))


if __name__ == '__main__':
    main()
