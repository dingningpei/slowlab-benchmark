#!/usr/bin/env python3
"""Independent structural audit of a bounded single-unit annual pilot artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_phase(day: int) -> str:
    if day <= 180 or 183 <= day <= 362:
        return 'active'
    if day in (181, 182, 363, 364):
        return 'cleanup'
    if day == 365:
        return 'idle'
    raise ValueError('day outside annual schedule')


def audit(result_path: Path, progress_path: Path, contract_path: Path) -> dict:
    result = json.loads(result_path.read_text())
    progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
    contract = json.loads(contract_path.read_text())
    assert result['status'] == 'passed_annual_runtime_pilot'
    assert result['pilot_days'] == 365
    assert result['completed_steps'] == 365 * 288
    assert result['completed_days'] == 365
    assert result['last_phase'] == 'idle'
    assert result['max_seconds'] == 10800
    assert result['max_rss_bytes'] <= 600_000_000
    assert result['elapsed_seconds'] < result['max_seconds']
    assert result['peak_rss_kib_linux'] * 1024 < result['max_rss_bytes']
    assert len(progress) == 365
    previous_elapsed = -math.inf
    previous_harvest = -math.inf
    for day, row in enumerate(progress, start=1):
        assert row['day'] == day
        assert row['clock_seconds'] == day * 86400
        assert row['phase'] == expected_phase(day)
        assert previous_elapsed < row['elapsed_seconds'] < result['max_seconds']
        assert 0 < row['rss_bytes'] < result['max_rss_bytes']
        assert row['harvest_kg_m2'] >= previous_harvest - 1e-10
        if row['phase'] != 'active':
            assert abs(row['harvest_kg_m2'] - previous_harvest) < 1e-10
        previous_elapsed = row['elapsed_seconds']
        previous_harvest = row['harvest_kg_m2']
    assert abs(result['elapsed_seconds'] - progress[-1]['elapsed_seconds']) < 1
    assert result['peak_rss_kib_linux'] * 1024 >= max(row['rss_bytes'] for row in progress)
    events = [(event['event'], event['time'] / 86400)
              for event in result['lifecycle_events']]
    assert events == [('stop', 180), ('replant', 182), ('stop', 362)]
    ledger = result['ledger']
    assert result['ledger_at_stop'] == ledger
    for phase, days in [('active', 360), ('cleanup', 4), ('idle', 1)]:
        assert abs(ledger['by_phase_per_m2'][phase]['days'] - days) < 1e-6
    assert abs(ledger['per_m2']['days'] - 365) < 1e-6
    assert abs(ledger['per_m2']['harvest_kg_m2'] - progress[-1]['harvest_kg_m2']) < 1e-10
    expected_cost = 2 * (contract['economics']['planting_eur_per_m2'] +
                         contract['economics']['cleanup_eur_per_m2'])
    assert ledger['event_cost_eur_m2'] == expected_cost
    expected_occupied = 364 * contract['facility']['floor_area_m2']
    assert abs(ledger['occupied_m2_days'] - expected_occupied) < 1e-6
    assert len(result['final_state']) == 28
    assert all(math.isfinite(v) for v in result['final_state'].values())
    return {
        'status': 'passed_structural_audit',
        'scope': 'single-compartment runtime/cause/accounting checks only; not empirical climate or crop validation',
        'result_sha256': sha256(result_path),
        'progress_sha256': sha256(progress_path),
        'contract_sha256': sha256(contract_path),
        'days': len(progress),
        'steps': result['completed_steps'],
        'wall_seconds': result['elapsed_seconds'],
        'model_load_seconds': result['model_load_seconds'],
        'peak_rss_bytes': result['peak_rss_kib_linux'] * 1024,
        'phase_days': {phase: ledger['by_phase_per_m2'][phase]['days']
                       for phase in ('active', 'cleanup', 'idle')},
        'lifecycle_events': events,
        'event_cost_eur_m2': ledger['event_cost_eur_m2'],
        'harvest_kg_m2_synthetic': ledger['per_m2']['harvest_kg_m2'],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.result, args.progress, args.contract)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'days': report['days'],
                      'wall_seconds': report['wall_seconds'],
                      'peak_rss_bytes': report['peak_rss_bytes']}))


if __name__ == '__main__':
    main()
