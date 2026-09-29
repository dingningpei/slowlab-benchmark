#!/usr/bin/env python3
"""Structural audit for the four-unit annual or bounded-prefix runtime pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_at_day_start(unit: str, day_start: int, stops: dict, replants: dict) -> str:
    for stop in stops[unit]:
        if stop <= day_start < stop + 2:
            return 'cleanup'
        if day_start >= stop + 2 and not any(stop < start <= day_start for start in replants[unit]):
            return 'idle'
    return 'active'


def audit(result_path: Path, progress_path: Path, contract_path: Path) -> dict:
    result = json.loads(result_path.read_text())
    progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
    contract = json.loads(contract_path.read_text())
    days = result['pilot_days']
    assert 1 <= days <= 365
    expected_status = ('passed_four_unit_annual_pilot' if days == 365
                       else 'passed_four_unit_annual_prefix')
    assert result['status'] == expected_status
    assert result['completed_steps'] == days * 288
    assert abs(result['completed_days'] - days) < 1e-9
    assert result['packed_observations'] is True
    assert result['max_seconds'] <= 10800
    assert result['max_rss_bytes'] <= 350_000_000
    assert result['elapsed_seconds'] < result['max_seconds']
    assert result['peak_rss_kib_linux'] * 1024 < result['max_rss_bytes']
    assert len(progress) == days
    units = tuple(str(i) for i in range(4))
    assert result['stop_days'] == {'0': [14, 196], '1': [30, 212],
                                  '2': [60, 242], '3': [180, 362]}
    assert result['replant_days'] == {'0': [16, 198], '1': [32, 214],
                                     '2': [62, 244], '3': [182]}
    expected_events = {(unit, 'plant', 0) for unit in units}
    for unit in units:
        expected_events |= {(unit, 'stop', day) for day in result['stop_days'][unit] if day < days}
        expected_events |= {(unit, 'replant', day) for day in result['replant_days'][unit] if day < days}
        if days == 365 and unit != '3':
            expected_events.add((unit, 'deadline_stop', 365))
    observed_events = [(row['unit'], row['event'], row['day']) for row in result['lifecycle_events']]
    assert len(observed_events) == len(set(observed_events)) == len(expected_events)
    assert set(observed_events) == expected_events
    phase_counts = {unit: {'active': 0, 'cleanup': 0, 'idle': 0} for unit in units}
    last_harvest = {unit: -math.inf for unit in units}
    last_elapsed = -math.inf
    for day, row in enumerate(progress, 1):
        assert row['day'] == day
        assert 0 < row['rss_bytes'] < result['max_rss_bytes']
        assert last_elapsed < row['elapsed_seconds'] < result['max_seconds']
        assert set(row['clocks']) == set(row['phase']) == set(row['harvest_kg_m2']) == set(units)
        for unit in units:
            assert row['clocks'][unit] == day * 86400
            expected = phase_at_day_start(unit, day - 1,
                                          result['stop_days'], result['replant_days'])
            assert row['phase'][unit] == expected
            phase_counts[unit][expected] += 1
            harvest = row['harvest_kg_m2'][unit]
            assert math.isfinite(harvest) and harvest >= last_harvest[unit] - 1e-10
            if day > 1 and expected != 'active':
                assert abs(harvest - last_harvest[unit]) < 1e-10
            last_harvest[unit] = harvest
        last_elapsed = row['elapsed_seconds']
    assert abs(result['elapsed_seconds'] - last_elapsed) < 1
    assert result['peak_rss_kib_linux'] * 1024 >= max(row['rss_bytes'] for row in progress)
    assert set(result['ledgers']) == set(result['final_states']) == set(result['last_feedback']) == set(units)
    for unit in units:
        state = result['final_states'][unit]
        assert len(state) == 28 and all(math.isfinite(value) for value in state.values())
        ledger = result['ledgers'][unit]
        for phase, count in phase_counts[unit].items():
            assert abs(ledger['by_phase_per_m2'][phase]['days'] - count) < 1e-6
        assert abs(ledger['per_m2']['days'] - days) < 1e-6
        assert abs(ledger['per_m2']['harvest_kg_m2'] - last_harvest[unit]) < 1e-9
        starts = sum(1 for row in observed_events if row[0] == unit and row[1] in ('plant', 'replant'))
        stops = sum(1 for row in observed_events if row[0] == unit and row[1] in ('stop', 'deadline_stop'))
        cost = (starts * contract['economics']['planting_eur_per_m2']
                + stops * contract['economics']['cleanup_eur_per_m2'])
        assert abs(ledger['event_cost_eur_m2'] - cost) < 1e-9
        last = result['last_feedback'][unit]
        if stops:
            assert last is not None and last['closed_at'] == max(
                row[2] for row in observed_events if row[0] == unit and row[1] in ('stop', 'deadline_stop')) * 86400
        else:
            assert last is None
    return {'status': 'passed_four_unit_structural_audit',
            'scope': 'fixed-policy runtime, clock, lifecycle and accounting structure only; no independent tick trace, empirical greenhouse validation or LLM benchmark',
            'result_sha256': sha256(result_path),
            'progress_sha256': sha256(progress_path),
            'contract_sha256': sha256(contract_path),
            'days': days,
            'steps': result['completed_steps'],
            'wall_seconds': result['elapsed_seconds'],
            'peak_rss_bytes': result['peak_rss_kib_linux'] * 1024,
            'phase_days': phase_counts,
            'events': observed_events}


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
