#!/usr/bin/env python3
"""Stream an independent structural audit of the private annual campaign trace."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.sensor_noise import SensorNoise, verify_trace_row  # noqa: E402

FLUX_KEYS = ('heat_kwh_m2', 'light_kwh_m2', 'co2_kg_m2',
             'harvest_kg_m2', 'transpiration_kg_m2', 'days')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit(result_path: Path, progress_path: Path, trace_path: Path) -> dict:
    result = json.loads(result_path.read_text())
    assert result['status'] == 'passed_annual_dynamic_campaign'
    assert result['pilot_days'] == 365 and result['final_clock'] == 365 * 86400
    assert result['elapsed_seconds'] < result['max_seconds']
    assert result['peak_rss_kib_linux'] * 1024 < result['max_rss_bytes']
    settlement = result['settlement']
    UNITS = set(settlement['ledger_by_unit'])
    assert UNITS == {str(i) for i in range(len(UNITS))} and UNITS
    assert settlement['clock'] == 365 * 86400
    assert settlement['trace_complete'] is True
    assert settlement['trace_ticks'] == result['trace_ticks'] == 365 * 288
    assert settlement['trace_sha256'] == result['trace_sha256']
    assert settlement['starts'] >= 4 and settlement['recommendation_fallback'] is False
    assert set(settlement['ledger_by_unit']) == UNITS
    progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert len(progress) == 365
    last_elapsed = -1.0
    for day, row in enumerate(progress, 1):
        assert row['day'] == day and row['clock_seconds'] == day * 86400
        assert row['trace_ticks'] == day * 288
        assert set(row['unit_status']) == set(row['cumulative_per_m2']) == UNITS
        assert last_elapsed < row['elapsed_seconds'] < result['max_seconds']
        assert 0 < row['rss_bytes'] < result['max_rss_bytes']
        last_elapsed = row['elapsed_seconds']
    assert abs(progress[-1]['elapsed_seconds'] - result['elapsed_seconds']) < 3

    noise_record = result.get('sensor_noise')
    noise = None
    if noise_record is not None:
        config_path = ROOT / noise_record['config']
        assert hashlib.sha256(config_path.read_bytes()).hexdigest() == noise_record['config_sha256']
        noise = SensorNoise.from_file(config_path, noise_record['seed'], noise_record['setting'])
        assert settlement['sensor_noise'] == noise.describe()
    previous = {}
    dropped_readings = 0
    plain_digest = hashlib.sha256()
    accrued = {unit: {key: 0.0 for key in FLUX_KEYS} for unit in UNITS}
    ticks = 0
    with gzip.open(trace_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            plain_digest.update(line.encode())
            frame = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            assert frame['tick'] == ticks
            assert frame['start'] == ticks * 300 and frame['end'] == (ticks + 1) * 300
            assert set(frame['units']) == UNITS
            for unit, entry in frame['units'].items():
                assert entry['model_input_rows'] == 1
                assert entry['requested'] == entry['realised']
                assert set(entry['ledger_increment']) == set(FLUX_KEYS)
                assert all(rec['available_at'] <= frame['start']
                           for rec in entry['controller_records'].values())
                noisy = noise.channels if noise is not None else frozenset()
                assert all(rec['available_at'] == frame['end']
                           for name, rec in entry['public_endpoint'].items() if name not in noisy)
                if noise is not None:
                    verify_trace_row(noise, unit, frame['end'], entry, previous.get(unit))
                    dropped_readings += len(entry['sensor_dropped'])
                    previous[unit] = entry
                for key in FLUX_KEYS:
                    value = entry['ledger_increment'][key]
                    assert math.isfinite(value)
                    accrued[unit][key] += value
            ticks += 1
    assert ticks == settlement['trace_ticks']
    assert plain_digest.hexdigest() == settlement['trace_sha256']
    for unit in UNITS:
        actual = settlement['ledger_by_unit'][unit]['per_m2']
        for key in FLUX_KEYS:
            assert math.isclose(accrued[unit][key], actual[key], rel_tol=1e-8, abs_tol=1e-7), (unit, key)
            assert math.isclose(progress[-1]['cumulative_per_m2'][unit][key], actual[key],
                                rel_tol=1e-8, abs_tol=1e-7), (unit, key)
    event_log = result['event_log']
    assert event_log[0]['event'] == 'campaign_open'
    assert sum(event['event'] == 'start' for event in event_log) == settlement['starts']
    assert any(event['event'] == 'observe' and event['clock'] == 14 * 86400
               for event in event_log)
    assert any(event['event'] == 'stop' and event.get('unit') == '0'
               for event in event_log)
    assert event_log[-1] == {'event': 'recommend', 'clock': 365 * 86400, 'fallback': False}
    assert result['midway_decision']['record_count'] > 0
    assert result['midway_decision']['selected_next_policy'] in ('policy_b', 'policy_c')
    return {'status': 'passed_annual_dynamic_structural_audit',
            'scope': 'mechanical causality/resource/settlement audit only; no real-greenhouse validation',
            'days': 365, 'ticks': ticks, 'starts': settlement['starts'],
            'sensor_noise': noise_record, 'dropped_readings': dropped_readings if noise else None,
            'wall_seconds': result['elapsed_seconds'],
            'peak_rss_bytes': result['peak_rss_kib_linux'] * 1024,
            'result_sha256': sha256(result_path),
            'progress_sha256': sha256(progress_path),
            'trace_compressed_sha256': sha256(trace_path),
            'trace_uncompressed_sha256': plain_digest.hexdigest(),
            'per_unit_per_m2': {unit: settlement['ledger_by_unit'][unit]['per_m2']
                               for unit in sorted(UNITS)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.result, args.progress, args.trace)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'days', 'ticks', 'wall_seconds')}))


if __name__ == '__main__':
    main()
