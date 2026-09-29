#!/usr/bin/env python3
"""Bounded one-compartment annual runtime pilot; not a formal benchmark episode."""
from __future__ import annotations

import argparse
import faulthandler
import json
import math
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab.online_observations import OnlineObservations
from slowlab.v22_cabauw_weather import CabauwLc1Weather
from slowlab.v22_controller import commands_from_observations
from slowlab.v22_greenlight_reuse import CROP_STATES, CropLifecycle
from slowlab.v22_resources import ResourceLedger
from check_v22_online_weather import record_at_endpoint


def rss_bytes() -> int:
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    elif sys.platform == 'darwin':
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    raise RuntimeError('VmRSS unavailable')


def vmsize_bytes() -> int | None:
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('VmSize:'):
                return int(line.split()[1]) * 1024
    return None


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--pilot-days', type=int, default=365,
                        help='Bounded prefix for smoke checks; 365 runs the full pilot')
    parser.add_argument('--soil-boundary-c', type=float, default=20.0,
                        help='Constant external deep-soil boundary in degrees C')
    parser.add_argument('--max-seconds', type=int, default=10800)
    parser.add_argument('--max-rss-bytes', type=int, default=600_000_000)
    args = parser.parse_args()
    if args.max_seconds <= 0 or args.max_rss_bytes <= 0:
        raise ValueError('positive runtime and RSS bounds required')
    if not math.isfinite(args.soil_boundary_c):
        raise ValueError('finite soil boundary required')
    sys.path.insert(0, str(args.source))
    contract = json.loads((ROOT / 'configs/v22_task_contract_v3.json').read_text())
    policy = json.loads((ROOT / 'configs/v22_campaign_example_v0.json').read_text())['policy_a']
    origin = datetime(2016, 12, 31, 23, tzinfo=timezone.utc)
    weather = CabauwLc1Weather(args.cache, ROOT / 'configs/v22_weather_gapfilled_plan.json')
    campaign_days = contract['budget']['campaign_days']
    crop_days = contract['budget']['crop_days']
    cleanup_days = contract['budget']['cleanup_days']
    if (campaign_days, crop_days, cleanup_days) != (365, 180, 2):
        raise ValueError('pilot requires frozen 365/180/2-day schedule')
    if not 1 <= args.pilot_days <= campaign_days:
        raise ValueError('pilot-days outside [1, 365]')
    active_end_1 = crop_days * 288
    cleanup_end_1 = (crop_days + cleanup_days) * 288
    active_end_2 = (2 * crop_days + cleanup_days) * 288
    cleanup_end_2 = (2 * crop_days + 2 * cleanup_days) * 288
    total_steps = args.pilot_days * 288
    started = time.monotonic()
    completed_steps = 0
    phase = 'initializing'
    lifecycle = None
    ledger = None
    result = {
        'status': 'running',
        'scope': 'single-compartment weather-driven runtime pilot; fixed policy; not a formal benchmark episode or final campaign settlement',
        'pilot_days': args.pilot_days,
        'schedule_days': {'active_1': 180, 'cleanup_1': 2, 'active_2': 180,
                          'cleanup_2': 2, 'idle': 1},
        'max_seconds': args.max_seconds,
        'max_rss_bytes': args.max_rss_bytes,
        'origin_utc': origin.isoformat(),
        'native_rhs': True,
        'array_output': True,
        'soil_boundary_c': args.soil_boundary_c,
    }
    try:
        lifecycle = CropLifecycle(
            contract, args.source, start=0, cached_solver=True, native_rhs=True,
            weather=weather, weather_origin_utc=origin,
            soil_boundary_c=args.soil_boundary_c,
            array_output=True,
        )
        result['model_load_seconds'] = time.monotonic() - started
        ledger = ResourceLedger(contract, 0)
        ledger.record_event('plant', 0)
        controller = OnlineObservations(contract['observations']['controller_channels'])
        public = OnlineObservations(contract['observations']['public_channels'])
        record_at_endpoint(lifecycle.engine, weather, origin, controller, public, '0', ledger)
        with args.progress.open('w') as progress:
            for tick in range(total_steps):
                if tick in (active_end_1, active_end_2):
                    before = dict(lifecycle.engine.state)
                    lifecycle.stop()
                    ledger.record_event('stop', lifecycle.engine.clock)
                    if any(lifecycle.engine.state[k] != value for k, value in before.items()
                           if k not in CROP_STATES):
                        raise AssertionError('facility state changed during crop removal')
                if tick == cleanup_end_1:
                    before = dict(lifecycle.engine.state)
                    lifecycle.replant()
                    ledger.record_event('plant', lifecycle.engine.clock)
                    if any(lifecycle.engine.state[k] != value for k, value in before.items()
                           if k not in CROP_STATES):
                        raise AssertionError('facility state changed during replant')
                if tick < active_end_1 or cleanup_end_1 <= tick < active_end_2:
                    phase = 'active'
                elif tick < cleanup_end_1 or active_end_2 <= tick < cleanup_end_2:
                    phase = 'cleanup'
                else:
                    phase = 'idle'
                now = lifecycle.engine.clock
                command, decision = commands_from_observations(
                    contract, controller, '0', phase=phase,
                    policy=policy if phase == 'active' else None,
                )
                if any(rec['available_at'] > now for rec in decision['sensor_records'].values()):
                    raise AssertionError('controller used unarrived sensor')
                state = lifecycle.step(command, now + 300)
                if any(not math.isfinite(value) for value in state.values()):
                    raise AssertionError('nonfinite physical state')
                ledger.add_segment(lifecycle.engine.model.full_sol, now, now + 300, phase)
                record_at_endpoint(lifecycle.engine, weather, origin, controller, public, '0', ledger)
                completed_steps = tick + 1
                if completed_steps % 12 == 0:
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError('pilot exceeded wall-time bound')
                    if rss_bytes() > args.max_rss_bytes:
                        raise MemoryError('pilot exceeded RSS bound')
                if completed_steps % 288 == 0:
                    entry = {'day': completed_steps // 288, 'phase': phase,
                             'elapsed_seconds': time.monotonic() - started,
                             'rss_bytes': rss_bytes(), 'vmsize_bytes': vmsize_bytes(),
                             'clock_seconds': lifecycle.engine.clock,
                             'harvest_kg_m2': ledger.summary()['per_m2']['harvest_kg_m2']}
                    progress.write(json.dumps(entry) + '\n')
                    progress.flush()
        if completed_steps != total_steps or lifecycle.engine.clock != args.pilot_days * 86400:
            raise AssertionError('incomplete annual schedule')
        if args.pilot_days == campaign_days:
            if lifecycle.mode != 'empty' or lifecycle.engine.clock < lifecycle.ready_at:
                raise AssertionError('final cleanup not complete')
            for name, expected in [('active', 360), ('cleanup', 4), ('idle', 1)]:
                if abs(ledger.totals[name]['days'] - expected) > 1e-8:
                    raise AssertionError('incorrect phase accounting: ' + name)
        result.update(status=('passed_annual_runtime_pilot' if args.pilot_days == campaign_days
                              else 'passed_prefix_runtime_pilot'), final_state=dict(lifecycle.engine.state),
                      lifecycle_events=lifecycle.events, ledger=ledger.summary())
    except Exception as exc:
        traceback.print_exc()
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    finally:
        result['completed_steps'] = completed_steps
        result['completed_days'] = completed_steps / 288
        result['last_phase'] = phase
        result['elapsed_seconds'] = time.monotonic() - started
        result['peak_rss_kib_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if ledger is not None:
            result['ledger_at_stop'] = ledger.summary()
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: result[key] for key in
                          ('status', 'completed_days', 'last_phase', 'elapsed_seconds',
                           'peak_rss_kib_linux', *(['error'] if 'error' in result else []))}),
              flush=True)
    if result['status'] not in ('passed_annual_runtime_pilot', 'passed_prefix_runtime_pilot'):
        sys.exit(1)


if __name__ == '__main__':
    main()
