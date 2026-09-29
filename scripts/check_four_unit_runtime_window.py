#!/usr/bin/env python3
"""Bounded four-compartment same-clock weather/runtime gate, not a campaign."""
from __future__ import annotations

import argparse
import gzip
from contextlib import nullcontext
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

from slowlab.online_observations import OnlineObservations, PackedOnlineObservations
from slowlab.cabauw_weather import CabauwLc1Weather
from slowlab.controller import commands_from_observations
from slowlab.feedback_view import FeedbackView
from slowlab.greenlight_reuse import ReusableGreenLight
from slowlab.resources import ResourceLedger, realise_independent_commands
from check_online_weather import record_at_endpoint
from check_annual_runtime_pilot import rss_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--days', type=int, default=1)
    parser.add_argument('--packed-observations', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=900)
    parser.add_argument('--max-rss-bytes', type=int, default=350_000_000)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--progress', type=Path, required=True)
    parser.add_argument('--trace-gz', type=Path, help='private per-tick requested→realised→observed JSONL gzip audit trace')
    args = parser.parse_args()
    if not 1 <= args.days <= 7:
        raise ValueError('runtime window must be 1–7 days')
    if args.max_seconds <= 0 or args.max_rss_bytes <= 0:
        raise ValueError('positive limits required')
    sys.path.insert(0, str(args.source))
    contract = json.loads((ROOT / 'configs/task_contract_v3.json').read_text())
    policies = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
    units = tuple(str(i) for i in range(contract['facility']['compartments']))
    if units != ('0', '1', '2', '3'):
        raise ValueError('pilot requires four contracted compartments')
    origin = datetime(2016, 12, 31, 23, tzinfo=timezone.utc)
    weather = CabauwLc1Weather(args.cache, ROOT / 'configs/weather_gapfilled_plan.json')
    store_class = PackedOnlineObservations if args.packed_observations else OnlineObservations
    controller = store_class(contract['observations']['controller_channels'])
    public = store_class(contract['observations']['public_channels'])
    full = FeedbackView('full', public, units)
    endpoint = FeedbackView('endpoint', public, units)
    started = time.monotonic()
    completed_steps = 0
    engines = {}
    ledgers = {}
    result = {'status': 'running', 'scope': 'four active compartments, one clock; fixed policies; bounded runtime window; no lifecycle or formal benchmark',
              'days': args.days, 'max_seconds': args.max_seconds,
              'max_rss_bytes': args.max_rss_bytes, 'native_rhs': True,
              'array_output': True, 'packed_observations': args.packed_observations,
              'origin_utc': origin.isoformat(),
              'trace_gz': str(args.trace_gz) if args.trace_gz else None}
    try:
        for unit in units:
            engines[unit] = ReusableGreenLight(
                contract, args.source, start=0, cached_solver=True, native_rhs=True,
                weather=weather, weather_origin_utc=origin, soil_boundary_c=20.,
                array_output=True,
            )
            ledgers[unit] = ResourceLedger(contract, 0)
            ledgers[unit].record_event('plant', 0)
            full.executor_set_status(unit, 'active', 'start', ledger=ledgers[unit])
            endpoint.executor_set_status(unit, 'active', 'start', ledger=ledgers[unit])
            record_at_endpoint(engines[unit], weather, origin, controller, public,
                               unit, ledgers[unit])
            if rss_bytes() > args.max_rss_bytes:
                raise MemoryError('model loading exceeded RSS bound')
        result['model_load_seconds'] = time.monotonic() - started
        trace_context = gzip.open(args.trace_gz, 'wt', compresslevel=1) if args.trace_gz else nullcontext(None)
        with args.progress.open('w') as progress, trace_context as trace:
            for tick in range(args.days * 288):
                now = tick * 300
                if any(engine.clock != now for engine in engines.values()):
                    raise AssertionError('compartment clocks diverged')
                requested = {}
                decisions = {}
                for unit in units:
                    policy = policies['policy_a' if int(unit) % 2 == 0 else 'policy_b']
                    requested[unit], decision = commands_from_observations(
                        contract, controller, unit, phase='active', policy=policy,
                    )
                    decisions[unit] = decision['sensor_records']
                    if any(record['available_at'] > now
                           for record in decision['sensor_records'].values()):
                        raise AssertionError('controller used unarrived observation')
                realised = realise_independent_commands(contract, requested)
                if realised != requested:
                    raise AssertionError('adequate central supply altered a request')
                deltas = {}
                for unit in units:
                    engine = engines[unit]
                    state = engine.step(realised[unit], now + 300)
                    if any(not math.isfinite(value) for value in state.values()):
                        raise AssertionError('nonfinite physical state')
                    deltas[unit] = ledgers[unit].add_segment(engine.model.full_sol, now, now + 300,
                                                            'active')
                    if len(engine.model.input_data) != 1:
                        raise AssertionError('future input rows present')
                # Publish an endpoint only after all four physical steps complete.
                for unit in units:
                    record_at_endpoint(engines[unit], weather, origin, controller,
                                       public, unit, ledgers[unit])
                if trace is not None:
                    frame = {'tick': tick, 'start': now, 'end': now + 300,
                             'units': {unit: {'phase': 'active',
                                             'controller_records': decisions[unit],
                                             'requested': requested[unit],
                                             'realised': realised[unit],
                                             'ledger_increment': deltas[unit],
                                             'public_endpoint': {name: public.latest(unit, name)
                                                                 for name in contract['observations']['public_channels']},
                                             'model_input_rows': len(engines[unit].model.input_data)}
                                       for unit in units}}
                    trace.write(json.dumps(frame, separators=(',', ':'), allow_nan=False) + '\n')
                completed_steps = tick + 1
                if completed_steps % 12 == 0:
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError('four-unit window exceeded wall-time bound')
                    if rss_bytes() > args.max_rss_bytes:
                        raise MemoryError('four-unit window exceeded RSS bound')
                if completed_steps % 288 == 0:
                    progress.write(json.dumps({
                        'day': completed_steps // 288,
                        'elapsed_seconds': time.monotonic() - started,
                        'rss_bytes': rss_bytes(),
                        'clocks': {unit: engines[unit].clock for unit in units},
                    }) + '\n')
                    progress.flush()
        if completed_steps != args.days * 288:
            raise AssertionError('incomplete four-unit window')
        for unit in units:
            if len(full.history(unit, 'air_temperature_c')) != completed_steps + 1:
                raise AssertionError('missing public endpoint')
            try:
                endpoint.history(unit, 'air_temperature_c')
            except PermissionError:
                pass
            else:
                raise AssertionError('Endpoint exposed running science')
            if endpoint.final_aggregate(unit) is not None:
                raise AssertionError('Endpoint released a running aggregate')
        result.update(status='passed_four_unit_runtime_window',
                      final_states={unit: dict(engines[unit].state) for unit in units},
                      ledgers={unit: ledgers[unit].summary() for unit in units},
                      public_air_records_per_unit=completed_steps + 1)
    except Exception as exc:
        traceback.print_exc()
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
    finally:
        result['completed_steps'] = completed_steps
        result['elapsed_seconds'] = time.monotonic() - started
        result['peak_rss_kib_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: result[key] for key in
                          ('status', 'completed_steps', 'elapsed_seconds',
                           'peak_rss_kib_linux', *(['error'] if 'error' in result else []))}),
              flush=True)
    if result['status'] != 'passed_four_unit_runtime_window':
        sys.exit(1)


if __name__ == '__main__':
    main()
