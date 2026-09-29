#!/usr/bin/env python3
"""Bounded four-unit staggered annual executor pilot, never an LLM episode."""
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

from slowlab.online_observations import PackedOnlineObservations
from slowlab.v22_cabauw_weather import CabauwLc1Weather
from slowlab.v22_controller import commands_from_observations
from slowlab.v22_feedback_view import FeedbackView
from slowlab.v22_greenlight_reuse import CROP_STATES, CropLifecycle
from slowlab.v22_resources import ResourceLedger, realise_independent_commands
from check_v22_online_weather import record_at_endpoint
from check_v22_annual_runtime_pilot import rss_bytes, vmsize_bytes


STOP_DAYS = {'0': (14, 196), '1': (30, 212),
             '2': (60, 242), '3': (180, 362)}
REPLANT_DAYS = {'0': (16, 198), '1': (32, 214),
                '2': (62, 244), '3': (182,)}


def main() -> None:
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--pilot-days', type=int, default=365)
    parser.add_argument('--max-seconds', type=int, default=10800)
    parser.add_argument('--max-rss-bytes', type=int, default=350_000_000)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--progress', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.pilot_days <= 365 or args.max_seconds <= 0 or args.max_rss_bytes <= 0:
        raise ValueError('invalid bounded pilot size or limits')
    sys.path.insert(0, str(args.source))
    contract = json.loads((ROOT / 'configs/v22_task_contract_v3.json').read_text())
    if (contract['budget']['campaign_days'], contract['budget']['crop_days'],
            contract['budget']['cleanup_days']) != (365, 180, 2):
        raise ValueError('pilot requires frozen 365/180/2-day contract')
    policies = json.loads((ROOT / 'configs/v22_campaign_example_v0.json').read_text())
    units = tuple(str(i) for i in range(contract['facility']['compartments']))
    if units != ('0', '1', '2', '3'):
        raise ValueError('pilot requires four compartments')
    origin = datetime(2016, 12, 31, 23, tzinfo=timezone.utc)
    weather = CabauwLc1Weather(args.cache, ROOT / 'configs/v22_weather_gapfilled_plan.json')
    controller = PackedOnlineObservations(contract['observations']['controller_channels'])
    public = PackedOnlineObservations(contract['observations']['public_channels'])
    full = FeedbackView('full', public, units)
    endpoint = FeedbackView('endpoint', public, units)
    started = time.monotonic()
    completed_steps = 0
    lifecycle: dict[str, CropLifecycle] = {}
    ledgers: dict[str, ResourceLedger] = {}
    event_log = []
    result = {
        'status': 'running',
        'scope': 'four staggered compartments, fixed policies and Cabauw weather; runtime/lifecycle pilot only, not formal LLM benchmark or empirical validation',
        'pilot_days': args.pilot_days,
        'max_seconds': args.max_seconds,
        'max_rss_bytes': args.max_rss_bytes,
        'origin_utc': origin.isoformat(),
        'stop_days': STOP_DAYS,
        'replant_days': REPLANT_DAYS,
        'packed_observations': True,
        'native_rhs': True,
        'array_output': True,
    }
    try:
        for unit in units:
            lifecycle[unit] = CropLifecycle(
                contract, args.source, start=0, cached_solver=True, native_rhs=True,
                weather=weather, weather_origin_utc=origin, soil_boundary_c=20.,
                array_output=True,
            )
            ledgers[unit] = ResourceLedger(contract, 0)
            ledgers[unit].record_event('plant', 0)
            full.executor_set_status(unit, 'active', 'start', ledger=ledgers[unit])
            endpoint.executor_set_status(unit, 'active', 'start', ledger=ledgers[unit])
            record_at_endpoint(lifecycle[unit].engine, weather, origin,
                               controller, public, unit, ledgers[unit])
            event_log.append({'unit': unit, 'event': 'plant', 'day': 0})
            if rss_bytes() > args.max_rss_bytes:
                raise MemoryError('four-unit model load exceeded RSS bound')
        result['model_load_seconds'] = time.monotonic() - started
        with args.progress.open('w') as progress:
            for tick in range(args.pilot_days * 288):
                now = tick * 300
                if any(lifecycle[unit].engine.clock != now for unit in units):
                    raise AssertionError('physical clocks diverged')
                if tick % 288 == 0:
                    day = tick // 288
                    for unit in units:
                        life = lifecycle[unit]
                        ledger = ledgers[unit]
                        if day in STOP_DAYS[unit]:
                            before = dict(life.engine.state)
                            life.stop()
                            ledger.record_event('stop', now)
                            if any(life.engine.state[k] != value for k, value in before.items()
                                   if k not in CROP_STATES):
                                raise AssertionError('facility state changed at stop')
                            for view in (full, endpoint):
                                view.executor_set_status(unit, 'cleanup', 'stop')
                            endpoint.executor_release_final(unit, 'stop', ledger)
                            event_log.append({'unit': unit, 'event': 'stop', 'day': day})
                        if day in REPLANT_DAYS[unit]:
                            before = dict(life.engine.state)
                            life.replant()
                            ledger.record_event('plant', now)
                            if any(life.engine.state[k] != value for k, value in before.items()
                                   if k not in CROP_STATES):
                                raise AssertionError('facility state changed at replant')
                            for view in (full, endpoint):
                                view.executor_set_status(unit, 'active', 'start', ledger=ledger)
                            event_log.append({'unit': unit, 'event': 'replant', 'day': day})
                        if life.mode == 'empty' and now >= life.ready_at and day not in REPLANT_DAYS[unit]:
                            for view in (full, endpoint):
                                view.executor_set_status(unit, 'idle')
                requested = {}
                phases = {}
                for unit in units:
                    life = lifecycle[unit]
                    phase = ('active' if life.mode == 'active' else
                             'cleanup' if now < life.ready_at else 'idle')
                    phases[unit] = phase
                    policy = policies['policy_a' if int(unit) % 2 == 0 else 'policy_b']
                    requested[unit], decision = commands_from_observations(
                        contract, controller, unit, phase=phase,
                        policy=policy if phase == 'active' else None,
                    )
                    if any(rec['available_at'] > now for rec in decision['sensor_records'].values()):
                        raise AssertionError('controller used future observation')
                realised = realise_independent_commands(contract, requested)
                if realised != requested:
                    raise AssertionError('adequate supply changed command')
                for unit in units:
                    life = lifecycle[unit]
                    state = life.step(realised[unit], now + 300)
                    if any(not math.isfinite(value) for value in state.values()):
                        raise AssertionError('nonfinite physical state')
                    ledgers[unit].add_segment(life.engine.model.full_sol, now,
                                              now + 300, phases[unit])
                    if len(life.engine.model.input_data) != 1:
                        raise AssertionError('future input rows present')
                # All compartments reach the endpoint before any new sensor is delivered.
                for unit in units:
                    record_at_endpoint(lifecycle[unit].engine, weather, origin,
                                       controller, public, unit, ledgers[unit])
                completed_steps = tick + 1
                if completed_steps % 12 == 0:
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError('four-unit annual pilot exceeded wall-time bound')
                    if rss_bytes() > args.max_rss_bytes:
                        raise MemoryError('four-unit annual pilot exceeded RSS bound')
                if completed_steps % 288 == 0:
                    progress.write(json.dumps({
                        'day': completed_steps // 288,
                        'elapsed_seconds': time.monotonic() - started,
                        'rss_bytes': rss_bytes(),
                        'vmsize_bytes': vmsize_bytes(),
                        'clocks': {unit: lifecycle[unit].engine.clock for unit in units},
                        'phase': phases,
                        'harvest_kg_m2': {unit: ledgers[unit].summary()['per_m2']['harvest_kg_m2']
                                          for unit in units},
                    }) + '\n')
                    progress.flush()
        if completed_steps != args.pilot_days * 288:
            raise AssertionError('incomplete bounded schedule')
        if args.pilot_days == 365:
            now = 365 * 86400
            for unit in units:
                if lifecycle[unit].mode == 'active':
                    lifecycle[unit].stop()
                    ledgers[unit].record_event('stop', now)
                    for view in (full, endpoint):
                        view.executor_set_status(unit, 'cleanup', 'stop')
                    endpoint.executor_release_final(unit, 'stop', ledgers[unit])
                    event_log.append({'unit': unit, 'event': 'deadline_stop', 'day': 365})
            if any(lifecycle[unit].mode != 'empty' for unit in units):
                raise AssertionError('active crop remained at deadline')
        for unit in units:
            now = args.pilot_days * 86400
            if len(full.history(unit, 'air_temperature_c', start=now, end=now)) != 1:
                raise AssertionError('missing final public endpoint')
            try:
                endpoint.history(unit, 'air_temperature_c')
            except PermissionError:
                pass
            else:
                raise AssertionError('Endpoint exposed running history')
        result.update(status=('passed_four_unit_annual_pilot' if args.pilot_days == 365
                              else 'passed_four_unit_annual_prefix'),
                      final_states={unit: dict(lifecycle[unit].engine.state) for unit in units},
                      ledgers={unit: ledgers[unit].summary() for unit in units},
                      last_feedback={unit: endpoint.final_aggregate(unit) for unit in units},
                      lifecycle_events=event_log)
    except Exception as exc:
        traceback.print_exc()
        context = {'tick': locals().get('tick'), 'unit': locals().get('unit'),
                   'clock_start_seconds': locals().get('now')}
        if context['unit'] in lifecycle:
            engine = lifecycle[context['unit']].engine
            context.update(mode=lifecycle[context['unit']].mode,
                           committed_state=dict(engine.state),
                           requested=locals().get('requested', {}).get(context['unit']),
                           realised=locals().get('realised', {}).get(context['unit']))
            solver = engine._cached_solver
            native = solver.rhs if solver is not None else None
            fault = getattr(native, 'last_fault', None)
            if fault is not None:
                context['native_fault'] = fault
                context['state_names'] = list(engine.model.states)
                context['input_names'] = ['Time', *(key for key in engine.model.inputs if key != 'Time')]
                try:
                    import numpy as np
                    import warnings
                    from slowlab.v22_cached_solver import CachedGreenLightSolver
                    reference = CachedGreenLightSolver(engine.model).rhs
                    with np.errstate(all='warn'), warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        reference_out = reference(fault['time'], np.asarray(fault['state']),
                                                  np.asarray(fault['inputs'])[None, :])
                    context['reference_rhs'] = {
                        'derivative_all_finite': bool(np.isfinite(reference_out).all()),
                        'derivative': [float(value) if np.isfinite(value) else None
                                       for value in reference_out],
                        'warnings': list(dict.fromkeys(str(w.message) for w in caught)),
                    }
                except Exception as diagnostic_exc:
                    context['reference_rhs_error'] = type(diagnostic_exc).__name__ + ': ' + str(diagnostic_exc)
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc),
                      failure_context=context)
    finally:
        result['completed_steps'] = completed_steps
        result['completed_days'] = completed_steps / 288
        result['elapsed_seconds'] = time.monotonic() - started
        result['peak_rss_kib_linux'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: result[key] for key in
                          ('status', 'completed_days', 'elapsed_seconds',
                           'peak_rss_kib_linux', *(['error'] if 'error' in result else []))}),
              flush=True)
    if result['status'] not in ('passed_four_unit_annual_pilot', 'passed_four_unit_annual_prefix'):
        sys.exit(1)


if __name__ == '__main__':
    main()
