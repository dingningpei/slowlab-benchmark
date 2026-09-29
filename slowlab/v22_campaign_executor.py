"""Phase-1 event-driven four-compartment campaign executor.

Only ``dispatch`` results belong in an agent tool response. The executor and
its private lifecycle, weather, controller store and ledger must stay in a
separate process/tool implementation. This module does not prove Python object
sandboxing or physical fidelity of GreenLight.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path

from .online_observations import PackedOnlineObservations
from .task_contract_v22 import validate_policy
from .v22_controller import commands_from_observations
from .v22_feedback_view import FeedbackView
from .v22_greenlight_reuse import CropLifecycle
from .v22_resources import ResourceLedger, realise_independent_commands


def _seconds(day: Real, limit_days: Real, tick_seconds: int) -> int:
    if isinstance(day, bool) or not isinstance(day, Real) or not math.isfinite(day):
        raise ValueError('day must be finite')
    value = float(day) * 86400
    rounded = round(value / tick_seconds) * tick_seconds
    if not 0 <= rounded <= limit_days * 86400 or abs(value - rounded) > 1e-6:
        raise ValueError('day must be an in-range 300-second endpoint')
    return int(rounded)


class CampaignExecutor:
    """Executor-owned physical campaign, with a JSON-only public dispatch surface.

    Injecting ``lifecycle_factory`` and ``sample_endpoint`` permits fast
    mechanical tests. Production defaults to the pinned GreenLight lifecycle
    and reached raw-endpoint sensor bridge.
    """

    def __init__(self, contract: dict, source: Path, weather, *,
                 feedback_mode: str, fallback_policy: dict,
                 origin_utc: datetime = datetime(2016, 12, 31, 23, tzinfo=timezone.utc),
                 lifecycle_factory=CropLifecycle, sample_endpoint=None,
                 native_rhs: bool = True, trace_sink=None):
        if feedback_mode not in ('full', 'endpoint'):
            raise ValueError('invalid feedback mode')
        validate_policy(contract, fallback_policy)
        if sample_endpoint is None:
            from scripts.check_v22_online_weather import record_at_endpoint
            sample_endpoint = record_at_endpoint
        self.contract = contract
        self.weather = weather
        self.origin_utc = origin_utc
        self._sample_endpoint = sample_endpoint
        self.units = tuple(str(i) for i in range(contract['facility']['compartments']))
        if self.units != ('0', '1', '2', '3'):
            raise ValueError('four units required by this executor')
        self.tick_seconds = contract['controller']['tick_seconds']
        self.deadline_seconds = contract['budget']['campaign_days'] * 86400
        self.feedback_mode = feedback_mode
        self._fallback_policy = copy.deepcopy(fallback_policy)
        self._recommendation = None
        self._recommendation_fallback = False
        self._tool_calls = 0
        self._decision_calls = 0
        self._starts = 0
        self._failed = None
        self._trace_sink = trace_sink
        self._trace_sha256 = hashlib.sha256()
        self._trace_ticks = 0
        self.clock = 0
        self._start_times = {}
        self._policies = {}
        self._unsafe_seconds = {unit: 0 for unit in self.units}
        self.event_log = []
        observations = contract['observations']
        self._controller = PackedOnlineObservations(observations['controller_channels'])
        self._public = PackedOnlineObservations(observations['public_channels'])
        self._full = FeedbackView('full', self._public, self.units)
        self._endpoint = FeedbackView('endpoint', self._public, self.units)
        self._lifecycles = {}
        self._ledgers = {}
        for unit in self.units:
            self._lifecycles[unit] = lifecycle_factory(
                contract, source, start=0, cached_solver=True, native_rhs=native_rhs,
                weather=weather, weather_origin_utc=origin_utc,
                soil_boundary_c=20., array_output=True, initially_empty=True)
            self._ledgers[unit] = ResourceLedger(contract, 0)
        for unit in self.units:
            self._sample_endpoint(self._lifecycles[unit].engine, weather, origin_utc,
                                  self._controller, self._public, unit, self._ledgers[unit])
        self.event_log.append({'event': 'campaign_open', 'clock': 0})

    def _view(self):
        return self._full if self.feedback_mode == 'full' else self._endpoint

    def _unit(self, value):
        if type(value) is not int or str(value) not in self.units:
            raise ValueError('unknown compartment')
        return str(value)

    def _start(self, unit: str, policy: dict) -> dict:
        validate_policy(self.contract, policy)
        if self.clock > self.contract['budget']['latest_start_day'] * 86400:
            raise ValueError('latest start passed')
        if self._starts >= self.contract['budget']['max_starts']:
            raise ValueError('start budget exhausted')
        life = self._lifecycles[unit]
        if life.mode != 'empty' or self.clock < life.ready_at:
            raise ValueError('compartment not ready to plant')
        life.replant()
        ledger = self._ledgers[unit]
        ledger.record_event('plant', self.clock)
        for view in (self._full, self._endpoint):
            view.executor_set_status(unit, 'active', 'start', ledger=ledger)
        self._policies[unit] = copy.deepcopy(policy)
        self._start_times[unit] = self.clock
        self._unsafe_seconds[unit] = 0
        self._starts += 1
        self.event_log.append({'event': 'start', 'unit': unit, 'clock': self.clock,
                               'run_index': self._view().operational_status(unit)['run_index'],
                               'policy': copy.deepcopy(policy)})
        return {'event': 'start', 'unit': unit, 'clock': self.clock,
                'run_index': self._view().operational_status(unit)['run_index']}

    def _stop(self, unit: str, reason: str) -> dict:
        life = self._lifecycles[unit]
        if life.mode != 'active':
            raise ValueError('no active crop to stop')
        life.stop()
        ledger = self._ledgers[unit]
        ledger.record_event('stop', self.clock)
        for view in (self._full, self._endpoint):
            view.executor_set_status(unit, 'cleanup', reason)
        self._endpoint.executor_release_final(unit, reason, ledger)
        self._full.executor_release_final(unit, reason, ledger)
        self._policies.pop(unit)
        self._start_times.pop(unit)
        self.event_log.append({'event': reason, 'unit': unit, 'clock': self.clock})
        return {'event': reason, 'unit': unit, 'clock': self.clock,
                'pending_cleanup_until': life.ready_at}

    def _refresh_idle(self) -> None:
        for unit, life in self._lifecycles.items():
            if (life.mode == 'empty' and self.clock >= life.ready_at
                    and self._view().operational_status(unit)['phase'] == 'cleanup'):
                for view in (self._full, self._endpoint):
                    view.executor_set_status(unit, 'idle')
                self.event_log.append({'event': 'cleanup_complete', 'unit': unit,
                                       'clock': self.clock})

    def _step(self) -> None:
        start = self.clock
        if any(life.engine.clock != start for life in self._lifecycles.values()):
            raise AssertionError('compartment clocks diverged')
        requested = {}
        decisions = {}
        for unit, life in self._lifecycles.items():
            phase = ('active' if life.mode == 'active' else
                     'cleanup' if start < life.ready_at else 'idle')
            requested[unit], decision = commands_from_observations(
                self.contract, self._controller, unit, phase=phase,
                policy=self._policies.get(unit))
            decisions[unit] = {'phase': phase, 'sensor_records': decision['sensor_records']}
            if any(record['available_at'] > start for record in decision['sensor_records'].values()):
                raise AssertionError('future controller record')
        realised = realise_independent_commands(self.contract, requested)
        if realised != requested:
            raise AssertionError('adequate supply changed command')
        increments = {}
        for unit, life in self._lifecycles.items():
            try:
                state = life.step(realised[unit], start + self.tick_seconds)
                if any(not math.isfinite(value) for value in state.values()):
                    raise FloatingPointError('nonfinite physical state')
                increments[unit] = self._ledgers[unit].add_segment(
                    life.engine.model.full_sol, start, start + self.tick_seconds,
                    decisions[unit]['phase'])
            except Exception as exc:
                self._failed = {'unit': unit, 'clock': start,
                                'error': type(exc).__name__ + ': ' + str(exc)}
                self.event_log.append({'event': 'infrastructure_failure', **self._failed})
                raise
        for unit, life in self._lifecycles.items():
            self._sample_endpoint(life.engine, self.weather, self.origin_utc,
                                  self._controller, self._public, unit, self._ledgers[unit])
        self.clock += self.tick_seconds
        if self._trace_sink is not None:
            frame = {'tick': self._trace_ticks, 'start': start, 'end': self.clock,
                     'units': {unit: {'phase': decisions[unit]['phase'],
                                     'controller_records': decisions[unit]['sensor_records'],
                                     'requested': requested[unit], 'realised': realised[unit],
                                     'ledger_increment': increments[unit],
                                     'public_endpoint': {name: self._public.latest(unit, name)
                                                         for name in self.contract['observations']['public_channels']},
                                     'model_input_rows': len(self._lifecycles[unit].engine.model.input_data)}
                               for unit in self.units}}
            encoded = json.dumps(frame, separators=(',', ':'), allow_nan=False) + '\n'
            self._trace_sha256.update(encoded.encode())
            self._trace_sink.write(encoded)
        self._trace_ticks += 1
        for unit, life in self._lifecycles.items():
            if life.mode != 'active':
                continue
            temp = self._public.latest(unit, 'air_temperature_c')['value']
            self._unsafe_seconds[unit] = (self._unsafe_seconds[unit] + self.tick_seconds
                                          if temp < 5 or temp > 40 else 0)
            if self._unsafe_seconds[unit] >= 3600:
                self._stop(unit, 'safety_stop')
            elif self.clock - self._start_times[unit] >= self.contract['budget']['crop_days'] * 86400:
                self._stop(unit, 'normal_completion')
        self._refresh_idle()
        if self.clock == self.deadline_seconds:
            for unit in self.units:
                if self._lifecycles[unit].mode == 'active':
                    self._stop(unit, 'stop')
            self._refresh_idle()

    def _advance(self, target: int) -> dict:
        if target < self.clock:
            raise ValueError('cannot rewind')
        while self.clock < target:
            self._step()
        return {'event': 'advance', 'clock': self.clock,
                'status': [self._view().operational_status(unit) for unit in self.units]}

    def _observe(self, unit: str, action: dict) -> dict:
        view = self._view()
        payload = {'event': 'observe', 'clock': self.clock,
                   'status': view.operational_status(unit)}
        if self.feedback_mode == 'endpoint':
            if any(key in action for key in ('variable', 'start_day', 'end_day')):
                raise PermissionError('endpoint cannot request science history')
            payload['final_aggregate'] = view.final_aggregate(unit)
        else:
            variable = action.get('variable')
            if variable not in self.contract['observations']['public_channels']:
                raise ValueError('unknown public variable')
            start = _seconds(action.get('start_day', 0), self.contract['budget']['campaign_days'], self.tick_seconds)
            end = _seconds(action.get('end_day', self.clock / 86400),
                           self.contract['budget']['campaign_days'], self.tick_seconds)
            records = view.history(unit, variable, start=start, end=end)
            if len(records) > 512:
                raise ValueError('observation window exceeds 512 records; request a smaller window')
            payload['records'] = records
        self.event_log.append({'event': 'observe', 'unit': unit, 'clock': self.clock,
                               'mode': self.feedback_mode})
        return payload

    def _recommend(self, policy) -> dict:
        if self.clock != self.deadline_seconds:
            raise ValueError('recommendation is due at campaign deadline')
        try:
            validate_policy(self.contract, policy)
        except ValueError:
            self._recommendation = copy.deepcopy(self._fallback_policy)
            self._recommendation_fallback = True
        else:
            self._recommendation = copy.deepcopy(policy)
            self._recommendation_fallback = False
        self.event_log.append({'event': 'recommend', 'clock': self.clock,
                               'fallback': self._recommendation_fallback})
        return {'event': 'recommend', 'clock': self.clock,
                'fallback': self._recommendation_fallback}

    def dispatch(self, action: dict) -> dict:
        """Return only JSON-serializable public fields, never private objects."""
        if self._failed is not None:
            raise RuntimeError('campaign paused after infrastructure failure')
        if not isinstance(action, dict) or not isinstance(action.get('action'), str):
            raise ValueError('invalid action payload')
        if self._tool_calls >= self.contract['budget']['max_tool_calls']:
            raise ValueError('tool-call budget exhausted')
        kind = action['action']
        if kind not in self.contract['events']['allowed']:
            raise ValueError('unsupported action')
        self._tool_calls += 1
        if kind in ('start', 'stop', 'observe', 'recommend'):
            if self._decision_calls >= self.contract['budget']['max_decision_calls']:
                raise ValueError('decision-call budget exhausted')
            self._decision_calls += 1
        if kind == 'start':
            if set(action) != {'action', 'unit', 'policy'}:
                raise ValueError('invalid start fields')
            return self._start(self._unit(action['unit']), action['policy'])
        if kind == 'stop':
            if set(action) != {'action', 'unit'}:
                raise ValueError('invalid stop fields')
            return self._stop(self._unit(action['unit']), 'stop')
        if kind == 'observe':
            if not set(action) <= {'action', 'unit', 'variable', 'start_day', 'end_day'} or 'unit' not in action:
                raise ValueError('invalid observation fields')
            return self._observe(self._unit(action['unit']), action)
        if kind == 'advance':
            if set(action) != {'action', 'day'}:
                raise ValueError('invalid advance fields')
            return self._advance(_seconds(action['day'], self.contract['budget']['campaign_days'], self.tick_seconds))
        if set(action) != {'action', 'policy'}:
            raise ValueError('invalid recommendation fields')
        return self._recommend(action['policy'])

    def settlement(self) -> dict:
        if self.clock != self.deadline_seconds or self._failed is not None:
            raise ValueError('campaign has not reached a valid deadline')
        return {'clock': self.clock,
                'ledger_by_unit': {unit: ledger.summary() for unit, ledger in self._ledgers.items()},
                'recommendation': copy.deepcopy(self._recommendation or self._fallback_policy),
                'recommendation_fallback': self._recommendation is None or self._recommendation_fallback,
                'starts': self._starts,
                'tool_calls': self._tool_calls,
                'decision_calls': self._decision_calls,
                'trace_ticks': self._trace_ticks,
                'trace_sha256': self._trace_sha256.hexdigest() if self._trace_sink is not None else None,
                'trace_complete': self._trace_sink is not None}
