"""A deterministic stand-in for a language model, for dry runs of the whole path.

It reads only the state digest in the final prompt message, like a model would,
and follows a fixed two-wave plan: four spread policies at day 0; when those
crops close, read their totals; four variations of the best one after
cleanup; read the rest at the final day; recommend the best closed crop's
policy. It exists to exercise the harness end to end without spending API
calls; it is not a baseline.
"""
from __future__ import annotations

import json

FIRST_WAVE = [
    {'day_temperature_c': 20.0, 'night_temperature_c': 16.0, 'co2_target_ppm': 600, 'supplemental_light_hours': 6.0,
     'vent_temperature_offset_c': 3.0, 'vent_rh_threshold_pct': 85.0},
    {'day_temperature_c': 23.0, 'night_temperature_c': 18.0, 'co2_target_ppm': 900, 'supplemental_light_hours': 10.0,
     'vent_temperature_offset_c': 4.0, 'vent_rh_threshold_pct': 80.0},
    {'day_temperature_c': 19.0, 'night_temperature_c': 17.0, 'co2_target_ppm': 1000, 'supplemental_light_hours': 2.0,
     'vent_temperature_offset_c': 2.5, 'vent_rh_threshold_pct': 88.0},
    {'day_temperature_c': 24.0, 'night_temperature_c': 15.5, 'co2_target_ppm': 500, 'supplemental_light_hours': 14.0,
     'vent_temperature_offset_c': 5.0, 'vent_rh_threshold_pct': 75.0},
]


def _reply(kind, **body):
    return json.dumps({'type': kind, **body})


def _variations(best, fields):
    out = []
    for i, (field, delta) in enumerate((('day_temperature_c', -1.0), ('co2_target_ppm', 150),
                                        ('supplemental_light_hours', 2.0), ('night_temperature_c', -1.0))):
        policy = dict(best)
        spec = fields[field]
        policy[field] = min(spec['max'], max(spec['min'], policy[field] + delta))
        policy['night_temperature_c'] = min(policy['night_temperature_c'], policy['day_temperature_c'])
        out.append(policy)
    return out


class ScriptedModel:
    def __init__(self, task: dict):
        self.fields = task['policy']['fields']
        self.cal = task['calendar']
        self.call_records = []

    def __call__(self, messages) -> str:
        try:
            final = json.loads(messages[-1]['content'])
        except json.JSONDecodeError:
            return _reply('initial_recommendation', policy=FIRST_WAVE[0], reason='scripted prior')
        state, instruction = final['state'], final['instruction']
        crops = state['crops']
        closed = [c for c in crops if c.get('contribution_margin_eur_m2') is not None]
        best = (max(closed, key=lambda c: c['contribution_margin_eur_m2'])['policy'] if closed
                else FIRST_WAVE[0])
        end = self.cal['campaign_days']
        if instruction.startswith('Day-0 design'):
            if len(crops) < len(FIRST_WAVE):
                return _reply('campaign', action={'action': 'start', 'unit': len(crops), 'policy': FIRST_WAVE[len(crops)]})
            return _reply('design_complete')
        if 'Reply only' in instruction or state['days_left'] <= 1e-9 and not self._unread(crops):
            return _reply('campaign', action={'action': 'recommend', 'policy': best})
        unread = self._unread(crops)
        if unread and state['day'] > 0:
            c = unread[0]
            return _reply('campaign', action={'action': 'observe', 'unit': c['unit'], 'run_index': c['crop']})
        if len(crops) < len(FIRST_WAVE):
            unit = len(crops)
            return _reply('campaign', action={'action': 'start', 'unit': unit, 'policy': FIRST_WAVE[unit]})
        wave_two = self.cal['crop_days'] + self.cal['cleanup_days']
        if len(crops) < 2 * len(FIRST_WAVE) and state['day'] >= wave_two - 1e-9 and wave_two <= self.cal['latest_start_day']:
            unit = len(crops) - len(FIRST_WAVE)
            return _reply('campaign', action={'action': 'start', 'unit': unit,
                                              'policy': _variations(best, self.fields)[unit]},
                          reason='vary the best first-wave policy')
        target = wave_two if state['day'] < wave_two - 1e-9 and wave_two <= self.cal['latest_start_day'] else end
        tick = self.cal['time_step_seconds'] / 86400
        return _reply('campaign', action={'action': 'advance', 'day': round(target / tick) * tick})

    @staticmethod
    def _unread(crops):
        return [c for c in crops if c['status'] == 'closed_totals_not_yet_observed']
