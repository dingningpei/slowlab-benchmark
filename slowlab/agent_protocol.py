"""Wire protocol between an agent-side session and the private executor server.

One JSON object per line. Requests: ``{"protocol", "id", "op", ...}`` with op
``hello`` | ``dispatch`` (carrying ``action``) | ``close``. Responses:
``{"protocol", "id", "ok": true, "result": ...}`` or
``{"protocol", "id", "ok": false, "error": {"kind", "message"}}``.

Everything in a response may reach an agent, so the protocol name and the
public task view avoid project, model, dataset and simulator identity words
(checked against the simulation-blinding firewall in tests).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

PROTOCOL = 'campaign-api-v1'
OPS = ('hello', 'dispatch', 'close')
ERROR_KINDS = ('invalid_action', 'infrastructure_failure', 'protocol_error')


def canonical(obj) -> str:
    """Deterministic single-line JSON; rejects NaN and infinity."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()


def parse(line: str) -> dict:
    def reject(value):
        raise ValueError(f'non-finite JSON constant {value}')
    obj = json.loads(line, parse_constant=reject)
    if not isinstance(obj, dict):
        raise ValueError('protocol message must be a JSON object')
    return obj


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


ACTION_SCHEMAS = {
    'start': {'unit': 'integer compartment index', 'policy': 'object with exactly the policy fields'},
    'observe': {'unit': 'integer compartment index',
                'run_index': 'optional: which crop of this compartment (1 = first); default the latest closed one',
                'variable': 'full feedback only, optional: a public channel name',
                'start_day': 'full feedback only: optional, day number at a 300-second boundary',
                'end_day': 'full feedback only: optional, day number at a 300-second boundary, not in the future'},
    'advance': {'day': 'target day number at a 300-second boundary, not earlier than now'},
    'stop': {'unit': 'integer compartment index'},
    'recommend': {'policy': 'object with exactly the policy fields; allowed only at the final day'},
}


def public_task_view(contract: dict, feedback_mode: str | None) -> dict:
    """The task description an agent may see: an explicit allowlist of contract fields.

    ``feedback_mode=None`` gives the condition-free view used for the shared
    day-0 design before a campaign branches into Full and Endpoint.
    """
    if feedback_mode not in ('full', 'endpoint', None):
        raise ValueError('invalid feedback mode')
    facility, budget = contract['facility'], contract['budget']
    observations, economics = contract['observations'], contract['economics']
    fields = {name: {'min': spec['min'], 'max': spec['max'], 'unit': spec['unit']}
              for name, spec in contract['policy']['fields'].items()}
    for spec in fields.values():
        if not (_finite(spec['min']) and _finite(spec['max'])):
            raise ValueError('policy bounds must be finite')
    view = {
        'protocol': PROTOCOL,
        'setting': ('A greenhouse experimentation campaign. You manage independent climate-controlled '
                    'compartments, start and stop tomato crops under management policies, read the '
                    'records your feedback condition allows, and finally recommend one policy.'),
        'compartments': list(range(facility['compartments'])),
        'compartment_floor_area_m2': facility['floor_area_m2'],
        'calendar': {'campaign_days': budget['campaign_days'], 'crop_days': budget['crop_days'],
                     'cleanup_days': budget['cleanup_days'], 'latest_start_day': budget['latest_start_day'],
                     'time_step_seconds': contract['controller']['tick_seconds']},
        'budget': {'max_starts': budget['max_starts'], 'max_decision_calls': budget['max_decision_calls'],
                   'max_tool_calls': budget['max_tool_calls'], 'max_context_tokens': budget['max_context_tokens']},
        'policy': {'fields': fields,
                   'constraints': ['night_temperature_c <= day_temperature_c'],
                   'fixed_while_running': True},
        'actions': copy.deepcopy(ACTION_SCHEMAS),
        'rules': [
            'A started crop runs for crop_days and then closes automatically; the compartment is then cleaned for cleanup_days.',
            'stop ends a running crop immediately: the standing crop is discarded with no salvage value, cleaning costs are paid, and only produce already harvested counts.',
            'A crop cannot start after latest_start_day, and no compartment can be reused while running or cleaning.',
            'If indoor air stays below 5 C or above 40 C for one hour, the crop is stopped for safety.',
            'At the final day all running crops are stopped and settled; recommend must be sent then.',
            'Every call counts against max_tool_calls; start, stop, observe and recommend also count against max_decision_calls.',
            'Your recommendation is scored later on independent growing seasons; the score is never shown to you.',
        ],
        'scoring': None,
        'feedback': ({'mode': 'assigned_after_day_0_design',
                      'description': ('Which records you may read while crops grow is assigned after you finish '
                                      'your day-0 design; closed crops always release their accrued totals.')}
                     if feedback_mode is None else
                     {'mode': feedback_mode,
                      'description': ('Every record measured so far can be queried at any time, without advancing time.'
                                      if feedback_mode == 'full' else
                                      'While a crop runs you receive operational status only; each closed crop releases its accrued totals.')}),
        'observations': {'channels': dict(observations['public_channels']),
                         'cadence_seconds': observations['cadence_seconds'],
                         'notes': ['canopy_lai_proxy is an image-based canopy estimate with measurement error.',
                                   'Climate readings carry sensor error; occasional readings are missing.',
                                   'Energy, CO2 and harvest channels are cumulative metered totals per square metre.']},
        'economics': {'unit': 'EUR per m2 of floor',
                      'fruit_price_eur_per_kg_fresh': economics['price_eur_per_kg_fresh_equivalent'],
                      'harvest_handling_eur_per_kg': economics['harvest_handling_eur_per_kg'],
                      'electricity_eur_per_kwh': economics['electricity_eur_per_kwh'],
                      'delivered_heat_eur_per_kwh': economics['delivered_heat_eur_per_kwh'],
                      'co2_eur_per_kg': economics['co2_eur_per_kg'],
                      'planting_eur_per_m2': economics['planting_eur_per_m2'],
                      'cleanup_eur_per_m2': economics['cleanup_eur_per_m2'],
                      'background_service_eur_per_m2_day': economics['background_service_eur_per_m2_day'],
                      'objective': 'contribution margin of the recommended policy over one crop cycle; fixed capital costs excluded'},
    }
    deployment = contract['evaluation'].get('deployment')
    if deployment:
        view['calendar']['campaign_day_0'] = '1 January, 00:00 local standard time'
        view['scoring'] = {'planting_calendar_days': list(deployment['plantings_calendar_day']),
                           'aggregation': deployment['aggregation'],
                           'rule': ('Your recommended policy is scored as the mean contribution margin per m2 of two '
                                    'crops grown under it in new, independent years: one planted on calendar day 0 '
                                    '(1 January) and one planted on calendar day 182 (2 July), each running crop_days. '
                                    'The score is never shown to you.')}
        view['rules'][-1] = view['scoring']['rule']
    return view
