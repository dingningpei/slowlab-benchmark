"""Isolated evaluation of a recommended policy (research plan decision, 2026-09-30).

One evaluation weather year is one 365-day run of the site's facility: half the
compartments are planted on calendar day 0 (1 January), the other half on
calendar day 182 (2 July), all under the recommended policy, each running the
contract's horizon. Every compartment carries the site's own parameter values
with freshly drawn unobservable compartment differences, so averaging over
compartments also averages over those differences. The year's score is the
mean over plantings of the mean crop contribution margin (EUR per m2 floor),
computed with the same function and public prices the agent's tools use.

The evaluator runs in its own process from private inputs only. Nothing it
produces is returned to a method during a campaign.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .agent_protocol import public_task_view
from .campaign_executor import CampaignExecutor
from .site_parameters import site_contract
from .tools import contribution_margin

FORMAT = 'slowlab-evaluation-v1'


def planting_units(contract: dict) -> list[tuple[float, list[int]]]:
    days = contract['evaluation']['deployment']['plantings_calendar_day']
    count = contract['facility']['compartments']
    if count % len(days):
        raise ValueError('compartments must split evenly across plantings')
    per = count // len(days)
    return [(float(day), list(range(i * per, (i + 1) * per))) for i, day in enumerate(sorted(days))]


def evaluate_year(contract: dict, policy: dict, *, site: dict | None, unit_parameters: dict | None,
                  weather, source, year: int, soil_boundary_c=None, sensor_noise=None,
                  executor_kwargs: dict | None = None) -> dict:
    """Score ``policy`` in one evaluation weather year; returns a private record."""
    contract = site_contract(contract, site or {})
    horizon = contract['evaluation']['horizon_days']
    if horizon != contract['budget']['crop_days']:
        raise ValueError('evaluation horizon must equal the crop length')
    plan = planting_units(contract)
    last = plan[-1][0] + horizon
    if last > contract['budget']['campaign_days'] or plan[-1][0] > contract['budget']['latest_start_day']:
        raise ValueError('evaluation plantings do not fit the calendar')
    x = CampaignExecutor(contract, source, weather, feedback_mode='full', fallback_policy=policy,
                         origin_utc=datetime(int(year) - 1, 12, 31, 23, tzinfo=timezone.utc),
                         soil_boundary_c=soil_boundary_c, unit_parameters=unit_parameters,
                         sensor_noise=sensor_noise, **(executor_kwargs or {}))
    for day, units in plan:
        if day > 0:
            x.dispatch({'action': 'advance', 'day': day})
        for unit in units:
            x.dispatch({'action': 'start', 'unit': unit, 'policy': policy})
    x.dispatch({'action': 'advance', 'day': last})
    economics = public_task_view(contract, 'full')['economics']
    crops = {record['unit']: record for record in x.private_crop_totals()}
    plantings = []
    for day, units in plan:
        rows = []
        for unit in units:
            record = crops.get(str(unit))
            if record is None:
                raise RuntimeError(f'evaluation crop in compartment {unit} did not close')
            rows.append({'unit': unit, 'reason': record['reason'], 'accrued': record['accrued'],
                         'event_cost_eur_m2': record['event_cost_eur_m2'],
                         'margin_eur_m2': contribution_margin(record['accrued'], record['event_cost_eur_m2'], economics)})
        plantings.append({'planting_day': day, 'crops': rows,
                          'mean_margin_eur_m2': sum(r['margin_eur_m2'] for r in rows) / len(rows)})
    return {'format': FORMAT, 'year': int(year), 'policy': policy,
            'score_eur_m2': sum(p['mean_margin_eur_m2'] for p in plantings) / len(plantings),
            'plantings': plantings, 'unit_parameters': x.unit_parameters,
            'soil_boundary_c': x.soil_boundary_c,
            'sensor_noise': sensor_noise.describe() if sensor_noise is not None else None,
            'all_normal_completion': all(r['reason'] == 'normal_completion' for p in plantings for r in p['crops'])}


def evaluate_policies_year(contract: dict, policies: list[dict], *, site: dict | None, unit_parameters: dict | None,
                           weather, source, year: int, soil_boundary_c=None, sensor_noise=None,
                           executor_kwargs: dict | None = None) -> dict:
    """Score several policies in one evaluation run (reference search).

    Policy j takes the j-th compartment of every planting, so each policy gets
    one January and one July crop when two policies share the run. Scores are
    noisier than evaluate_year's (one compartment per planting) and are used
    only to rank search candidates.
    """
    contract = site_contract(contract, site or {})
    plan = planting_units(contract)
    per = len(plan[0][1])
    if not 1 <= len(policies) <= per or per % len(policies):
        raise ValueError('the number of policies must divide the compartments per planting')
    horizon = contract['evaluation']['horizon_days']
    last = plan[-1][0] + horizon
    x = CampaignExecutor(contract, source, weather, feedback_mode='full', fallback_policy=policies[0],
                         origin_utc=datetime(int(year) - 1, 12, 31, 23, tzinfo=timezone.utc),
                         soil_boundary_c=soil_boundary_c, unit_parameters=unit_parameters,
                         sensor_noise=sensor_noise, **(executor_kwargs or {}))
    owner = {}
    for day, units in plan:
        if day > 0:
            x.dispatch({'action': 'advance', 'day': day})
        for k, unit in enumerate(units):
            j = k % len(policies)
            owner[unit] = j
            x.dispatch({'action': 'start', 'unit': unit, 'policy': policies[j]})
    x.dispatch({'action': 'advance', 'day': last})
    economics = public_task_view(contract, 'full')['economics']
    crops = {record['unit']: record for record in x.private_crop_totals()}
    out = []
    for j, policy in enumerate(policies):
        by_planting = []
        for day, units in plan:
            margins = [contribution_margin(crops[str(u)]['accrued'], crops[str(u)]['event_cost_eur_m2'], economics)
                       for u in units if owner[u] == j]
            by_planting.append(sum(margins) / len(margins))
        out.append({'policy': policy, 'score_eur_m2': sum(by_planting) / len(by_planting),
                    'by_planting_eur_m2': by_planting})
    return {'format': FORMAT + '-multi', 'year': int(year), 'results': out}
