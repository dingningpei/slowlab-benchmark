"""E3 history packets and the GP diagnostic Reader (RESEARCH_PLAN decision 2026-10-01).

A history packet is the public evidence a method held when a Full campaign
ended, before its recommendation, in a fixed format: every crop's policy,
planting day, status and (for closed crops) released totals and contribution
margin at the public prices; and every channel the method read, summarised per
compartment and day (count, mean, minimum, maximum). It is a deterministic
function of the public task view and transcript and contains nothing private.

Selection rule (decisions 2026-10-01 and 2026-10-03): from the repeat-0 Full
campaigns of every test site and method, a stratified random sample of 64
packets (16 per method, committed seed). Every packet is read by every Reader:
the method's own recommendation, the GP Reader below, and each language model
as a one-shot Reader.

GP Reader (gp-reader-v2, decision 2026-10-03): the main baseline's final step
(configs/prior_bo_v1.json): the frozen kernel, and the tried policy with the
highest posterior mean of the scoring rule; untried policies are never chosen.
"""
from __future__ import annotations

import math

import numpy as np

from .tools import PublicHistory, contribution_margin, scoring_days, season_features

PACKET_FORMAT = 'history-packet-v1'  # shown to Readers: no benchmark name
GP_READER_VERSION = 'gp-reader-v2'


def build_packet(task: dict, transcript: list) -> dict:
    history = PublicHistory(transcript)
    crops = []
    for (unit, run_index) in sorted(history.runs):
        run = history.runs[(unit, run_index)]
        row = {'unit': unit, 'crop': run_index, 'policy': run.get('policy'), 'start_day': run.get('start_day'),
               'status': run.get('status'), 'reason': run.get('reason')}
        if run.get('accrued') is not None:
            row['accrued_per_m2'] = run['accrued']
            row['contribution_margin_eur_m2'] = contribution_margin(run['accrued'], run['event_cost_eur_m2'],
                                                                    task['economics'])
        crops.append(row)
    series = []
    for (unit, variable) in sorted(history.records):
        days: dict[int, list[float]] = {}
        for t, v in sorted(history.records[(unit, variable)].items()):
            days.setdefault(int(t // 86400), []).append(v)
        series.append({'unit': unit, 'variable': variable,
                       'days': [{'day': d, 'n': len(v), 'mean': float(np.mean(v)), 'min': min(v), 'max': max(v)}
                                for d, v in sorted(days.items())]})
    return {'format': PACKET_FORMAT, 'task_scoring_days': scoring_days(task), 'crops': crops, 'series': series}


def gp_reader(task: dict, packet: dict, config: dict) -> dict:
    """The main baseline's recommendation step on a packet. ``config``: a loaded prior BO config
    (kernel, noise ratio) and the anchor policy under ``anchor_policy`` (used only without crops)."""
    from .prior_bo import FrozenGP
    fields = task['policy']['fields']
    names = list(fields)

    def scale(p):
        return [(p[k] - fields[k]['min']) / (fields[k]['max'] - fields[k]['min']) for k in names]
    done = [c for c in packet['crops'] if c.get('reason') == 'normal_completion' and c.get('policy')
            and c.get('contribution_margin_eur_m2') is not None]
    if not done:
        return {'reader': GP_READER_VERSION, 'policy': dict(config['anchor_policy']), 'basis': 'anchor_no_completed_crops'}
    x = np.array([scale(c['policy']) + season_features(c['start_day']) for c in done])
    y = np.array([c['contribution_margin_eur_m2'] for c in done])
    model = FrozenGP(x, y, config['hyperparameters']['kernel'], config['hyperparameters']['noise_ratio'])
    tried = []
    for c in done:
        if c['policy'] not in tried:
            tried.append(c['policy'])
    mean, sd, per_day = model.predict_average(np.array([scale(p) for p in tried]), packet['task_scoring_days'])
    j = int(np.argmax(mean))
    return {'reader': GP_READER_VERSION, 'policy': dict(tried[j]), 'basis': 'max_posterior_scored_mean_tried',
            'predicted_mean': float(mean[j]), 'predicted_sd': float(sd[j]), 'crops_used': len(done)}
