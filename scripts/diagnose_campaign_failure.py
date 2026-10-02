#!/usr/bin/env python3
"""Replay a failed LLM campaign branch from its outbound audit and inspect the failing step (private).

The campaign actions the model sent (labels in --labels, in audit order) are
dispatched again to an executor built from the branch's private site spec.
Rejected actions are rejected again; the simulation is deterministic, so the
replay reaches the same state. On the failing step the diagnostic records the
compartment, the integrator's last time points and the crop states with the
largest relative rates, plus the running minimum of the carbohydrate buffer.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.executor_server import build_executor  # noqa: E402
from slowlab.llm_agent import FormatError, parse_reply  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site-spec', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--labels', default='shared_day_0,full')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(args.site_spec.read_text())
    spec['trace'] = None
    spec['settlement_out'] = str(args.out.with_suffix('.settlement.json'))
    spec['failure_out'] = str(args.out.with_suffix('.failure.json'))
    contract, x, _ = build_executor(spec)
    for unit, life in x._lifecycles.items():
        solver = getattr(life.engines['active'], '_cached_solver', None)
        if solver is not None:
            solver.track_state_minima = ['cBuf']
    labels = args.labels.split(',')
    actions = []
    for line in args.audit.read_text().splitlines():
        rec = json.loads(line)
        if rec['label'] not in labels:
            continue
        try:
            reply = parse_reply(rec.get('reply') or '')
        except FormatError:
            continue
        if reply['type'] == 'campaign':
            actions.append(reply['action'])
    out = {'actions_in_audit': len(actions), 'dispatched': 0, 'rejected': 0}
    for action in actions:
        try:
            x.dispatch(action)
            out['dispatched'] += 1
        except (ValueError, PermissionError):
            if x._failed is not None:
                break
            out['rejected'] += 1
        except Exception:
            break
        if x._failed is not None:
            break
    out['failed'] = x._failed
    if x._failed:
        unit = x._failed['unit']
        life = x._lifecycles[unit]
        model = life.engine.model
        sol = getattr(model, 'states_sol', None)
        names = list(model.states)
        out['mode'] = life.mode
        if sol is not None:
            t = np.asarray(sol.t, dtype=float)
            out['solver'] = {'success': bool(sol.success), 'points': int(len(t)), 't_first': float(t[0]), 't_last': float(t[-1]),
                             'duplicate_times': int(np.sum(np.diff(t) <= 0)), 'smallest_step_s': float(np.min(np.diff(t))) if len(t) > 1 else None,
                             'message': str(sol.message)}
            y = np.asarray(sol.y, dtype=float)
            last, prev = y[:, -1], y[:, max(0, y.shape[1] - 2)]
            dt = max(t[-1] - t[max(0, len(t) - 2)], 1e-30)
            rate = np.abs(last - prev) / dt / np.maximum(np.abs(last), 1e-9)
            order = np.argsort(-rate)[:10]
            out['fastest_states'] = [{'state': names[i], 'value': float(last[i]), 'relative_rate_per_s': float(rate[i])} for i in order]
            out['crop_states'] = {k: float(last[names.index(k)]) for k in ('cBuf', 'cLeaf', 'cStem', 'cFruit', 'tCanSum', 'co2Air', 'tAir') if k in names}
        solver_obj = getattr(life.engines['active'], '_cached_solver', None)
        out['cBuf_minimum_since_start'] = (getattr(solver_obj, 'state_minima', {}) or {}).get('cBuf')
        out['policy'] = x._policies.get(unit).as_dict() if unit in x._policies else None
        out['unit_parameters'] = x.unit_parameters.get(unit)
    args.out.write_text(json.dumps(out, indent=2, default=str) + '\n')
    print(json.dumps({k: out.get(k) for k in ('dispatched', 'rejected', 'failed', 'solver', 'crop_states', 'cBuf_minimum_since_start')}, indent=1, default=str))
    print(json.dumps(out.get('fastest_states', [])[:6], indent=1))


if __name__ == '__main__':
    main()
