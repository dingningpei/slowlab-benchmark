#!/usr/bin/env python3
"""Pilot accounting and process-feedback analysis from campaign records (public material only).

Accounting: per campaign calls, tokens (prompt, completion, reasoning, DeepSeek cache hits), billed
cost, characters and errors per branch, wall time. Process feedback: for each branched LLM campaign,
compare the Full and Endpoint branches after the shared day-0 design: later plantings (day, unit,
policy), stops, the in-season reads and predictor calls the Full branch made before each later
planting, and the distance between the two recommendations (policies scaled to [0, 1]).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from slowlab.spend import call_cost  # noqa: E402


def scaled(fields, p):
    return [(p[k] - s['min']) / (s['max'] - s['min']) for k, s in fields.items()]


def distance(fields, a, b):
    if a is None or b is None:
        return None
    return math.dist(scaled(fields, a), scaled(fields, b)) / math.sqrt(len(fields))


def branch_events(transcript):
    starts, stops, reads = [], [], []
    for e in transcript:
        if not e.get('ok'):
            continue
        req, res = e['request'], e['result']
        day = res.get('clock', 0) / 86400
        if req['action'] == 'start':
            starts.append({'day': day, 'unit': req['unit'], 'policy': req['policy']})
        elif req['action'] == 'stop':
            stops.append({'day': day, 'unit': req['unit']})
        elif req['action'] == 'observe' and 'variable' in req:
            reads.append({'day': day, 'variable': req['variable'], 'resolution': req.get('resolution', 'raw'),
                          'records': len(res.get('records') or []) + len(res.get('daily') or [])})
    return starts, stops, reads


def tool_calls(audit_path, label):
    out = []
    if not audit_path.exists():
        return out
    for line in audit_path.read_text().splitlines():
        rec = json.loads(line)
        if rec['label'] != label:
            continue
        reply = rec.get('reply') or ''
        m = re.search(r'"name"\s*:\s*"([a-z_]+)"', reply)
        if '"tool"' in reply and m:
            try:
                day = json.loads(rec['messages'][-1]['content'])['state']['day']
            except Exception:
                day = None
            out.append({'name': m.group(1), 'day': day})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--models', default='configs/pilot_models_v3.json')
    ap.add_argument('--accounting', type=Path, required=True)
    ap.add_argument('--feedback', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads((ROOT / args.models).read_text())
    prices = {m['name']: m['price'] for m in cfg['models']}
    fields = json.loads((ROOT / 'configs/task_contract_v8.json').read_text())['policy']['fields']
    rows, fb, bo = [], [], []
    for f in sorted((args.run_dir / 'campaigns').glob('site*.json')):
        if f.name.endswith('.audit.jsonl'):
            continue
        r = json.loads(f.read_text()); job = f.stem
        m = re.match(r'site(\d+)_(.+?)_r(\d)$', job)
        if not m:
            bo.append({'job': job, 'status': r['status'], 'minutes': round(r.get('elapsed_seconds', 0) / 60, 1),
                       'completed_crops': (r.get('bo') or {}).get('completed_crops')})
            continue
        site, model, rep = int(m[1]), m[2], int(m[3])
        recs = r.get('provider_call_records') or []
        use = [x.get('usage') or {} for x in recs]
        row = {'job': job, 'site': site, 'model': model, 'repeat': rep, 'status': r['status'],
               'minutes': round(r.get('elapsed_seconds', 0) / 60, 1), 'calls': len(recs),
               'prompt_tokens': sum(u.get('prompt_tokens') or 0 for u in use),
               'completion_tokens': sum(u.get('completion_tokens') or 0 for u in use),
               'reasoning_tokens': sum((u.get('completion_tokens_details') or {}).get('reasoning_tokens') or 0 for u in use),
               'cache_hit_tokens': sum(u.get('prompt_cache_hit_tokens') or 0 for u in use),
               'usd': sum(call_cost(x, prices[model]) for x in recs)}
        if r['status'] == 'completed':
            for b in ('full', 'endpoint'):
                c = r['branches'][b]['summary']['counts']
                row[b] = {k: c[k] for k in ('llm_calls', 'input_chars', 'output_chars', 'format_errors', 'invalid_actions',
                                           'tool_calls', 'tool_errors', 'observe_calls', 'records_received',
                                           'daily_rows_received', 'notes_updates', 'context_exhausted', 'recommended')}
            full_s, full_stop, full_reads = branch_events(r['branches']['full']['public_transcript'])
            end_s, end_stop, _ = branch_events(r['branches']['endpoint']['public_transcript'])
            audit = args.run_dir / 'campaigns' / f'{job}.audit.jsonl'
            full_tools, end_tools = tool_calls(audit, 'full'), tool_calls(audit, 'endpoint')
            later_full = [s for s in full_s if s['day'] > 1e-9]
            later_end = [s for s in end_s if s['day'] > 1e-9]
            first_later = min((s['day'] for s in later_full), default=None)
            full_rec = json.loads(Path(r['branches']['full']['settlement']).read_text())['recommendation']
            end_rec = json.loads(Path(r['branches']['endpoint']['settlement']).read_text())['recommendation']
            fb.append({'job': job, 'model': model, 'site': site, 'repeat': rep,
                       'later_starts_full': [(round(s['day'], 2), s['unit']) for s in later_full],
                       'later_starts_endpoint': [(round(s['day'], 2), s['unit']) for s in later_end],
                       'stops_full': len(full_stop), 'stops_endpoint': len(end_stop),
                       'full_reads_total': len(full_reads), 'full_reads_before_first_later_start':
                           sum(1 for x in full_reads if first_later is None or x['day'] < first_later),
                       'full_read_days': sorted({round(x['day']) for x in full_reads}),
                       'full_predictor_calls': sum(1 for t in full_tools if t['name'] == 'predict_crop_outcome'),
                       'endpoint_predictor_calls': sum(1 for t in end_tools if t['name'] == 'predict_crop_outcome'),
                       'second_wave_policy_distance': (st.mean(distance(fields, a['policy'], b['policy'])
                                                               for a, b in zip(sorted(later_full, key=lambda s: s['unit']),
                                                                               sorted(later_end, key=lambda s: s['unit'])))
                                                       if later_full and later_end else None),
                       'same_later_schedule': [(round(s['day']), s['unit']) for s in later_full] ==
                                              [(round(s['day']), s['unit']) for s in later_end],
                       'recommendation_distance': distance(fields, full_rec, end_rec),
                       'identical_recommendation': full_rec == end_rec})
        rows.append(row)
    by_model = defaultdict(list)
    for r in rows:
        by_model[r['model']].append(r)
    summary = {}
    for model, rs in by_model.items():
        ok = [r for r in rs if r['status'] == 'completed']
        summary[model] = {'campaigns': len(rs), 'completed': len(ok),
                          **{f'mean_{k}': st.mean(r[k] for r in ok) for k in ('usd', 'calls', 'minutes', 'prompt_tokens',
                                                                                'completion_tokens', 'reasoning_tokens', 'cache_hit_tokens')},
                          'total_usd': sum(r['usd'] for r in rs),
                          **{f'mean_{b}_{k}': st.mean(r[b][k] for r in ok) for b in ('full', 'endpoint')
                             for k in ('input_chars', 'format_errors', 'invalid_actions', 'tool_calls', 'observe_calls')}}
    args.accounting.write_text(json.dumps({'by_model': summary, 'bo': bo, 'campaigns': rows}, indent=2) + '\n')
    fsum = {}
    for model in sorted({x['model'] for x in fb}):
        xs = [x for x in fb if x['model'] == model]
        dists = [x['recommendation_distance'] for x in xs if x['recommendation_distance'] is not None]
        fsum[model] = {'campaigns': len(xs), 'any_stop': sum(1 for x in xs if x['stops_full'] or x['stops_endpoint']),
                       'same_later_schedule': sum(x['same_later_schedule'] for x in xs),
                       'identical_recommendation': sum(x['identical_recommendation'] for x in xs),
                       'mean_recommendation_distance': st.mean(dists) if dists else None,
                       'mean_full_reads_before_first_later_start': st.mean(x['full_reads_before_first_later_start'] for x in xs),
                       'mean_full_predictor_calls': st.mean(x['full_predictor_calls'] for x in xs),
                       'campaigns_with_non_two_wave_schedule': sum(1 for x in xs if any(abs(d - 182) > 3 for d, _ in x['later_starts_full']))}
    args.feedback.write_text(json.dumps({'summary_by_model': fsum, 'campaigns': fb}, indent=2) + '\n')
    print(json.dumps({'accounting': {m: {k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()} for m, s in summary.items()},
                      'feedback': fsum}, indent=1))


if __name__ == '__main__':
    main()
