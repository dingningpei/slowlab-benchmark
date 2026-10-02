#!/usr/bin/env python3
"""Connectivity check: one short message per pilot model (approved, < USD 0.01 in total).

Records the exact model and provider that answered, the reasoning mode in force,
reasoning tokens, whether the reply is valid JSON, and the billed cost, which
is booked in the pilot spend ledger.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.providers import openai_compatible  # noqa: E402
from slowlab.spend import call_cost  # noqa: E402

MESSAGE = [{'role': 'system', 'content': 'You answer with JSON only.'},
           {'role': 'user', 'content': 'Reply with exactly this JSON object and nothing else: {"ok": true}'}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='configs/pilot_models_v1.json')
    ap.add_argument('--ledger', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads((ROOT / args.models).read_text())
    results = []
    for m in cfg['models']:
        row = {'name': m['name'], 'requested': m['llm']['model']}
        try:
            complete = openai_compatible(m['llm']['model'], temperature=m['llm'].get('temperature', 0.7), max_tokens=64,
                                         max_attempts=2)
            text = complete(MESSAGE)
            rec = complete.call_records[-1]
            usage = rec.get('usage') or {}
            details = usage.get('completion_tokens_details') or {}
            try:
                parsed = json.loads(text.strip().strip('`').removeprefix('json').strip())
            except json.JSONDecodeError:
                parsed = None
            row.update(status='ok', actual_model=rec.get('actual_model'), provider=rec.get('provider'),
                       finish_reason=rec.get('finish_reason'), reply=text[:120], valid_json=parsed == {'ok': True},
                       reasoning_mode=rec.get('reasoning_mode'), reasoning_tokens=details.get('reasoning_tokens'),
                       reasoning_disabled=json.loads(rec.get('reasoning_mode') or '{}') in (
                           {'reasoning': {'enabled': False}}, {'thinking': {'type': 'disabled'}})
                       and not details.get('reasoning_tokens'),
                       usage=usage, usd=call_cost(rec, m['price']))
            with args.ledger.open('a') as f:
                f.write(json.dumps({'kind': 'connectivity', 'model': m['name'], 'usd': row['usd'], 'calls': 1}) + '\n')
        except Exception as error:  # report every model, then stop the pilot by hand if any failed
            row.update(status='failed', error=f'{type(error).__name__}: {error}'[:300])
        results.append(row)
        print(json.dumps({k: row.get(k) for k in ('name', 'status', 'actual_model', 'provider', 'valid_json',
                                                   'reasoning_disabled', 'reasoning_tokens', 'usd', 'error')}))
    args.out.write_text(json.dumps({'checked_models': results}, indent=2) + '\n')


if __name__ == '__main__':
    main()
