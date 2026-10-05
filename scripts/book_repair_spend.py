#!/usr/bin/env python3
"""Spend of the repair's live model calls (resumed Full branches, every attempt), from their outbound audits.

    book_repair_spend.py --run-dir R --out R/ledger_repair_resume.jsonl

Each audit line of a live call carries the provider record (usage, billed cost when the provider gives it);
replayed calls have none and cost nothing. Attempts that failed are booked too. Rebuilt from scratch on
every call, so it can be rerun safely.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.spend import records_cost  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    models = json.loads((ROOT / 'configs/formal_models_v1.json').read_text())['models']
    by_model = {m['llm']['model']: m for m in models}
    rows, total = [], 0.0
    for audit in sorted(args.run_dir.glob('llm/**/record.audit.jsonl')):
        records = [json.loads(line).get('provider_record') for line in audit.read_text().splitlines()]
        records = [r for r in records if r]
        if not records:
            continue
        model = by_model[records[0]['requested_model']]
        usd = records_cost(records, model['price'])
        total += usd
        rel = audit.parent.relative_to(args.run_dir / 'llm')
        rows.append({'kind': 'repair_resume', 'job': rel.parts[-1], 'attempt': str(rel.parent) if rel.parent != Path('.') else 'final',
                     'model': model['name'], 'usd': usd, 'calls': len(records)})
    args.out.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    print(json.dumps({'entries': len(rows), 'usd': round(total, 4)}))


if __name__ == '__main__':
    main()
