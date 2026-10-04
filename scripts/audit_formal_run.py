#!/usr/bin/env python3
"""Audit of the formal E1/E2 run (Phase 5): identities, models and providers, spend, evaluations.

    audit_formal_run.py --run-dir RUN --report OUT.json

Checks: every planned campaign has exactly one record and it completed; every LLM call was served
by the planned model and provider (DeepSeek: official API; others: the pinned OpenRouter provider)
with the planned reasoning mode, temperature and token limit; booked spend in the ledger equals the
spend recomputed from the call records; every planned evaluation (from evaluations.json) has
exactly one completed record whose identity names the planned site and year. Reads no scores.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.spend import records_cost  # noqa: E402

REASONING = {'deepseek-v4.1-flash': '{"thinking": {"type": "disabled"}}',
             'mimo-v2.6-flash': '{"reasoning": {"enabled": false}}',
             'glm-5.3-flash': '{"reasoning": {"max_tokens": 1024}}'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir
    models = {m['name']: m for m in json.loads((ROOT / 'configs/formal_models_v1.json').read_text())['models']}
    batch = json.loads((run / 'campaigns.json').read_text())
    ids = [j['id'] for j in batch['jobs']]
    problems = []
    if len(ids) != len(set(ids)):
        problems.append('duplicate planned campaign ids')
    files = {p.stem for p in (run / 'campaigns').glob('site*.json')}
    unplanned = sorted(files - set(ids))
    if unplanned:
        problems.append(f'unplanned campaign records: {unplanned[:5]}')
    calls = Counter(); spend_records = 0.0; seen_models = Counter(); seen_providers = Counter()
    for job in batch['jobs']:
        path = run / 'campaigns' / f"{job['id']}.json"
        if not path.exists():
            problems.append(f"missing campaign record {job['id']}")
            continue
        r = json.loads(path.read_text())
        if r.get('status') != 'completed':
            problems.append(f"campaign not completed {job['id']}")
        if job['method'] != 'llm':
            continue
        m = models[job['model_name']]
        expected_provider = m['llm'].get('expected_provider')
        for c in r.get('provider_call_records') or []:
            calls[job['model_name']] += 1
            seen_models[c.get('actual_model')] += 1
            seen_providers[str(c.get('provider'))] += 1
            if c.get('actual_model') != m['llm']['model'] or c.get('requested_model') != m['llm']['model']:
                problems.append(f"{job['id']}: model {c.get('actual_model')}")
            if expected_provider and c.get('provider') != expected_provider:
                problems.append(f"{job['id']}: provider {c.get('provider')}")
            if c.get('reasoning_mode') != REASONING[job['model_name']]:
                problems.append(f"{job['id']}: reasoning {c.get('reasoning_mode')}")
            if c.get('temperature') != m['llm']['temperature'] or c.get('max_tokens') != m['llm']['max_tokens']:
                problems.append(f"{job['id']}: generation settings")
        spend_records += records_cost(r.get('provider_call_records') or [], m['price'])
    ledger = [json.loads(line) for line in (run / 'ledger.jsonl').read_text().splitlines() if line.strip()]
    booked = sum(x['usd'] for x in ledger if x['kind'] == 'campaign')
    if abs(booked - spend_records) > 1e-6:
        problems.append(f'ledger {booked:.6f} differs from call records {spend_records:.6f}')
    ev = json.loads((run / 'evaluations.json').read_text())
    eids = [j['id'] for j in ev['jobs']]
    if len(eids) != len(set(eids)):
        problems.append('duplicate planned evaluation ids')
    for j in ev['jobs']:
        path = Path(ev['out_dir']) / f"{j['id']}.json"
        if not path.exists():
            problems.append(f"missing evaluation {j['id']}")
            continue
        r = json.loads(path.read_text())
        ident = r.get('identity') or {}
        if r.get('status') != 'completed' or ident.get('site_index') != j['site']['site_index'] or ident.get('year') != j['year']:
            problems.append(f"evaluation identity or status {j['id']}")
    out = {'format': 'slowlab-formal-audit-v1', 'campaigns_planned': len(ids), 'campaign_records': len(files),
           'evaluations_planned': len(eids), 'llm_calls_by_model': dict(calls), 'served_models': dict(seen_models),
           'providers': dict(seen_providers), 'spend_from_call_records_usd': spend_records, 'ledger_campaign_usd': booked,
           'ledger_total_usd': sum(x['usd'] for x in ledger), 'ledger_by_kind': dict(Counter(x['kind'] for x in ledger)),
           'problems': problems, 'passed': not problems}
    args.report.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('passed', 'llm_calls_by_model', 'providers', 'spend_from_call_records_usd',
                                          'ledger_campaign_usd', 'ledger_total_usd')} | {'problems': problems[:10]}, indent=1))


if __name__ == '__main__':
    main()
