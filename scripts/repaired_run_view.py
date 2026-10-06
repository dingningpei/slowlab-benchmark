#!/usr/bin/env python3
"""A run directory view of the repaired formal run, for the accounting script and the call audit (decision 2026-10-05).

    repaired_run_view.py --formal F --repair R --assembled A --out V [--audit-report OUT.json]

V/campaigns/<job>.json is the assembled record with settlement paths pointed at the local copies;
for every repaired LLM job its call records and V/campaigns/<job>.audit.jsonl are the conversation as it
now stands: the original audit lines outside the Full branch, the original Full lines up to the resume
point (replayed byte-identically in the repair) and the repair's live Full lines. Superseded original
Full calls are left out (their cost stays in the original ledger). V/evaluations.json names the
assembled evaluations. Then scripts/analyse_pilot_campaigns.py runs on V unchanged.

--audit-report: every live repair call (resumed branches, their failed attempts and E3 Readers) served by the planned model
and provider with the planned reasoning mode, temperature and token limit; booked repair spend equal
to the spend recomputed from the call records.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_formal_run import REASONING  # noqa: E402
from slowlab.spend import records_cost  # noqa: E402


def local_settlement(path: str, F: Path, R: Path) -> str:
    p = Path(path)
    parts = p.parts
    if 'llm' in parts and 'private' in parts:  # repair: .../llm/<job>/private/<branch>/settlement.json
        i = parts.index('llm')
        return str(R.joinpath(*parts[i:]))
    i = parts.index('private')  # original: .../private/<job>/<branch>/settlement.json
    return str(F.joinpath(*parts[i:]))


def check_call(c, m, label, problems):
    if c.get('actual_model') != m['llm']['model'] or c.get('requested_model') != m['llm']['model']:
        problems.append(f"{label}: model {c.get('actual_model')}")
    exp = m['llm'].get('expected_provider')
    if exp and c.get('provider') != exp:
        problems.append(f"{label}: provider {c.get('provider')}")
    if c.get('reasoning_mode') != REASONING[m['name']]:
        problems.append(f"{label}: reasoning {c.get('reasoning_mode')}")
    if c.get('temperature') != m['llm']['temperature'] or c.get('max_tokens') != m['llm']['max_tokens']:
        problems.append(f'{label}: generation settings')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--formal', type=Path, required=True)
    ap.add_argument('--repair', type=Path, required=True)
    ap.add_argument('--assembled', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--audit-report', type=Path)
    args = ap.parse_args()
    F, R, A, V = (p.resolve() for p in (args.formal, args.repair, args.assembled, args.out))
    models = {m['name']: m for m in json.loads((ROOT / 'configs/formal_models_v1.json').read_text())['models']}
    repaired = set(json.loads((R / 'llm_jobs.json').read_text()))
    batch = json.loads((A / 'campaigns.json').read_text())
    (V / 'campaigns').mkdir(parents=True, exist_ok=True)
    problems, live = [], Counter()
    spend_resume = 0.0
    for job in batch['jobs']:
        jid = job['id']
        rec = json.loads((A / 'campaigns' / f'{jid}.json').read_text())
        if job['method'] == 'llm':
            for b in ('full', 'endpoint'):
                rec['branches'][b]['settlement'] = local_settlement(rec['branches'][b]['settlement'], F, R)
            orig = [json.loads(x) for x in (F / 'campaigns' / f'{jid}.audit.jsonl').read_text().splitlines()]
            if jid in repaired:
                new = [json.loads(x) for x in (R / 'llm' / jid / 'record.audit.jsonl').read_text().splitlines()]
                new_full = [x for x in new if x['label'] == 'full']
                k = sum(1 for x in new_full if not x.get('provider_record'))
                old_full = [x for x in orig if x['label'] == 'full']
                if [x['messages_sha256'] for x in old_full[:k]] != [x['messages_sha256'] for x in new_full[:k]]:
                    problems.append(f'{jid}: replayed Full prefix differs from the original audit')
                if any(x.get('provider_record') for x in new if x['label'] != 'full'):
                    problems.append(f'{jid}: live calls outside the Full branch')
                merged = [x for x in orig if x['label'] != 'full'] + old_full[:k] + new_full[k:]
                rec['provider_call_records'] = [x['provider_record'] for x in merged if x.get('provider_record')]
                m = models[job['model_name']]
                for x in new_full[k:]:
                    check_call(x['provider_record'], m, jid, problems)
                    live[m['name']] += 1
                spend_resume += records_cost([x['provider_record'] for x in new_full[k:]], m['price'])
            else:
                merged = orig
            (V / 'campaigns' / f'{jid}.audit.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in merged))
        (V / 'campaigns' / f'{jid}.json').write_text(json.dumps(rec) + '\n')
    (V / 'campaigns.json').write_text(json.dumps({**batch, 'out_dir': str(V / 'campaigns')}, indent=1))
    ev = json.loads((F / 'evaluations.json').read_text())
    (V / 'evaluations.json').write_text(json.dumps({**ev, 'out_dir': str(A / 'evaluations')}, indent=1))
    print(json.dumps({'campaigns': len(batch['jobs']), 'repaired_llm': len(repaired)}))
    if not args.audit_report:
        return
    failed_calls = Counter()
    for f in sorted(R.glob('llm/*/*/record.audit.jsonl')):  # failed attempts kept under llm/<attempt>/<job>/
        jid = f.parent.name
        m = models[next(j['model_name'] for j in batch['jobs'] if j['id'] == jid)]
        for x in f.read_text().splitlines():
            c = json.loads(x).get('provider_record')
            if c:
                check_call(c, m, f'{f.parent.parent.name}/{jid}', problems)
                failed_calls[m['name']] += 1
    spend_e3, e3_calls = 0.0, Counter()
    for f in sorted((A / 'e3' / 'llm').glob('*.audit.jsonl')):
        recs = [json.loads(x).get('provider_record') for x in f.read_text().splitlines()]
        recs = [x for x in recs if x]
        if not recs:
            continue
        name = next(n for n in models if f.name.startswith(n + '_'))
        orig_audit = F / 'e3' / 'llm' / f.name
        if orig_audit.exists() and orig_audit.read_bytes() == f.read_bytes():
            continue  # reused from the original E3 run, audited there
        for c in recs:
            check_call(c, models[name], f.stem, problems)
            e3_calls[name] += 1
        spend_e3 += records_cost(recs, models[name]['price'])
    ledger = [json.loads(x) for x in (R / 'ledger.jsonl').read_text().splitlines() if x.strip()]
    booked_e3 = sum(x['usd'] for x in ledger if x['kind'] == 'e3_reader')
    resume_rows = [json.loads(x) for x in (R / 'ledger_repair_resume.jsonl').read_text().splitlines() if x.strip()]
    booked_resume_final = sum(x['usd'] for x in resume_rows if x['attempt'] == 'final')
    if abs(booked_e3 - spend_e3) > 1e-9:
        problems.append(f'E3 ledger {booked_e3:.6f} differs from call records {spend_e3:.6f}')
    if abs(booked_resume_final - spend_resume) > 1e-9:
        problems.append(f'resume ledger (final attempts) {booked_resume_final:.6f} differs from call records {spend_resume:.6f}')
    orig_audit = json.loads((ROOT / 'results/formal_audit_20261005.json').read_text())
    resume_all = sum(x['usd'] for x in resume_rows)
    if sum(failed_calls.values()) + sum(live.values()) != sum(x['calls'] for x in resume_rows):
        problems.append('resume call count differs from the repair ledger')
    out = {'format': 'slowlab-repair-audit-v1', 'live_resume_calls_by_model': dict(live),
           'failed_attempt_calls_by_model': dict(failed_calls), 'e3_reader_calls_by_model': dict(e3_calls),
           'spend_resume_final_usd': spend_resume, 'spend_resume_all_attempts_usd': resume_all,
           'spend_e3_readers_usd': spend_e3, 'spend_repair_total_usd': resume_all + spend_e3,
           'original_ledger_total_usd': orig_audit['ledger_total_usd'],
           'total_with_repair_usd': orig_audit['ledger_total_usd'] + resume_all + spend_e3,
           'original_llm_calls': sum(orig_audit['llm_calls_by_model'].values()),
           'repair_llm_calls_all_attempts': sum(x['calls'] for x in resume_rows) + sum(e3_calls.values()),
           'problems': problems, 'passed': not problems}
    args.audit_report.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
