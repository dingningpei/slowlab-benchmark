#!/usr/bin/env python3
"""Regenerate a campaign's private trace from its public transcript and compare it with the stored one.

    replay_campaign_trace.py --site-spec PRIVATE/site.json --transcript RECORD.json [--branch full] --out DIR

Every request in the public transcript is dispatched again, in order, to a fresh executor built from
the same private site spec (rejected requests are rejected again). The simulation is deterministic, so
the regenerated trace and settlement should be byte-identical (decompressed) to the stored ones.
No model is called.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.agent_client import CampaignProcess, InvalidAction  # noqa: E402


def digest(path: Path) -> str:
    with gzip.open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site-spec', type=Path, required=True)
    ap.add_argument('--transcript', type=Path, required=True)
    ap.add_argument('--branch', default=None, help='full or endpoint for an LLM record; omit for a baseline record')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    spec = json.loads(args.site_spec.read_text())
    record = json.loads(args.transcript.read_text())
    transcript = record['branches'][args.branch]['public_transcript'] if args.branch else record['public_transcript']
    args.out.mkdir(parents=True, exist_ok=True)
    stored_trace, stored_settlement = Path(spec['trace']), Path(spec['settlement_out'])
    spec.update(trace=str(args.out / 'trace.jsonl.gz'), settlement_out=str(args.out / 'settlement.json'),
                failure_out=str(args.out / 'failure.json'))
    (args.out / 'site.json').write_text(json.dumps(spec))
    mismatches = 0
    with CampaignProcess(args.out / 'site.json', private_log=args.out / 'server.log') as campaign:
        for entry in transcript:
            try:
                campaign.session.dispatch(entry['request'])
            except InvalidAction:
                pass
            mismatches += campaign.session.transcript[-1]['response_sha256'] != entry['response_sha256']
        code = campaign.close()
    out = {'stored_trace_sha256': digest(stored_trace), 'replayed_trace_sha256': digest(args.out / 'trace.jsonl.gz'),
           'settlement_identical': stored_settlement.read_bytes() == (args.out / 'settlement.json').read_bytes(),
           'response_mismatches': mismatches, 'requests': len(transcript), 'exit_code': code}
    out['trace_identical'] = out['stored_trace_sha256'] == out['replayed_trace_sha256']
    (args.out / 'replay.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == '__main__':
    main()
