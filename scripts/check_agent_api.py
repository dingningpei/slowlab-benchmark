#!/usr/bin/env python3
"""Real-GreenLight campaign prefix through the agent API process boundary.

Replays the action sequence of check_campaign_executor_dynamic.py through
CampaignProcess -> slowlab.executor_server, then compares the server's private
trace digest with a reference digest from a direct run. Equal digests show the
process boundary changes nothing physical. Not an LLM run.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.agent_client import CampaignProcess  # noqa: E402
from slowlab.agent_protocol import canonical  # noqa: E402
from slowlab.prompt_firewall import assert_outbound_messages_safe, load_blinding_policy  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=None)
    parser.add_argument('--noise-seed', type=int, default=None)
    parser.add_argument('--reference-trace-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    policies = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
    blinding = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
    began = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='agent-api-check-') as private:
        private = Path(private)
        spec = {'backend': 'greenlight', 'contract': 'configs/task_contract_v5.json', 'feedback_mode': 'full',
                'fallback_policy': policies['policy_a'], 'origin_utc': '2016-12-31T23:00:00+00:00',
                'weather': {'cache': str(args.cache.resolve()), 'plan': 'configs/weather_gapfilled_plan.json'},
                'greenlight_source': str(args.source.resolve()) if args.source else None,
                'sensor_noise': ({'config': 'configs/sensor_noise_v0.json', 'seed': args.noise_seed, 'setting': 'main'}
                                 if args.noise_seed is not None else None),
                'soil_boundary_c': None, 'trace': str(private / 'trace.jsonl.gz'),
                'settlement_out': str(private / 'settlement.json'), 'failure_out': str(private / 'failure.json')}
        (private / 'site.json').write_text(json.dumps(spec))
        with CampaignProcess(private / 'site.json', private_log=private / 'server.log') as campaign:
            session = campaign.session
            assert_outbound_messages_safe([{'role': 'user', 'content': canonical(session.task)}], blinding)
            d = 300 / 86400
            session.dispatch({'action': 'start', 'unit': 0, 'policy': policies['policy_a']})
            session.dispatch({'action': 'start', 'unit': 1, 'policy': policies['policy_b']})
            session.dispatch({'action': 'advance', 'day': d})
            first = session.dispatch({'action': 'observe', 'unit': 0, 'variable': 'air_temperature_c', 'start_day': 0})
            session.dispatch({'action': 'stop', 'unit': 0})
            session.dispatch({'action': 'advance', 'day': 1})
            session.dispatch({'action': 'start', 'unit': 2, 'policy': policies['policy_a']})
            session.dispatch({'action': 'advance', 'day': 2 + d})
            session.dispatch({'action': 'start', 'unit': 0, 'policy': policies['policy_b']})
            session.dispatch({'action': 'observe', 'unit': 0, 'variable': 'air_temperature_c',
                              'start_day': 2, 'end_day': 2 + d})
            transcript = session.transcript
            code = campaign.close()
        with gzip.open(private / 'trace.jsonl.gz', 'rt', encoding='utf-8') as stream:
            data = stream.read()
        trace_sha = hashlib.sha256(data.encode()).hexdigest()
        ticks = data.count('\n')
        settlement = json.loads((private / 'settlement.json').read_text())
    public = canonical(transcript)
    assert_outbound_messages_safe([{'role': 'user', 'content': public}], blinding)
    result = {'status': 'passed_agent_api_prefix' if trace_sha == args.reference_trace_sha256 and code == 0 else 'failed',
              'scope': 'real-GreenLight four-unit 577-tick prefix through the process boundary; no LLM',
              'noise_seed': args.noise_seed, 'server_exit_code': code, 'trace_ticks': ticks,
              'trace_uncompressed_sha256': trace_sha, 'reference_trace_sha256': args.reference_trace_sha256,
              'settlement_status': settlement.get('status', 'written'),
              'first_observation_records': len(first['records']), 'public_exchanges': len(transcript),
              'public_transcript_passes_firewall': True, 'elapsed_seconds': round(time.monotonic() - began, 1)}
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if result['status'] != 'passed_agent_api_prefix':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
