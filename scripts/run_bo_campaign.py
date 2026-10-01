#!/usr/bin/env python3
"""Run one GP/BO baseline campaign on the real model through the agent API.

Development use only: one development weather year and one noise seed, not a
formal site. Writes the public summary (BO log, public transcript) to --out;
the private settlement and trace stay in --private-dir.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.agent_client import CampaignProcess  # noqa: E402
from slowlab.bo_agent import BOConfig, GPBOAgent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=None)
    parser.add_argument('--private-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--feedback', choices=('full', 'endpoint'), default='full')
    parser.add_argument('--schedule', choices=('two_waves', 'staggered'), default='two_waves')
    parser.add_argument('--initial-units', type=int, default=2)
    parser.add_argument('--stagger-fraction', type=float, default=0.5)
    parser.add_argument('--bo-seed', type=int, required=True)
    parser.add_argument('--noise-seed', type=int, default=20260929)
    parser.add_argument('--origin-utc', default='2016-12-31T23:00:00+00:00')
    args = parser.parse_args()
    private = args.private_dir.resolve()
    private.mkdir(parents=True, exist_ok=True)
    policies = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
    spec = {'backend': 'greenlight', 'contract': 'configs/task_contract_v7.json', 'feedback_mode': args.feedback,
            'fallback_policy': policies['policy_a'], 'origin_utc': args.origin_utc,
            'weather': {'cache': str(args.cache.resolve()), 'plan': 'configs/weather_gapfilled_plan.json'},
            'greenlight_source': str(args.source.resolve()) if args.source else None,
            'sensor_noise': {'config': 'configs/sensor_noise_v0.json', 'seed': args.noise_seed, 'setting': 'main'},
            'soil_boundary_c': None, 'trace': str(private / 'trace.jsonl.gz'),
            'settlement_out': str(private / 'settlement.json'), 'failure_out': str(private / 'failure.json')}
    (private / 'site.json').write_text(json.dumps(spec, indent=2))
    config = BOConfig(schedule=args.schedule, initial_units=args.initial_units, stagger_fraction=args.stagger_fraction)
    began = time.monotonic()
    with CampaignProcess(private / 'site.json', private_log=private / 'server.log') as campaign:
        summary = GPBOAgent(campaign.session, args.bo_seed, config).run()
        transcript = campaign.session.transcript
        code = campaign.close()
    out = {'status': 'completed' if code == 0 else 'failed', 'scope': 'development BO campaign; not a formal site',
           'feedback': args.feedback, 'bo_seed': args.bo_seed, 'elapsed_seconds': round(time.monotonic() - began, 1),
           'bo': summary, 'public_transcript': transcript}
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('status', 'feedback', 'elapsed_seconds')} |
                     {'completed_crops': summary['completed_crops'], 'recommendation': summary['recommendation']}))


if __name__ == '__main__':
    main()
