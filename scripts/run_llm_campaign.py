#!/usr/bin/env python3
"""Run one language-model campaign: initial recommendation, then the campaign.

Every model call passes the blinding firewall and is written to an outbound
audit log. The initial recommendation is asked from the public task view only
(identical for every site under one contract and feedback condition) and
becomes the site's fallback recommendation. The private settlement and trace
stay in --private-dir; --out receives only public material.

Real provider calls spend money: use --model scripted for a dry run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.agent_client import CampaignProcess  # noqa: E402
from slowlab.agent_protocol import public_task_view  # noqa: E402
from slowlab.llm_agent import LLMCampaignAgent, LLMConfig, initial_recommendation  # noqa: E402
from slowlab.outbound_audit import AuditedCompleter  # noqa: E402
from slowlab.prompt_firewall import load_blinding_policy  # noqa: E402
from slowlab.scripted_model import ScriptedModel  # noqa: E402
from slowlab.tools import Toolbox  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='provider model id, or "scripted" for a dry run')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--generation-seed', type=int, default=None)
    parser.add_argument('--json-mode', action='store_true')
    parser.add_argument('--backend', choices=('greenlight', 'fake'), default='greenlight')
    parser.add_argument('--cache', type=Path, default=None, help='weather cache (greenlight backend)')
    parser.add_argument('--source', type=Path, default=None)
    parser.add_argument('--contract', default='configs/task_contract_v5.json')
    parser.add_argument('--feedback', choices=('full', 'endpoint'), default='full')
    parser.add_argument('--branched', action='store_true',
                        help='formal protocol: shared day-0 design, then Full and Endpoint branches')
    parser.add_argument('--tool-seed', type=int, required=True)
    parser.add_argument('--noise-seed', type=int, default=20260929)
    parser.add_argument('--origin-utc', default='2016-12-31T23:00:00+00:00')
    parser.add_argument('--private-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    private = args.private_dir.resolve()
    private.mkdir(parents=True, exist_ok=True)
    contract = json.loads((ROOT / args.contract).read_text())
    task = public_task_view(contract, args.feedback)
    blinding = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
    if args.model == 'scripted':
        provider = ScriptedModel(task)
    else:
        provider = None
    if provider is None:
        from slowlab.providers import openai_compatible
        provider = openai_compatible(args.model, temperature=args.temperature, max_tokens=args.max_tokens,
                                     generation_seed=args.generation_seed, json_mode=args.json_mode)
    audit_path = args.out.with_suffix('.audit.jsonl')
    began = time.monotonic()
    base = {'backend': args.backend, 'contract': args.contract, 'origin_utc': args.origin_utc, 'soil_boundary_c': None}
    if args.backend == 'greenlight':
        base.update(weather={'cache': str(args.cache.resolve()), 'plan': 'configs/weather_gapfilled_plan.json'},
                    greenlight_source=str(args.source.resolve()) if args.source else None,
                    sensor_noise={'config': 'configs/sensor_noise_v0.json', 'seed': args.noise_seed, 'setting': 'main'})
    if args.branched:
        from slowlab.branching import run_branched_campaign
        completers = []

        def make(label):
            completers.append(AuditedCompleter(provider, blinding, audit_path, label=label))
            return completers[-1]
        result = run_branched_campaign(contract, base, private,  make, tool_seed=args.tool_seed,
                                       default_policy=json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())['policy_a'])
        audit_bytes = audit_path.read_bytes() if audit_path.exists() else b''
        out = {'status': 'completed' if all(result[m]['server_exit_code'] == 0 for m in ('full', 'endpoint')) else 'failed',
               'protocol': 'branched: shared day-0 design, then full and endpoint', 'model': args.model,
               'tool_seed': args.tool_seed, 'generation_seed': args.generation_seed, **result,
               'outbound_audit': {'path': audit_path.name, 'sha256': hashlib.sha256(audit_bytes).hexdigest(),
                                  'model_calls': sum(c.calls for c in completers)},
               'provider_call_records': getattr(provider, 'call_records', []),
               'elapsed_seconds': round(time.monotonic() - began, 1)}
        args.out.write_text(json.dumps(out, indent=2) + '\n')
        print(json.dumps({'status': out['status'], 'model': args.model, 'elapsed_seconds': out['elapsed_seconds'],
                          'full': result['full']['summary']['counts'], 'endpoint': result['endpoint']['summary']['counts']}))
        return

    initial_complete = AuditedCompleter(provider, blinding, audit_path, label='initial_recommendation')
    initial, initial_log = initial_recommendation(task, initial_complete)
    fallback_frozen = initial is None
    if initial is None:
        initial = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())['policy_a']

    spec = {**base, 'feedback_mode': args.feedback, 'fallback_policy': initial,
            'trace': str(private / 'trace.jsonl.gz'), 'settlement_out': str(private / 'settlement.json'),
            'failure_out': str(private / 'failure.json')}
    (private / 'site.json').write_text(json.dumps(spec, indent=2))

    campaign_complete = AuditedCompleter(provider, blinding, audit_path, label='campaign')
    with CampaignProcess(private / 'site.json', private_log=private / 'server.log') as campaign:
        toolbox = Toolbox(campaign.session, args.tool_seed)
        summary = LLMCampaignAgent(campaign.session, toolbox, campaign_complete, LLMConfig()).run()
        transcript = campaign.session.transcript
        code = campaign.close()
    audit_bytes = audit_path.read_bytes() if audit_path.exists() else b''
    out = {'status': 'completed' if code == 0 else 'failed', 'model': args.model, 'feedback': args.feedback,
           'tool_seed': args.tool_seed, 'generation_seed': args.generation_seed,
           'initial_recommendation': initial, 'initial_recommendation_log': initial_log,
           'initial_recommendation_is_frozen_default': fallback_frozen,
           'agent': summary, 'public_transcript': transcript,
           'outbound_audit': {'path': audit_path.name, 'sha256': hashlib.sha256(audit_bytes).hexdigest(),
                              'model_calls': initial_complete.calls + campaign_complete.calls},
           'provider_call_records': getattr(provider, 'call_records', []),
           'elapsed_seconds': round(time.monotonic() - began, 1)}
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: out[k] for k in ('status', 'model', 'feedback', 'elapsed_seconds')} |
                     {'counts': summary['counts']}))


if __name__ == '__main__':
    main()
