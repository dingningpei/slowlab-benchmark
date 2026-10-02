#!/usr/bin/env python3
"""Run one campaign of one method on one sampled site (batch job; private executor process).

    python3 scripts/run_campaign_job.py SPEC.json

SPEC: {"method": "bo", "bo": {"schedule", "seed", ...BOConfig fields},
       or {"method": "llm", "llm": {"model" ("scripted" or a provider id), "temperature", "max_tokens",
           "generation_seed", "json_mode", "reasoning_effort"}, "tool_seed"} (branched Full/Endpoint protocol,
           "feedback" ignored),
       "site": {"distribution", "master_seed", "site_index"}, "year", "feedback": "full"|"endpoint",
       "contract", "backend", "dev_cache", "formal_cache", "private_dir", "out"}
The public summary (method log, public transcript) goes to "out"; the private
settlement, site spec, trace and server log stay in "private_dir". Failures are
recorded with status "failed" and exit code 3.
"""
from __future__ import annotations

import json
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.agent_client import CampaignProcess  # noqa: E402
from slowlab.bo_agent import BOConfig, GPBOAgent  # noqa: E402
from slowlab.executor_server import _path  # noqa: E402
from slowlab.private_runs import campaign_spec, sha  # noqa: E402


class _Done(Exception):
    def __init__(self, record, code):
        self.record, self.code = record, code


def run_llm(spec, server, private, out, default_policy):
    import hashlib
    from slowlab.agent_protocol import public_task_view
    from slowlab.branching import run_branched_campaign
    from slowlab.outbound_audit import AuditedCompleter
    from slowlab.prompt_firewall import load_blinding_policy
    from slowlab.site_parameters import site_contract
    contract = site_contract(json.loads(_path(spec['contract']).read_text()), server.get('site') or {})
    llm = spec['llm']
    if llm['model'] == 'scripted':
        from slowlab.scripted_model import ScriptedModel
        provider = ScriptedModel(public_task_view(contract, None))
    elif llm['model'].startswith('python:'):
        # Test hook: a model factory "python:module:attr(task)"; only with the toy backend.
        if spec.get('backend') != 'fake':
            raise ValueError('python: model factories are allowed only with the toy backend')
        import importlib
        module, attr = llm['model'][len('python:'):].rsplit(':', 1)
        provider = getattr(importlib.import_module(module), attr)(public_task_view(contract, None))
    else:
        from slowlab.providers import load_dotenv, openai_compatible
        load_dotenv()
        provider = openai_compatible(llm['model'], temperature=llm.get('temperature', 0.7),
                                     max_tokens=llm.get('max_tokens', 2048), generation_seed=llm.get('generation_seed'),
                                     json_mode=llm.get('json_mode', False), reasoning_effort=llm.get('reasoning_effort'))
    blinding = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
    audit_path = out.with_suffix('.audit.jsonl')
    completers = []

    def make(label):
        completers.append(AuditedCompleter(provider, blinding, audit_path, label=label))
        return completers[-1]
    base = {k: v for k, v in server.items()
            if k not in ('feedback_mode', 'fallback_policy', 'trace', 'settlement_out', 'failure_out')}
    result = run_branched_campaign(json.loads(_path(spec['contract']).read_text()), base, private, make,
                                   tool_seed=int(spec['tool_seed']), default_policy=default_policy)
    audit = audit_path.read_bytes() if audit_path.exists() else b''
    ok = all(result[m]['server_exit_code'] == 0 for m in ('full', 'endpoint'))
    return {'status': 'completed' if ok else 'failed', 'method': 'llm', 'model': llm['model'],
            'initial_recommendation': result['initial_recommendation'],
            'initial_recommendation_is_frozen_default': result['initial_recommendation_is_frozen_default'],
            'initial_recommendation_log': result['initial_recommendation_log'],
            'shared_day_0': {k: v for k, v in result['shared_day_0'].items() if k != 'history'} if 'shared_day_0' in result else None,
            'branches': {m: {'summary': result[m]['summary'], 'public_transcript': result[m]['transcript'],
                             'settlement': str(private / m / 'settlement.json'), 'server_exit_code': result[m]['server_exit_code']}
                         for m in ('full', 'endpoint')},
            'outbound_audit': {'path': audit_path.name, 'sha256': hashlib.sha256(audit).hexdigest(),
                               'model_calls': sum(c.calls for c in completers)},
            'provider_call_records': getattr(provider, 'call_records', [])}


def main():
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text())
    out, private = _path(spec['out']), _path(spec['private_dir'])
    private.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    try:
        policies = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
        server = campaign_spec(site=spec['site'], year=spec['year'], contract=spec['contract'],
                               feedback_mode=spec['feedback'], fallback_policy=policies['policy_a'], private_dir=private,
                               dev_cache=spec.get('dev_cache'), formal_cache=spec.get('formal_cache'),
                               backend=spec.get('backend', 'greenlight'), trace=spec.get('trace', True))
        (private / 'site.json').write_text(json.dumps(server, indent=2))
        if spec['method'] == 'llm':
            record = run_llm(spec, server, private, out, policies['policy_a'])
            exit_code = 0 if record['status'] == 'completed' else 3
            raise _Done(record, exit_code)
        if spec['method'] != 'bo':
            raise ValueError('unknown method')
        bo = dict(spec['bo'])
        seed = bo.pop('seed')
        with CampaignProcess(private / 'site.json', private_log=private / 'server.log') as campaign:
            summary = GPBOAgent(campaign.session, seed, BOConfig(**bo)).run()
            transcript = campaign.session.transcript
            code = campaign.close()
        record = {'status': 'completed' if code == 0 else 'failed', 'method': 'bo', 'bo': summary,
                  'public_transcript': transcript, 'settlement': str(private / 'settlement.json')}
        exit_code = 0 if code == 0 else 3
    except _Done as done:
        record, exit_code = done.record, done.code
    except Exception as error:
        record = {'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
        exit_code = 3
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    record.update(spec_sha256=sha(spec_path), elapsed_seconds=round(time.monotonic() - began, 1),
                  peak_rss_mb=round(rss / (1024 * 1024 if sys.platform == 'darwin' else 1024), 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record) + '\n')
    print(json.dumps({k: record.get(k) for k in ('status', 'elapsed_seconds', 'error')}))
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
