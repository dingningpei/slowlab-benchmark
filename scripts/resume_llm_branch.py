#!/usr/bin/env python3
"""Repair one LLM Full branch from the first point a predictor output reached the model (decision 2026-10-05).

    resume_llm_branch.py --job-spec SPEC.json --audit RECORD.audit.jsonl --private-dir NEW_DIR --out NEW_RECORD.json
                         [--live-from auto|none|N] [--env-file KEYS]

The branch is rebuilt by the same code path as the formal run (initial recommendation, shared day-0
design in the Full process, then the Full branch). Every model call before the live point is answered
from the recorded audit, and the outgoing messages must be byte-identical to the recorded ones
(sha256 of the canonical messages); any difference stops the repair. ``auto`` sets the live point at
the first Full-branch call whose message carries a process-predictor output: that message now holds
the corrected output, and from there on the model is called live with the job's settings. ``none``
replays the whole branch (a check that the repair path reproduces the original). The Endpoint branch
is not rerun. Model calls are audited to NEW_RECORD.audit.jsonl; live provider records are kept.
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
from slowlab.agent_protocol import canonical, digest, public_task_view  # noqa: E402
from slowlab.branching import assignment_notice  # noqa: E402
from slowlab.executor_server import _path  # noqa: E402
from slowlab.llm_agent import LLMCampaignAgent, LLMConfig, initial_recommendation  # noqa: E402
from slowlab.outbound_audit import AuditedCompleter, read_audit  # noqa: E402
from slowlab.private_runs import campaign_spec  # noqa: E402
from slowlab.prompt_firewall import load_blinding_policy  # noqa: E402
from slowlab.tools import Toolbox  # noqa: E402

PREDICTION_KEY = 'predicted_contribution_margin_eur_m2'


class ReplayMismatch(RuntimeError):
    pass


class ReplayThenLive:
    """Answers calls 1..live_from-1 from the recorded audit (verifying the outgoing messages), then calls live."""

    def __init__(self, label, recorded, live_from, make_live):
        self.label, self.recorded, self.live_from, self._make_live = label, recorded, live_from, make_live
        self.calls, self.modes, self._live = 0, [], None
        self.call_records = []

    def __call__(self, messages):
        self.calls += 1
        if self.live_from is None or self.calls < self.live_from:
            if self.calls > len(self.recorded):
                raise ReplayMismatch(f'{self.label}: more calls than recorded ({self.calls})')
            rec = self.recorded[self.calls - 1]
            if digest(canonical(messages)) != rec['messages_sha256']:
                raise ReplayMismatch(f'{self.label}: call {self.calls} differs from the recorded message')
            self.modes.append('replayed')
            return rec['reply']
        if self._live is None:
            self._live = self._make_live()
        self.modes.append('live')
        reply = self._live(messages)
        self.call_records = getattr(self._live, 'call_records', [])
        return reply


def first_prediction_seq(audit):
    for rec in audit:
        if rec['label'] == 'full' and PREDICTION_KEY in rec['messages'][-1]['content']:
            return rec['seq']
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job-spec', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--private-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--live-from', default='auto')
    ap.add_argument('--env-file', default=None)
    ap.add_argument('--dev-cache', default=None, help='override the job spec weather caches (run on another host)')
    ap.add_argument('--formal-cache', default=None)
    args = ap.parse_args()
    spec = json.loads(args.job_spec.read_text())
    for key in ('dev_cache', 'formal_cache'):
        if getattr(args, key):
            spec[key] = getattr(args, key)
    audit = [r for r in read_audit(args.audit) if r.get('firewall') == 'pass']
    by_label = {}
    for rec in audit:
        by_label.setdefault(rec['label'], []).append(rec)
    for label, recs in by_label.items():
        if [r['seq'] for r in recs] != list(range(1, len(recs) + 1)):
            raise SystemExit(f'audit for {label} is not a complete sequence')
    live_from = (first_prediction_seq(audit) if args.live_from == 'auto' else
                 None if args.live_from == 'none' else int(args.live_from))
    if args.live_from == 'auto' and live_from is None:
        raise SystemExit('this branch never received a predictor output; nothing to repair')
    llm = spec['llm']
    contract_raw = json.loads(_path(spec['contract']).read_text())

    def make_live():
        if llm['model'].startswith('python:'):
            if spec.get('backend') != 'fake':
                raise ValueError('python: model factories are allowed only with the toy backend')
            import importlib
            module, attr = llm['model'][len('python:'):].rsplit(':', 1)
            return getattr(importlib.import_module(module), attr)(public_task_view(contract_raw, None))
        import os
        from slowlab.providers import load_dotenv, openai_compatible
        if args.env_file:
            os.environ['SLOWLAB_ENV_FILE'] = args.env_file
        load_dotenv()
        return openai_compatible(llm['model'], temperature=llm.get('temperature', 0.7), max_tokens=llm.get('max_tokens', 2048),
                                 generation_seed=llm.get('generation_seed'), json_mode=llm.get('json_mode', False),
                                 reasoning_effort=llm.get('reasoning_effort'), provider_only=llm.get('provider_only'),
                                 expected_provider=llm.get('expected_provider'))

    private = args.private_dir.resolve()
    private.mkdir(parents=True, exist_ok=True)
    policies = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
    server = campaign_spec(site=spec['site'], year=spec['year'], contract=spec['contract'], feedback_mode=None,
                           fallback_policy=policies['policy_a'], private_dir=private, dev_cache=spec.get('dev_cache'),
                           formal_cache=spec.get('formal_cache'), backend=spec.get('backend', 'greenlight'),
                           trace=spec.get('trace', True))
    blinding = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
    audit_out = args.out.with_suffix('.audit.jsonl')
    replays = {}

    def make(label):
        replays[label] = ReplayThenLive(label, by_label.get(label, []), live_from if label == 'full' else None, make_live)
        return AuditedCompleter(replays[label], blinding, audit_out, label=label)

    began = time.monotonic()
    config = LLMConfig()
    shared_view = public_task_view(contract_raw, None)
    initial, initial_log = initial_recommendation(shared_view, make('initial_recommendation'),
                                                  config.initial_recommendation_attempts)
    fallback = initial if initial is not None else policies['policy_a']
    base = {k: v for k, v in server.items() if k not in ('feedback_mode', 'fallback_policy', 'trace', 'settlement_out', 'failure_out')}
    folder = private / 'full'
    folder.mkdir(parents=True, exist_ok=True)
    site = {**base, 'feedback_mode': 'full', 'fallback_policy': fallback, 'trace': str(folder / 'trace.jsonl.gz'),
            'settlement_out': str(folder / 'settlement.json'), 'failure_out': str(folder / 'failure.json')}
    (folder / 'site.json').write_text(json.dumps(site, indent=2))
    with CampaignProcess(folder / 'site.json', private_log=folder / 'server.log') as campaign:
        toolbox = Toolbox(campaign.session, int(spec['tool_seed']))
        prefix_agent = LLMCampaignAgent(campaign.session, toolbox, make('shared_day_0'), config, task=shared_view)
        prefix = prefix_agent.run_day_zero_design(24)
        view = campaign.session.task
        agent = LLMCampaignAgent(campaign.session, toolbox, make('full'), config, task=view, history=prefix['history'],
                                 counts=prefix['counts'], notice=assignment_notice(view), notes=prefix.get('notes', ''))
        summary = agent.run()
        transcript = campaign.session.transcript
        code = campaign.close()
    audit_bytes = audit_out.read_bytes() if audit_out.exists() else b''
    out = {'status': 'completed' if code == 0 else 'failed', 'method': 'llm', 'model': llm['model'],
           'repair': {'decision': 'RESEARCH_PLAN decision 2026-10-05', 'live_from_full_call': live_from,
                      'modes': {label: r.modes for label, r in replays.items()},
                      'replayed_calls_verified': sum(m == 'replayed' for r in replays.values() for m in r.modes),
                      'live_calls': sum(m == 'live' for r in replays.values() for m in r.modes),
                      'source_audit_sha256': hashlib.sha256(args.audit.read_bytes()).hexdigest()},
           'initial_recommendation': fallback, 'initial_recommendation_log': initial_log,
           'branches': {'full': {'summary': summary, 'public_transcript': transcript,
                                 'settlement': str(folder / 'settlement.json'), 'server_exit_code': code}},
           'outbound_audit': {'path': audit_out.name, 'sha256': hashlib.sha256(audit_bytes).hexdigest()},
           'provider_call_records': replays['full'].call_records,
           'elapsed_seconds': round(time.monotonic() - began, 1)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out) + '\n')
    print(json.dumps({'status': out['status'], **{k: out['repair'][k] for k in ('live_from_full_call', 'replayed_calls_verified', 'live_calls')}}))
    raise SystemExit(0 if code == 0 else 3)


if __name__ == '__main__':
    main()
