#!/usr/bin/env python3
"""Run one campaign of one method on one sampled site (batch job; private executor process).

    python3 scripts/run_campaign_job.py SPEC.json

SPEC: {"method": "bo", "bo": {"schedule", "seed", ...BOConfig fields},
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
        if spec['method'] != 'bo':
            raise ValueError('only the BO method is wired into batch jobs so far')
        bo = dict(spec['bo'])
        seed = bo.pop('seed')
        with CampaignProcess(private / 'site.json', private_log=private / 'server.log') as campaign:
            summary = GPBOAgent(campaign.session, seed, BOConfig(**bo)).run()
            transcript = campaign.session.transcript
            code = campaign.close()
        record = {'status': 'completed' if code == 0 else 'failed', 'method': 'bo', 'bo': summary,
                  'public_transcript': transcript, 'settlement': str(private / 'settlement.json')}
        exit_code = 0 if code == 0 else 3
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
