#!/usr/bin/env python3
"""Lock for the field-order repair (lock v5, RESEARCH_PLAN decisions 2026-10-05).

    make_repair_lock.py --environment ENV.json --out configs/formal_lock_v5.json
    make_repair_lock.py --verify configs/formal_lock_v5.json

Hashes every file under slowlab/ and scripts/ and the configs the repair reads, and the rendered prompts.
Records, against lock v1, which files changed and why (only the three fixed modules may differ) and
requires the prompts to equal lock v1's, so the replayed conversation prefixes stay byte-identical.
``--verify`` fails on any difference from this lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_formal_lock import CONFIGS, prompt_hashes, versions  # noqa: E402

REPAIR_CONFIGS = CONFIGS + ['configs/prior_bo_v2.json', 'configs/formal_lock_v1.json']
FIXED = {'slowlab/process_predictor.py': 'policy features in the training order (POLICY_ORDER), whatever the task key order',
         'slowlab/prior_bo.py': 'kernel inputs in the recorded policy_order; config v2 required; version gp-bo-prior-v1.1',
         'slowlab/history_packet.py': 'GP Reader maps fields by name into the kernel order; version gp-reader-v2.1'}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree() -> dict:
    files = sorted(p for d in ('slowlab', 'scripts') for p in (ROOT / d).rglob('*.py') if '__pycache__' not in p.parts)
    out = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in files}
    for c in REPAIR_CONFIGS:
        out[c] = sha((ROOT / c).read_bytes())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--environment', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--verify', type=Path)
    args = ap.parse_args()
    if args.verify:
        lock = json.loads(args.verify.read_text())
        files, prompts = tree(), prompt_hashes()
        bad = sorted(k for k in lock['files'] if lock['files'][k] != files.get(k))
        bad += [k for k in lock['prompts'] if lock['prompts'][k] != prompts.get(k)]
        if versions() != lock['versions']:
            bad.append('versions')
        added = sorted(set(files) - set(lock['files']))
        print(json.dumps({'lock': lock['lock_id'], 'match': not bad, 'differences': bad, 'added_files': added}))
        raise SystemExit(1 if bad else 0)
    v1 = json.loads((ROOT / 'configs/formal_lock_v1.json').read_text())
    files, prompts = tree(), prompt_hashes()
    if prompts != v1['prompts']:
        raise SystemExit('prompts differ from lock v1: replayed prefixes would not be byte-identical')
    changed = sorted(k for k in v1['files'] if v1['files'][k] != files.get(k))
    unexpected = [k for k in changed if k not in FIXED]
    if unexpected:
        raise SystemExit(f'files changed since lock v1 without a recorded reason: {unexpected}')
    lock = {'lock_id': 'slowlab-formal-lock-v5', 'date': '2026-10-05',
            'decision': 'RESEARCH_PLAN decisions 2026-10-05 (field-order bug; repair on the original 48 test sites)',
            'scope': ('repair runs: radius re-selection on development sites, main baseline rerun, resumed LLM Full branches, '
                      'their evaluations, E3 packets with changed sources, boundary sensitivity of changed recommendations; '
                      'analysis with the code locked in v2 and v3'),
            'fix_commit': 'ac75ed0', 'changed_since_lock_v1': {k: FIXED[k] for k in changed},
            'prompts_equal_lock_v1': True, 'versions': versions(), 'prompts': prompts,
            'lock_v1_sha256': sha((ROOT / 'configs/formal_lock_v1.json').read_bytes()),
            'environment': json.loads(args.environment.read_text()) if args.environment else None, 'files': files}
    args.out.write_text(json.dumps(lock, indent=2) + '\n')
    print(json.dumps({'files': len(files), 'changed_since_lock_v1': changed, 'versions': lock['versions']}, indent=1))


if __name__ == '__main__':
    main()
