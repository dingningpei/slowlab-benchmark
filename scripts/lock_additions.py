#!/usr/bin/env python3
"""Lock files added after the formal lock v1 (the v1 files themselves must stay unchanged).

    lock_additions.py --base configs/formal_lock_v1.json --files A B ... --purpose TEXT --out configs/formal_lock_v2.json
    lock_additions.py --verify configs/formal_lock_v2.json

The record names its base lock by sha256 and lists the sha256 of each added file. ``--verify``
checks the base lock's files (through the base lock's own verifier) and the added files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path)
    ap.add_argument('--files', nargs='+')
    ap.add_argument('--purpose')
    ap.add_argument('--lock-id')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--verify', type=Path)
    args = ap.parse_args()
    if args.verify:
        lock = json.loads(args.verify.read_text())
        base = ROOT / lock['base']['file']
        bad = [] if sha(base) == lock['base']['sha256'] else [lock['base']['file']]
        verifier = lock['base'].get('verifier')
        if verifier:
            proc = subprocess.run([sys.executable, str(ROOT / verifier[0]), *verifier[1:]], cwd=ROOT,
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                bad.append('base lock: ' + proc.stdout.strip())
        bad += [f for f, h in lock['added_files'].items() if not (ROOT / f).exists() or sha(ROOT / f) != h]
        print(json.dumps({'lock': lock['lock_id'], 'match': not bad, 'differences': bad}))
        raise SystemExit(1 if bad else 0)
    base_lock = json.loads(args.base.read_text())
    clash = [f for f in args.files if f in base_lock.get('files', {})]
    if clash:
        raise SystemExit(f'already locked by the base: {clash}')
    verifier = (['scripts/make_formal_lock.py', '--verify', str(args.base)] if 'files' in base_lock
                else ['scripts/lock_additions.py', '--verify', str(args.base)])
    out = {'lock_id': args.lock_id, 'purpose': args.purpose,
           'base': {'file': str(args.base), 'sha256': sha(ROOT / args.base), 'verifier': verifier},
           'added_files': {f: sha(ROOT / f) for f in args.files}}
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
