#!/usr/bin/env python3
"""Commit to a private seed list: public record to Git, opening kept private.

    python3 scripts/commit_seeds.py --label test-sites-v0 --secret seeds.json \\
        --public configs/seed_commitment_test_sites_v0.json --opening output/private/seed_opening_test_sites_v0.json

The opening must be written under the gitignored output/private/ (or outside
the repository). ``--verify`` checks an existing public record against its opening.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.commitment import make_commitment, verify  # noqa: E402


def ignored(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
    except ValueError:
        return True
    return subprocess.run(['git', 'check-ignore', '-q', str(path)], cwd=ROOT).returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label')
    parser.add_argument('--secret', type=Path)
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--opening', type=Path, required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        ok = verify(json.loads(args.public.read_text()), json.loads(args.opening.read_text()))
        print(json.dumps({'verified': ok}))
        raise SystemExit(0 if ok else 1)
    if not ignored(args.opening):
        raise SystemExit('refusing to write the opening to a path Git would track: ' + str(args.opening))
    if args.public.exists() or args.opening.exists():
        raise SystemExit('refusing to overwrite an existing commitment or opening')
    public, opening = make_commitment(json.loads(args.secret.read_text()), label=args.label)
    args.opening.parent.mkdir(parents=True, exist_ok=True)
    args.opening.write_text(json.dumps(opening, indent=2) + '\n')
    args.public.write_text(json.dumps(public, indent=2) + '\n')
    print(json.dumps(public))


if __name__ == '__main__':
    main()
