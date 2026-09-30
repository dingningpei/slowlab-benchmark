#!/usr/bin/env python3
"""Scan public run outputs against one private site spec; exit 1 on any error finding.

    python3 scripts/check_public_leaks.py --spec PRIVATE/site.json --public OUT_DIR [OUT_FILE ...] \\
        [--private-dir PRIVATE] [--report report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.leak_check import private_needles, scan  # noqa: E402
from slowlab.prompt_firewall import load_blinding_policy  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True, help='private executor site spec')
    parser.add_argument('--public', type=Path, nargs='+', required=True)
    parser.add_argument('--private-dir', type=Path, action='append', default=[])
    parser.add_argument('--report', type=Path, default=None)
    args = parser.parse_args()
    policy = load_blinding_policy(ROOT / 'configs/simulation_blinding_v2_2.json')
    result = scan(args.public, private_needles(json.loads(args.spec.read_text()), policy), args.private_dir)
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'errors', 'warnings')} |
                     {'first_findings': result['findings'][:5]}, indent=1))
    raise SystemExit(1 if result['errors'] else 0)


if __name__ == '__main__':
    main()
