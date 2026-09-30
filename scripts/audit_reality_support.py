#!/usr/bin/env python3
"""Classify proposed values against the reality-support manifest and audit policy.

Exit status is nonzero only for physical errors, undeclared assumptions or unknown
constraints. Source extrapolation is reported, never failed or removed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slowlab.reality_constraints import audit_reality_support, load_audit_policy, load_reality_constraints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", type=Path, help="JSON list or object containing a records list")
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "configs" / "reality_constraints_v2_2.json",
    )
    parser.add_argument("--policy", type=Path, default=ROOT / "configs" / "reality_audit_policy_v1.json")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.records.read_text())
    records = payload["records"] if isinstance(payload, dict) else payload
    report = audit_reality_support(records, load_reality_constraints(args.manifest),
                                   load_audit_policy(args.policy, args.manifest))
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        print(text, end="")
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
