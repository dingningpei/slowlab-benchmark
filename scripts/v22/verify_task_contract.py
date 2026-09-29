#!/usr/bin/env python3
"""Validate the Phase-0 contract and run its logical campaign example."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from slowlab.v22.task_contract import dry_run

if __name__ == '__main__':
    contract = json.loads((ROOT/'configs/v22/task_contract_v0.json').read_text())
    example = json.loads((ROOT/'configs/v22/campaign_example_v0.json').read_text())
    print(json.dumps(dry_run(contract, example), indent=2))
