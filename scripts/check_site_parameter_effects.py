#!/usr/bin/env python3
"""One crop season per compartment with different site parameters (mechanism check).

All four compartments start the same fixed policy at day 0 under one weather
year; each compartment carries its own parameter values. After one crop cycle
the private ledgers give harvest, resources and margin per compartment. This
checks that each parameter acts in the model and how strongly; it is not used
to choose distribution ranges, which come from the literature.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab.cabauw_weather import CabauwLc1Weather  # noqa: E402
from slowlab.campaign_executor import CampaignExecutor  # noqa: E402
from slowlab.greenlight_source import resolve_greenlight_source  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=None)
    parser.add_argument('--variants', type=Path, required=True, help='JSON list of four {"label", "parameters"}')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads((ROOT / 'configs/task_contract_v5.json').read_text())
    source = resolve_greenlight_source(args.source, contract)
    policy = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())['policy_a']
    variants = json.loads(args.variants.read_text())
    weather = CabauwLc1Weather(args.cache, ROOT / 'configs/weather_gapfilled_plan.json')
    began = time.monotonic()
    x = CampaignExecutor(contract, source, weather, feedback_mode='full', fallback_policy=policy,
                         origin_utc=datetime(2016, 12, 31, 23, tzinfo=timezone.utc),
                         unit_parameters={str(i): v['parameters'] for i, v in enumerate(variants)})
    for unit in range(len(variants)):
        x.dispatch({'action': 'start', 'unit': unit, 'policy': policy})
    x.dispatch({'action': 'advance', 'day': contract['budget']['crop_days']})
    rows = []
    for unit, variant in enumerate(variants):
        summary = x._ledgers[str(unit)].summary()
        rows.append({'label': variant['label'], 'parameters': variant['parameters'],
                     'per_m2': summary['per_m2'], 'margin_eur_m2': summary['synthetic_margin_eur']
                     / contract['facility']['floor_area_m2']})
    out = {'scope': 'mechanism check: one January crop per compartment, policy_a, 2017 weather, no noise',
           'elapsed_seconds': round(time.monotonic() - began, 1), 'rows': rows}
    args.out.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'elapsed_seconds': out['elapsed_seconds'], 'labels': [r['label'] for r in rows]}))


if __name__ == '__main__':
    main()
