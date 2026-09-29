#!/usr/bin/env python3
"""Verify an already-audited second historical forcing year without simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab.v22_cabauw_weather import CabauwLc1ExpandedDevelopmentWeather


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = ROOT / 'configs/v22_weather_v3_expanded_acquisition_proposal.json'
    audit = ROOT / 'configs/v22_weather_v3_development_full_audit.json'
    weather = CabauwLc1ExpandedDevelopmentWeather(args.cache, plan, audit)
    origin = datetime(2013, 12, 31, 23, tzinfo=timezone.utc)
    month_swd_difference = {}
    for tick in range(365 * 288 + 1):
        now = origin + timedelta(seconds=tick * 300)
        forcing = weather.at_utc(now)
        if not forcing.interval_start_utc <= now < forcing.interval_end_utc:
            raise AssertionError('weather interval does not cover step')
        if forcing.i_glob_w_m2 < 0:
            raise AssertionError('negative clipped solar forcing')
        month_swd_difference[now.strftime('%Y%m')] = weather.last_duplicate_swd_max_abs
    if set(month_swd_difference) != {'201312', *(f'2014{m:02d}' for m in range(1, 13))}:
        raise AssertionError('incorrect bootstrap or campaign months')
    if max(month_swd_difference.values()) <= 0:
        raise AssertionError('expected duplicate SWD discrepancy was hidden')
    report = {'status': 'passed_2014_weather_boundary_scan',
              'scope': 'Cabauw 2014 365-day private physical forcing coverage only; not greenhouse trajectory validation or blind year',
              'origin_utc': origin.isoformat(),
              'queries': 365 * 288 + 1,
              'selected_swd_product': 'cesar_surface_radiation_lc1_t10',
              'monthly_max_duplicate_swd_difference_w_m2': month_swd_difference,
              'plan_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
              'audit_sha256': hashlib.sha256(audit.read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'queries': report['queries']}))


if __name__ == '__main__':
    main()
