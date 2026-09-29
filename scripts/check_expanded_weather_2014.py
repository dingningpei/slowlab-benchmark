#!/usr/bin/env python3
"""Verify an already-audited second historical forcing year without simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab.cabauw_weather import CabauwLc1ExpandedDevelopmentWeather


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--matrix', type=Path,
                        help='Previously built independent 2014 five-channel development matrix')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = ROOT / 'configs/weather_v3_expanded_acquisition_proposal.json'
    audit = ROOT / 'results/weather_v3_development_full_audit.json'
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
    matrix_comparison = None
    if args.matrix is not None:
        import numpy as np
        metadata = json.loads((ROOT / 'results/weather_v3_complete_years_metadata.json').read_text())
        raw = args.matrix.read_bytes()
        if hashlib.sha256(raw).hexdigest() != metadata['archive_sha256']:
            raise ValueError('independently built matrix archive hash mismatch')
        with np.load(args.matrix) as archive:
            matrix = archive['2014']
        if (matrix.shape != (365 * 144, 5)
                or hashlib.sha256(matrix.astype('<f8').tobytes()).hexdigest()
                != metadata['years']['2014']['sha256_f64_le']):
            raise ValueError('2014 matrix identity mismatch')
        january = datetime(2014, 1, 1, tzinfo=timezone.utc)
        maximum = 0.0
        for index, row in enumerate(matrix):
            forcing = weather.at_utc(january + timedelta(seconds=index * 600))
            expected = (row[0],
                        610.78 * math.exp(17.2694 * row[1] / (row[1] + 238.3)),
                        row[2], row[3], row[4])
            actual = (forcing.t_out_c, forcing.vp_out_pa, forcing.wind_m_s,
                      forcing.i_glob_w_m2, forcing.lwd_w_m2)
            maximum = max(maximum, *(abs(float(a) - float(b)) for a, b in zip(actual, expected)))
        if maximum > 1e-9:
            raise AssertionError(f'2014 reader differs from independent matrix: {maximum}')
        matrix_comparison = {'rows': 365 * 144, 'max_abs_difference': maximum,
                             'archive_sha256': metadata['archive_sha256']}
    report = {'status': 'passed_2014_weather_boundary_scan',
              'scope': 'Cabauw 2014 365-day private physical forcing coverage only; not greenhouse trajectory validation or blind year',
              'origin_utc': origin.isoformat(),
              'queries': 365 * 288 + 1,
              'selected_swd_product': 'cesar_surface_radiation_lc1_t10',
              'monthly_max_duplicate_swd_difference_w_m2': month_swd_difference,
              'independent_matrix_comparison': matrix_comparison,
              'plan_sha256': hashlib.sha256(plan.read_bytes()).hexdigest(),
              'audit_sha256': hashlib.sha256(audit.read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'queries': report['queries']}))


if __name__ == '__main__':
    main()
