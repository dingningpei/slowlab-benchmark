#!/usr/bin/env python3
"""Verify the Phase-1 public-weather partition without reading NetCDF files."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))
from slowlab.v22.frozen_paths import frozen_path  # noqa: E402



def verify() -> dict:
    protocol = json.loads((ROOT / 'configs/v22/weather_partition_v0.json').read_text())
    if protocol['status'] != 'phase1_source_weather_partition_not_confirmatory_preregistration':
        raise ValueError('wrong source-weather protocol status')
    source = {}
    for name, hash_name in ((protocol['source_plan'], 'source_plan_sha256'),
                            (protocol['quality_audit'], 'quality_audit_sha256')):
        path = frozen_path(ROOT, name)
        if path.resolve().parent not in {(ROOT / 'configs' / 'v22').resolve(),
                                         (ROOT / 'results' / 'v22').resolve()}:
            raise ValueError('source path must be a frozen v2.2 configs or results file')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != protocol[hash_name]:
            raise ValueError('source audit or file plan changed')
        source[hash_name] = json.loads(raw)
    plan = source['source_plan_sha256']
    audit = source['quality_audit_sha256']
    groups = protocol['groups']
    if (groups != {'development_and_generator_fit': [2017, 2018],
                   'generator_selection': [2019],
                   'out_of_year_weather_diagnostic': [2020]}
            or set(protocol['windows']) != {'2017', '2018', '2019', '2020'}):
        raise ValueError('unexpected or overlapping partition')
    if (plan['proposed_years'] != [2017, 2018, 2019, 2020]
            or plan['boundary_month'] != '201612'
            or plan['file_count'] != 98 or plan['total_bytes'] != 18_413_090
            or audit['status'] != 'audited'
            or audit['total_bytes'] != plan['total_bytes']
            or len(audit['months']) != 48):
        raise ValueError('source inventory or audit changed')
    source_files = {f['filename'] for f in plan['files']}
    expected = {f'cesar_surface_{kind}_lc1_t10_v1.0_{year}{month:02d}.nc'
                for kind in ('meteo', 'radiation')
                for year, months in [(2016, (12,)),
                                     *((year, range(1, 13)) for year in range(2017, 2021))]
                for month in months}
    if source_files != expected:
        raise ValueError('missing or unexpected source month')
    if protocol['campaign_days'] != 365 or protocol['tick_seconds'] != 300 or protocol['weather_interval_seconds'] != 600:
        raise ValueError('clock contract changed')
    intervals = []
    for year in range(2017, 2021):
        qa = audit['years'][str(year)]
        expected_rows = (366 if year == 2020 else 365) * 144
        if qa['rows'] != expected_rows or qa['joint_invalid'] != 0:
            raise ValueError(f'incomplete {year} source weather')
        window = protocol['windows'][str(year)]
        start = datetime.fromisoformat(window['start_utc'].replace('Z', '+00:00'))
        end = datetime.fromisoformat(window['end_utc_exclusive'].replace('Z', '+00:00'))
        if (start != datetime(year - 1, 12, 31, 23, tzinfo=timezone.utc)
                or end - start != timedelta(days=365)
                or start.utcoffset() != timedelta(0)):
            raise ValueError(f'invalid {year} campaign window')
        intervals.append((start, end))
    if any(left[1] != right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError('campaign windows are not contiguous')
    if intervals[-1][1] != datetime(2020, 12, 30, 23, tzinfo=timezone.utc):
        raise ValueError('2020 leap-year exclusion drifted')
    return {'status': 'passed_source_weather_partition_v0',
            'years': list(range(2017, 2021)),
            'campaign_days_per_year': 365,
            'source_files': len(source_files),
            'source_10min_rows': sum(audit['years'][str(y)]['rows'] for y in range(2017, 2021)),
            'confirmatory_weather_generated': False}


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
