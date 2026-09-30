#!/usr/bin/env python3
"""Fail-closed source, time and channel audit of the formal-weather sample months.

Same checks as the development sample audit (scripts/audit_weather_v3_development_sample.py):
listed size and manifest hash, 10-minute time bounds, daily coverage flags,
missing or fill values in every required channel, matching meteo/radiation
bounds, and physical support. Adds per-channel summary statistics so older
years can be compared with the development years.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from audit_weather_v3_development_sample import load  # noqa: E402

CHANNELS = (('meteo', 'cesar_surface_meteo_lc1_t10', ('TA002', 'TD002', 'F010', 'SWD')),
            ('radiation', 'cesar_surface_radiation_lc1_t10', ('SWD', 'LWD')))


def audit_month(ym, rows, manifest, cache, issues, cautions):
    year, month = int(ym[:4]), int(ym[4:])
    days = calendar.monthrange(year, month)[1]
    expected = days * 144
    pair = {}
    for role, dataset, channels in CHANNELS:
        name = f'{dataset}_v1.0_{ym}.nc'
        path = cache / dataset / name
        entry = manifest['files'].get(name)
        if entry is None or not path.is_file() or path.stat().st_size != rows[name]['size']:
            raise ValueError('missing or size-mismatched sample: ' + name)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != entry['sha256']:
            raise ValueError('sample SHA-256 mismatch: ' + name)
        arrays, fills, fmt = load(path, ('time', 'time_bnds', 'valid_dates', *channels))
        b = arrays['time_bnds']
        if (len(arrays['time']) != expected or b.shape != (expected, 2)
                or abs(b[0, 0]) * 3600 > 1 or abs(b[-1, 1] - expected / 6) * 3600 > 1
                or np.max(np.abs((b[:, 1] - b[:, 0]) * 3600 - 600)) > 1
                or np.max(np.abs((b[1:, 0] - b[:-1, 1]) * 3600)) > 1
                or np.max(np.abs((arrays['time'] - b[:, 0]) * 3600)) > 1):
            issues.append(f'{name}: invalid 10-minute time bounds')
        if len(arrays['valid_dates']) != days or not np.all(arrays['valid_dates'] == 1):
            issues.append(f'{name}: invalid daily coverage flags')
        invalid, stats = {}, {}
        for ch in channels:
            v = arrays[ch]
            bad = (~np.isfinite(v)) | (v == fills.get(ch, np.nan))
            invalid[ch] = int(bad.sum())
            if len(v) != expected or invalid[ch]:
                issues.append(f'{name}: {ch} missing/shape ({invalid[ch]} invalid rows)')
            good = v[~bad]
            stats[ch] = {'min': float(good.min()), 'mean': float(good.mean()), 'max': float(good.max())} if good.size else None
        pair[role] = {'filename': name, 'sha256': sha, 'format': fmt, 'rows': expected,
                      'invalid_by_channel': invalid, 'stats': stats, '_b': b, '_a': arrays}
    if np.max(np.abs(pair['meteo']['_b'] - pair['radiation']['_b'])) * 3600 > 1:
        issues.append(f'{ym}: meteo/radiation time bounds differ')
    met, rad = pair['meteo']['_a'], pair['radiation']['_a']
    diff = np.abs(met['SWD'] - rad['SWD'])
    differing = int(np.sum(diff > 1e-6))
    if differing:
        cautions.append(f'{ym}: meteo and radiation SWD differ in {differing} rows; use radiation SWD only')
    if (np.nanmin(met['TA002']) < 230 or np.nanmax(met['TA002']) > 330
            or np.nanmin(met['TD002']) < 220 or np.nanmax(met['TD002']) > 330
            or np.nanmin(met['F010']) < 0 or np.nanmax(met['F010']) > 70
            or np.nanmin(rad['SWD']) < -20 or np.nanmax(rad['SWD']) > 1500
            or np.nanmin(rad['LWD']) < 100 or np.nanmax(rad['LWD']) > 600):
        issues.append(f'{ym}: required physical support exceeded')
    out = {role: {k: v for k, v in r.items() if not k.startswith('_')} for role, r in pair.items()}
    out['swd_product_difference'] = {'rows_above_1e_minus_6': differing, 'max_abs_w_m2': float(np.max(diff)),
                                     'selected_source': 'radiation_lc1_t10.SWD'}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--months', nargs='+', default=['200201', '201501'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan_path = ROOT / 'configs/weather_formal_acquisition_proposal_v0.json'
    stage = json.loads(plan_path.read_text())['stages']['recommended']
    rows = {r['filename']: r for r in stage['files'] + stage['boundary_files']}
    manifest = json.loads((args.cache / 'manifest-formal-v0.json').read_text())
    issues, cautions = [], []
    months = {ym: audit_month(ym, rows, manifest, args.cache, issues, cautions) for ym in args.months}
    report = {'gate': 'pass' if not issues else 'fail',
              'scope': 'formal-weather sample months only; not annual completeness or greenhouse validation',
              'months': months, 'issues': issues, 'cautions': cautions,
              'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'gate': report['gate'], 'months': list(months), 'issues': issues, 'cautions': cautions}))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
