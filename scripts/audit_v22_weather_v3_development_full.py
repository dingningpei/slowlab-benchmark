#!/usr/bin/env python3
"""Audit every authorized KNMI lc1 development month before fitting v3 weather."""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_v22_weather_v3_development_sample import load


MET = 'cesar_surface_meteo_lc1_t10'
RAD = 'cesar_surface_radiation_lc1_t10'
FIELDS = {
    MET: ('TA002', 'TD002', 'RH002', 'F010', 'SWD', 'ITA002', 'IQ002', 'IF010', 'ISWD'),
    RAD: ('SWD', 'LWD', 'ISWD', 'ILWD'),
}
SOURCE_CODES = {1, 2, 3, 5, 6, 7}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())['stages']['development']
    manifest = json.loads(args.manifest.read_text())
    rows = {(r['dataset'], r['filename'][-9:-3]): r for r in plan['files']}
    years = (2012, 2013, 2014, 2016)
    if (set(plan['years']) != set(years) or len(rows) != 96
            or set(manifest['files']) != {r['filename'] for r in plan['files']}
            or manifest['net_new_bytes_downloaded'] != plan['net_new_bytes_cap']):
        raise ValueError('development acquisition incomplete or scope changed')
    issues, cautions, months = [], [], {}
    for year in years:
        for month in range(1, 13):
            ym = f'{year}{month:02d}'
            n = calendar.monthrange(year, month)[1] * 144
            pair = {}
            report = {}
            for role, dataset in (('meteo', MET), ('radiation', RAD)):
                row = rows[dataset, ym]
                filename = row['filename']
                path = args.cache / dataset / filename
                entry = manifest['files'][filename]
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                if len(raw) != row['size'] or digest != entry['sha256']:
                    raise ValueError('size/hash mismatch: ' + filename)
                arrays, fills, fmt = load(path, ('time', 'time_bnds', 'valid_dates', *FIELDS[dataset]))
                bounds = arrays['time_bnds']
                if (len(arrays['time']) != n or bounds.shape != (n, 2)
                        or abs(bounds[0, 0]) * 3600 > 1
                        or abs(bounds[-1, 1] - n / 6) * 3600 > 1
                        or np.max(np.abs((bounds[:, 1] - bounds[:, 0]) * 3600 - 600)) > 1
                        or np.max(np.abs((bounds[1:, 0] - bounds[:-1, 1]) * 3600)) > 1
                        or np.max(np.abs((arrays['time'] - bounds[:, 0]) * 3600)) > 1):
                    issues.append(filename + ': invalid ten-minute time grid')
                if (len(arrays['valid_dates']) != n // 144
                        or not np.all(arrays['valid_dates'] == 1)):
                    issues.append(filename + ': incomplete daily flags')
                missing = {}
                source_indices = {}
                for key in FIELDS[dataset]:
                    values = arrays[key]
                    missing[key] = int(np.sum(~np.isfinite(values) | (values == fills.get(key, np.nan))))
                    if len(values) != n or missing[key]:
                        issues.append(filename + ': invalid/missing ' + key)
                    if key.startswith('I'):
                        source_indices[key] = sorted(set(values.astype(int).tolist()))
                        if not set(source_indices[key]).issubset(SOURCE_CODES):
                            issues.append(filename + ': unknown source code ' + key)
                if dataset == MET:
                    limits = {'TA002': (230, 330), 'TD002': (220, 330),
                              'RH002': (0, 120), 'F010': (0, 70), 'SWD': (-20, 1500)}
                else:
                    limits = {'SWD': (-20, 1500), 'LWD': (100, 600)}
                extrema = {}
                for key, (low, high) in limits.items():
                    values = arrays[key]
                    extrema[key] = [float(np.min(values)), float(np.max(values))]
                    if not np.all((values >= low) & (values <= high)):
                        issues.append(filename + ': outside physical support ' + key)
                pair[role] = arrays
                report[role] = {'filename': filename, 'sha256': digest, 'format': fmt,
                                'rows': n, 'missing_by_channel': missing,
                                'source_codes': source_indices, 'extrema': extrema}
            bound_difference = float(np.max(np.abs(
                pair['meteo']['time_bnds'] - pair['radiation']['time_bnds'])) * 3600)
            if bound_difference > 1:
                issues.append(ym + ': product time bounds diverge by >1s')
            swd_difference = np.abs(pair['meteo']['SWD'] - pair['radiation']['SWD'])
            n_different = int(np.sum(swd_difference > 1e-6))
            if n_different:
                cautions.append(ym + ': duplicate SWD differs; radiation product selected')
            report['cross_product'] = {
                'max_bound_difference_seconds': bound_difference,
                'swd_rows_different': n_different,
                'max_swd_difference_w_m2': float(np.max(swd_difference)),
                'mean_abs_swd_difference_w_m2': float(np.mean(swd_difference)),
                'swd_rows_difference_over_25_w_m2': int(np.sum(swd_difference > 25)),
                'swd_rows_difference_over_100_w_m2': int(np.sum(swd_difference > 100)),
                'selected_swd': 'radiation_lc1_t10.SWD',
            }
            months[ym] = report
    result = {
        'gate': 'pass' if not issues else 'fail',
        'scope': '2012/2013/2014/2016 same-station v3 development years only; not greenhouse validation',
        'files': len(manifest['files']),
        'net_new_bytes': manifest['net_new_bytes_downloaded'],
        'plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        'months': months,
        'issues': issues,
        'cautions': cautions,
        'complete_years': [year for year in years
                           if not any(str(year) in issue for issue in issues)],
        'incomplete_years': {str(year): [issue for issue in issues if str(year) in issue]
                             for year in years if any(str(year) in issue for issue in issues)},
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'gate': result['gate'], 'months': len(months),
                      'issues': issues[:12], 'issue_count': len(issues),
                      'cross_product_swd_months': len(cautions)}))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
