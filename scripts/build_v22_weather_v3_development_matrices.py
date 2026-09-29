#!/usr/bin/env python3
"""Materialize audited development-only weather matrices for v3 model fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_v22_weather_v3_development_sample import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    args = parser.parse_args()
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)['stages']['development']
    audit = json.loads(args.audit.read_text())
    years = (2013, 2014, 2016)
    if (audit['gate'] != 'fail'
            or audit['complete_years'] != list(years)
            or set(audit['incomplete_years']) != {'2012'}
            or audit['plan_sha256'] != hashlib.sha256(plan_bytes).hexdigest()
            or set(plan['years']) != {2012, *years}
            or len(audit['months']) != 48):
        raise ValueError('audited complete-year subset changed')
    by_identity = {(r['dataset'], r['filename'][-9:-3]): r for r in plan['files']}
    matrices = {}
    metadata = {'scope': 'audited complete Cabauw development years only; no greenhouse validation',
                'source_selection': {'temperature/dewpoint/wind': 'meteo_lc1_t10',
                                     'shortwave/longwave': 'radiation_lc1_t10'},
                'leap_rule': 'drop Feb 29 for 365-day generator fitting only',
                'excluded_years': audit['incomplete_years'],
                'audit_sha256': hashlib.sha256(args.audit.read_bytes()).hexdigest(),
                'years': {}}
    for year in years:
        pieces = []
        for month in range(1, 13):
            ym = f'{year}{month:02d}'
            values = {}
            for role, dataset, fields in (
                    ('meteo', 'cesar_surface_meteo_lc1_t10', ('TA002', 'TD002', 'F010')),
                    ('radiation', 'cesar_surface_radiation_lc1_t10', ('SWD', 'LWD'))):
                row = by_identity[dataset, ym]
                path = args.cache / dataset / row['filename']
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != audit['months'][ym][role]['sha256']:
                    raise ValueError('weather data changed since full audit: ' + path.name)
                values[role] = load(path, fields)[0]
            met, rad = values['meteo'], values['radiation']
            pieces.append(np.column_stack((met['TA002'] - 273.15,
                                           met['TD002'] - 273.15,
                                           met['F010'], np.maximum(rad['SWD'], 0),
                                           rad['LWD'])))
        path = np.concatenate(pieces)
        if year == 2016:
            path = np.concatenate((path[:59 * 144], path[60 * 144:]))
        if path.shape != (365 * 144, 5) or not np.isfinite(path).all():
            raise ValueError('invalid annual weather matrix: ' + str(year))
        matrices[str(year)] = path
        metadata['years'][str(year)] = {'rows': len(path),
                                        'sha256_f64_le': hashlib.sha256(
                                            path.astype('<f8').tobytes()).hexdigest()}
    np.savez_compressed(args.out, **matrices)
    metadata['archive_sha256'] = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.metadata.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'years': list(matrices), 'archive_bytes': args.out.stat().st_size,
                      'archive_sha256': metadata['archive_sha256']}))


if __name__ == '__main__':
    main()
