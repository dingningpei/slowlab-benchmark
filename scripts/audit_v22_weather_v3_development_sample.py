#!/usr/bin/env python3
"""Fail-closed source/time/channel audit for approved v3 development sample."""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path, keys):
    with path.open('rb') as stream:
        magic = stream.read(4)
    if magic == b'\x89HDF':
        import h5py
        with h5py.File(path) as file:
            arrays = {key: np.asarray(file[key][:], dtype=np.float64) for key in keys}
            fills = {key: float(np.asarray(file[key].attrs['_FillValue']).item())
                     for key in keys if '_FillValue' in file[key].attrs}
        return arrays, fills, 'hdf5'
    if magic == b'CDF\x01':
        from scipy.io import netcdf_file
        with netcdf_file(path, mmap=False) as file:
            arrays = {key: np.asarray(file.variables[key][:], dtype=np.float64) for key in keys}
            fills = {key: float(file.variables[key]._FillValue)
                     for key in keys if hasattr(file.variables[key], '_FillValue')}
        return arrays, fills, 'netcdf3'
    raise ValueError('unsupported NetCDF container: ' + path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    manifest = json.loads(args.manifest.read_text())
    if set(plan['stages']['development']['years']) != {2012,2013,2014,2016}:
        raise ValueError('development year scope changed')
    rows = {row['filename']:row for row in plan['stages']['development']['files']}
    results = {}
    issues = []
    cautions = []
    for year in (2012,2013,2014,2016):
        ym = f'{year}01'
        pair = {}
        for role, dataset, channels in (
            ('meteo','cesar_surface_meteo_lc1_t10',('TA002','TD002','F010','SWD')),
            ('radiation','cesar_surface_radiation_lc1_t10',('SWD','LWD'))):
            filename = f'{dataset}_v1.0_{ym}.nc'
            row = rows[filename]
            path = args.cache / dataset / filename
            entry = manifest['files'].get(filename)
            if entry is None or not path.is_file() or path.stat().st_size != row['size']:
                raise ValueError('missing or size-mismatched sample: ' + filename)
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != entry['sha256']:
                raise ValueError('sample SHA-256 mismatch: ' + filename)
            arrays, fills, fmt = load(path, ('time','time_bnds','valid_dates',*channels))
            expected = calendar.monthrange(year,1)[1] * 144
            bounds = arrays['time_bnds']
            if (len(arrays['time']) != expected or bounds.shape != (expected,2)
                    or abs(bounds[0,0])*3600 > 1 or abs(bounds[-1,1] - expected/6)*3600 > 1
                    or np.max(np.abs((bounds[:,1]-bounds[:,0])*3600-600)) > 1
                    or np.max(np.abs((bounds[1:,0]-bounds[:-1,1])*3600)) > 1
                    or np.max(np.abs((arrays['time']-bounds[:,0])*3600)) > 1):
                issues.append(f'{filename}: invalid 10-minute time bounds')
            if len(arrays['valid_dates']) != 31 or not np.all(arrays['valid_dates'] == 1):
                issues.append(f'{filename}: invalid daily coverage flags')
            invalid = {}
            for name in channels:
                values = arrays[name]
                fill = fills.get(name, np.nan)
                invalid[name] = int(np.sum((~np.isfinite(values)) | (values == fill)))
                if len(values) != expected or invalid[name]:
                    issues.append(f'{filename}: {name} missing/shape')
            pair[role] = {'filename':filename,'sha256':actual_hash,'format':fmt,
                          'rows':expected,'joint_invalid_by_channel':invalid,
                          'bounds':bounds,'arrays':arrays}
        if np.max(np.abs(pair['meteo']['bounds']-pair['radiation']['bounds']))*3600 > 1:
            issues.append(f'{ym}: meteo/radiation time bounds differ')
        met = pair['meteo']['arrays']
        rad = pair['radiation']['arrays']
        swd_difference = np.abs(met['SWD']-rad['SWD'])
        differing_swd = int(np.sum(swd_difference > 1e-6))
        if differing_swd:
            cautions.append(f'{ym}: meteo and radiation SWD differ in {differing_swd} rows; use radiation SWD only')
        if (np.min(met['TA002']) < 230 or np.max(met['TA002']) > 330
                or np.min(met['TD002']) < 220 or np.max(met['TD002']) > 330
                or np.min(met['F010']) < 0 or np.max(met['F010']) > 70
                or np.min(rad['SWD']) < -20 or np.max(rad['SWD']) > 1500
                or np.min(rad['LWD']) < 100 or np.max(rad['LWD']) > 600):
            issues.append(f'{ym}: required physical support exceeded')
        results[ym] = {role:{key:value for key,value in row.items()
                              if key not in ('bounds','arrays')}
                       for role,row in pair.items()}
        results[ym]['swd_product_difference'] = {
            'rows_above_1e_minus_6': differing_swd,
            'max_abs_w_m2': float(np.max(swd_difference)),
            'selected_source': 'radiation_lc1_t10.SWD',
        }
    report = {'gate':'pass' if not issues else 'fail',
              'scope':'four January sample pairs only; not annual completeness or greenhouse validation',
              'months':results,'issues':issues,'cautions':cautions,
              'plan_sha256':hashlib.sha256(args.plan.read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'gate':report['gate'],'sample_pairs':len(results),'issues':issues}))
    if issues:
        raise SystemExit(1)


if __name__=='__main__':
    main()
