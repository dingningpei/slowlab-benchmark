#!/usr/bin/env python3
"""Audit formal weather under repair rule v0 (slowlab/weather_repair.py).

--months M ...  audits single months, each repaired on its own (the sample gate).
--years Y ...   audits complete campaign years: the previous December plus twelve
                months form one series per channel, so repairs may use records
                across month edges.

Every file must match the approved list and the download manifest. Time grid,
time origin and daily coverage failures stop the audit. Invalid records in the
five channels the model uses are handled by rule v0; each repair is written
out and the formal reader applies exactly those values. Invalid records in
other stored fields (relative humidity, duplicate meteo shortwave, source
indices) are counted but not repaired: the model does not use them. The 2014-12
boundary month for 2015 comes from the verified development cache.
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
sys.path.insert(0, str(ROOT))
from slowlab.cabauw_weather import FIELDS, MET, RAD, _read  # noqa: E402
from slowlab.weather_repair import MAX_REPAIRS_PER_MONTH, MAX_RUN, REQUIRED, repair_months  # noqa: E402

PLAN = ROOT / 'configs/weather_formal_acquisition_proposal_v0.json'
DEV_BOUNDARY = '201412'


def approved_files():
    stage = json.loads(PLAN.read_text())['stages']['recommended']
    return {r['filename']: r for r in stage['files'] + stage['boundary_files']}


class Source:
    def __init__(self, cache: Path, dev_cache: Path | None):
        self.cache, self.dev_cache = cache, dev_cache
        self.rows = approved_files()
        self.manifest = json.loads((cache / 'manifest-formal-v0.json').read_text())['files']
        self.dev_manifest = (json.loads((dev_cache / 'manifest-v3-development.json').read_text())['files']
                             if dev_cache else {})

    def path(self, dataset, ym):
        name = f'{dataset}_v1.0_{ym}.nc'
        if ym == DEV_BOUNDARY:
            if name not in self.dev_manifest:
                raise ValueError('boundary month not in development manifest: ' + name)
            return self.dev_cache / dataset / name, self.dev_manifest[name]['sha256'], 'development cache'
        if name not in self.rows or name not in self.manifest:
            raise ValueError('file outside the approved, downloaded scope: ' + name)
        path = self.cache / dataset / name
        if path.stat().st_size != self.rows[name]['size']:
            raise ValueError('size mismatch: ' + name)
        return path, self.manifest[name]['sha256'], 'formal download'


def load_month(src: Source, ym: str, issues: list):
    year, month = int(ym[:4]), int(ym[4:])
    days = calendar.monthrange(year, month)[1]
    n = days * 144
    out, record = {}, {}
    for role, dataset in (('meteo', MET), ('radiation', RAD)):
        path, sha, origin = src.path(dataset, ym)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != sha:
            raise ValueError('SHA-256 mismatch: ' + path.name)
        arrays, units, fills = _read(path, {'time', 'time_bnds', 'valid_dates', *FIELDS[role]})
        b = arrays['time_bnds']
        if (len(arrays['time']) != n or b.shape != (n, 2)
                or units['time'] != f'hours since {ym[:4]}-{ym[4:]}-01 00:00:00 0:00'
                or abs(b[0, 0]) > 1e-6 or abs(b[-1, 1] - n / 6) > 1e-5
                or np.max(np.abs((b[:, 1] - b[:, 0]) * 3600 - 600)) > 1
                or np.max(np.abs((b[1:, 0] - b[:-1, 1]) * 3600)) > 1
                or np.max(np.abs((arrays['time'] - b[:, 0]) * 3600)) > 1):
            issues.append(f'{path.name}: time grid or time origin')
        if len(arrays['valid_dates']) != days or not np.all(arrays['valid_dates'] == 1):
            issues.append(f'{path.name}: daily coverage flags')
        other = {}
        for key in FIELDS[role]:
            if key in REQUIRED[role]:
                continue
            v = arrays[key]
            other[key] = int(np.sum(~np.isfinite(v) | (v == fills.get(key, np.nan))))
        out[role] = {'arrays': arrays, 'fills': fills}
        record[role] = {'filename': path.name, 'sha256': actual, 'origin': origin,
                        'unrepaired_other_fields_invalid': {k: c for k, c in other.items() if c}}
    if np.max(np.abs(out['meteo']['arrays']['time_bnds'] - out['radiation']['arrays']['time_bnds'])) * 3600 > 1:
        issues.append(f'{ym}: meteo/radiation time bounds differ')
    return out, record


def stats(months, repairs):
    fixed = {}
    for r in repairs:
        fixed[(r['month'], r['role'], r['channel'], r['index'])] = r['repaired']
    series = {}
    for role, channels in REQUIRED.items():
        for ch in channels:
            parts = []
            for ym, data in months:
                v = np.array(data[role]['arrays'][ch], dtype=np.float64)
                for (m, ro, c, i), value in fixed.items():
                    if m == ym and ro == role and c == ch:
                        v[i] = value
                parts.append(v)
            series[ch] = np.concatenate(parts)
    swd = np.maximum(series['SWD'], 0)
    return {'mean_air_temperature_c': float(series['TA002'].mean() - 273.15),
            'mean_dewpoint_c': float(series['TD002'].mean() - 273.15),
            'mean_wind_m_s': float(series['F010'].mean()),
            'shortwave_sum_mj_m2': float(swd.sum() * 600 / 1e6),
            'mean_longwave_w_m2': float(series['LWD'].mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--dev-cache', type=Path, default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--months', nargs='+')
    group.add_argument('--years', nargs='+', type=int)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    src = Source(args.cache, args.dev_cache)
    issues, units = [], {}
    if args.months:
        for ym in args.months:
            data, record = load_month(src, ym, issues)
            rep = repair_months([(ym, data)])
            units[ym] = {'files': record, 'repairs': rep['repairs'], 'problems': rep['problems'],
                         'status': 'excluded' if rep['problems'] else ('repaired' if rep['repairs'] else 'clean')}
    else:
        for year in args.years:
            yms = [f'{year - 1}12'] + [f'{year}{m:02d}' for m in range(1, 13)]
            loaded = [(ym, *load_month(src, ym, issues)) for ym in yms]
            months = [(ym, data) for ym, data, _ in loaded]
            rep = repair_months(months)
            units[str(year)] = {
                'status': 'excluded' if rep['problems'] else ('repaired' if rep['repairs'] else 'clean'),
                'problems': rep['problems'], 'repairs': rep['repairs'],
                'repairs_per_month': {k: v for k, v in rep['repairs_per_month'].items() if v},
                'files': {ym: record for ym, _, record in loaded},
                'summary_after_repair': stats(months[1:], rep['repairs'])}
    # Sample gate: every sample month must be usable under the rule. Year audit:
    # exclusions are reported outcomes; only file-level issues fail the gate.
    usable = bool(args.years) or all(u['status'] != 'excluded' for u in units.values())
    report = {'gate': 'pass' if not issues and usable else 'fail',
              'rule': {'id': 'weather-repair-v0', 'max_run_records': MAX_RUN,
                       'max_repairs_per_month': MAX_REPAIRS_PER_MONTH, 'channels': REQUIRED},
              'scope': ('sample months, each repaired alone' if args.months
                        else 'complete campaign years: previous December plus twelve months'),
              'units': units, 'issues': issues,
              'plan_sha256': hashlib.sha256(PLAN.read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'gate': report['gate'], 'issues': issues,
                      'status': {k: u['status'] for k, u in units.items()},
                      'repairs': {k: len(u['repairs']) for k, u in units.items()}}))
    if report['gate'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
