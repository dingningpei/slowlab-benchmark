import calendar
import copy
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pytest
from scipy.io import netcdf_file

from slowlab.cabauw_weather import FIELDS, MET, RAD, CabauwLc1FormalWeather
from slowlab.weather_repair import repair_months

YEAR = 2003
MONTHS = [f'{YEAR - 1}12'] + [f'{YEAR}{m:02d}' for m in range(1, 13)]
UNITS = {'TA002': 'degC', 'TD002': 'degC', 'RH002': '1e-2', 'F010': 'm s-1', 'SWD': 'W m-2', 'LWD': 'W m-2'}


def write_month(folder, dataset, role, ym, corrupt=None):
    days = calendar.monthrange(int(ym[:4]), int(ym[4:]))[1]
    n = days * 144
    path = folder / dataset / f'{dataset}_v1.0_{ym}.nc'
    path.parent.mkdir(parents=True, exist_ok=True)
    hours = np.arange(n, dtype=np.float32) / np.float32(6)
    values = {'TA002': np.full(n, 280.0), 'TD002': np.full(n, 275.0), 'RH002': np.full(n, 80.0),
              'F010': np.full(n, 5.0), 'SWD': np.full(n, 100.0), 'LWD': np.full(n, 300.0)}
    if corrupt:
        for key, index, value in corrupt:
            values[key][index] = value
    with netcdf_file(path, 'w') as f:
        f.createDimension('time', n)
        f.createDimension('nv', 2)
        f.createDimension('day', days)
        t = f.createVariable('time', 'f4', ('time',))
        t[:] = hours
        t.units = f'hours since {ym[:4]}-{ym[4:]}-01 00:00:00 0:00'
        b = f.createVariable('time_bnds', 'f4', ('time', 'nv'))
        b[:] = np.stack((hours, hours + np.float32(1 / 6)), axis=1)
        d = f.createVariable('valid_dates', 'i4', ('day',))
        d[:] = 1
        for key in FIELDS[role]:
            v = f.createVariable(key, 'f4', ('time',))
            v[:] = np.ones(n) if key.startswith('I') else values[key]
            v._FillValue = np.float32(-9999.0)
            if key in UNITS:
                v.units = UNITS[key]
    return path


@pytest.fixture
def year(tmp_path):
    files, loaded = {}, []
    for ym in MONTHS:
        corrupt = [('SWD', 100, -93.9)] if ym == f'{YEAR}03' else None
        rec, data = {}, {}
        for role, dataset in (('meteo', MET), ('radiation', RAD)):
            path = write_month(tmp_path, dataset, role, ym, corrupt if role == 'radiation' else None)
            rec[role] = {'filename': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'origin': 'formal download'}
            with netcdf_file(path, mmap=False) as f:
                data[role] = {'arrays': {k: np.asarray(f.variables[k][:], dtype=float) for k in FIELDS[role]},
                              'fills': {k: -9999.0 for k in FIELDS[role]}}
        files[ym] = rec
        loaded.append((ym, data))
    rep = repair_months(loaded)
    assert rep['problems'] == [] and len(rep['repairs']) == 1
    audit = {'gate': 'pass', 'rule': {'id': 'weather-repair-v0'},
             'scope': 'complete campaign years: previous December plus twelve months',
             'units': {str(YEAR): {'status': 'repaired', 'files': files, 'repairs': rep['repairs'], 'problems': []}}}
    (tmp_path / 'audit.json').write_text(json.dumps(audit))
    return tmp_path, audit


def at(reader, month, index):
    start = datetime(int(month[:4]), int(month[4:]), 1, tzinfo=timezone.utc)
    return reader.at_utc(datetime.fromtimestamp(start.timestamp() + index * 600, tz=timezone.utc))


def test_reader_serves_the_audited_repair(year):
    folder, _ = year
    w = CabauwLc1FormalWeather(folder, folder / 'audit.json', YEAR)
    assert at(w, f'{YEAR}03', 100).swd_raw_w_m2 == pytest.approx(100.0)
    assert at(w, f'{YEAR}03', 101).swd_raw_w_m2 == pytest.approx(100.0)
    first = w.at_utc(datetime(YEAR - 1, 12, 31, 23, tzinfo=timezone.utc))
    assert first.t_out_c == pytest.approx(280.0 - 273.15, abs=1e-4)
    with pytest.raises(ValueError, match='outside formal campaign year'):
        w.at_utc(datetime(YEAR + 1, 1, 1, tzinfo=timezone.utc))


def test_reader_without_the_repair_rejects_the_month(year):
    folder, audit = year
    broken = copy.deepcopy(audit)
    broken['units'][str(YEAR)]['repairs'] = []
    (folder / 'broken.json').write_text(json.dumps(broken))
    w = CabauwLc1FormalWeather(folder, folder / 'broken.json', YEAR)
    at(w, f'{YEAR}02', 0)
    with pytest.raises(ValueError, match='invalid SWD after audited repairs'):
        at(w, f'{YEAR}03', 0)


def test_reader_rejects_tampered_files_and_excluded_years(year):
    folder, audit = year
    path = folder / RAD / f'{RAD}_v1.0_{YEAR}05.nc'
    path.write_bytes(path.read_bytes()[:-1] + b'\x00')
    w = CabauwLc1FormalWeather(folder, folder / 'audit.json', YEAR)
    with pytest.raises(ValueError, match='hash mismatch'):
        at(w, f'{YEAR}05', 0)
    excluded = copy.deepcopy(audit)
    excluded['units'][str(YEAR)]['status'] = 'excluded'
    (folder / 'excluded.json').write_text(json.dumps(excluded))
    with pytest.raises(ValueError, match='excluded'):
        CabauwLc1FormalWeather(folder, folder / 'excluded.json', YEAR)
    with pytest.raises(ValueError, match='not in the formal audit'):
        CabauwLc1FormalWeather(folder, folder / 'audit.json', YEAR + 1)
