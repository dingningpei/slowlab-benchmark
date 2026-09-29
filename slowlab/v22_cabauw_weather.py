"""Strict private-executor weather reader for KNMI Cabauw lc1 v1.0.

This parses archived forcing; it is not an agent observation interface and does
not establish greenhouse or actuator validity. No formal train/test split here.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

MET = 'cesar_surface_meteo_lc1_t10'
RAD = 'cesar_surface_radiation_lc1_t10'
FIELDS = {'meteo': ('TA002', 'TD002', 'RH002', 'F010', 'SWD', 'ITA002', 'IQ002', 'IF010', 'ISWD'),
          'radiation': ('SWD', 'LWD', 'ISWD', 'ILWD')}
SOURCE_CODES = frozenset((1, 2, 3, 5, 6, 7))
SIGMA = 5.670374419e-8


@dataclass(frozen=True)
class WeatherForcing:
    interval_start_utc: datetime
    interval_end_utc: datetime
    t_out_c: float
    vp_out_pa: float
    wind_m_s: float
    t_sky_c: float
    i_glob_w_m2: float
    swd_raw_w_m2: float
    swd_correction_w_m2: float
    lwd_w_m2: float
    rh_percent_qa: float
    source_indices: dict[str, int]

    def greenlight_inputs(self) -> dict[str, float]:
        return {'tOut': self.t_out_c, 'vpOut': self.vp_out_pa,
                'wind': self.wind_m_s, 'tSky': self.t_sky_c,
                'iGlob': self.i_glob_w_m2}


def _read(path: Path, keys: set[str]):
    with path.open('rb') as stream:
        magic = stream.read(4)
    if magic == b'\x89HDF':
        import h5py
        with h5py.File(path) as file:
            missing = keys - set(file)
            if missing:
                raise ValueError(f'{path.name}: missing {sorted(missing)}')
            data = {key: np.asarray(file[key][:], dtype=np.float64) for key in keys}
            units = {key: file[key].attrs.get('units', b'') for key in keys}
            fills = {key: float(np.asarray(file[key].attrs['_FillValue']).item())
                     for key in keys if '_FillValue' in file[key].attrs}
    elif magic == b'CDF\x01':
        from scipy.io import netcdf_file
        with netcdf_file(path, mmap=False) as file:
            missing = keys - set(file.variables)
            if missing:
                raise ValueError(f'{path.name}: missing {sorted(missing)}')
            data = {key: np.asarray(file.variables[key][:], dtype=np.float64) for key in keys}
            units = {key: getattr(file.variables[key], 'units', b'') for key in keys}
            fills = {key: float(file.variables[key]._FillValue)
                     for key in keys if hasattr(file.variables[key], '_FillValue')}
    else:
        raise ValueError(f'{path.name}: unsupported NetCDF container')
    units = {key: value.decode(errors='replace') if isinstance(value, bytes) else str(value)
             for key, value in units.items()}
    return data, units, fills


class CabauwLc1Weather:
    """Load only the requested UTC month; fail closed on source/schema drift.

    The month record can be queried by the private executor for physical forcing.
    Ten-minute archived means are not real-time observations at interval start.
    """

    def __init__(self, cache: Path, plan: Path):
        self.cache = Path(cache)
        definition = json.loads(Path(plan).read_text())
        self.files = {(f['dataset'], f['filename'][-9:-3]): f for f in definition['files']}
        expected_months = {'201612', *(f'{year}{month:02d}' for year in range(2017, 2021)
                                      for month in range(1, 13))}
        expected_keys = {(dataset, ym) for dataset in (MET, RAD) for ym in expected_months}
        if (definition['total_bytes'] != 18_413_090 or len(definition['files']) != 98
                or set(self.files) != expected_keys):
            raise ValueError('unexpected Cabauw lc1 file plan')
        for (dataset, ym), entry in self.files.items():
            if entry['filename'] != f'{dataset}_v1.0_{ym}.nc' or entry['size'] <= 0:
                raise ValueError('unsafe or inconsistent Cabauw lc1 plan entry')
        manifest = json.loads((self.cache / 'manifest.json').read_text())
        if manifest['total_bytes_downloaded'] != definition['total_bytes'] or len(manifest['files']) != 98:
            raise ValueError('lc1 cache manifest incomplete')
        self.manifest = manifest['files']
        self._month_key = None
        self._month_data = None

    def _load_month(self, ym: str):
        data = {}
        for role, dataset in (('meteo', MET), ('radiation', RAD)):
            entry = self.files.get((dataset, ym))
            if entry is None:
                raise ValueError(f'outside audited lc1 scope: {ym}')
            path = self.cache / dataset / entry['filename']
            raw = path.read_bytes()
            expected = self.manifest[entry['filename']]
            if len(raw) != entry['size'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
                raise ValueError(f'lc1 cache hash/size mismatch: {path.name}')
            arrays, units, fills = _read(path, {'time', 'time_bnds', 'valid_dates', *FIELDS[role]})
            data[role] = arrays
            expected_days = calendar.monthrange(int(ym[:4]), int(ym[4:]))[1]
            n = expected_days * 144
            if len(arrays['time']) != n or arrays['time_bnds'].shape != (n, 2):
                raise ValueError(f'{path.name}: incomplete time grid')
            expected_time_units = f'hours since {ym[:4]}-{ym[4:]}-01 00:00:00 0:00'
            if units['time'] != expected_time_units:
                raise ValueError(f'{path.name}: unexpected UTC time origin')
            bounds = arrays['time_bnds']
            if np.max(abs(arrays['time'] - bounds[:, 0])) * 3600 > 1:
                raise ValueError(f'{path.name}: time coordinate is not interval start')
            if (abs(bounds[0, 0]) > 1e-6 or abs(bounds[-1, 1] - n / 6) > 1e-5
                    or np.max(abs((bounds[:, 1] - bounds[:, 0]) * 3600 - 600)) > 1
                    or np.max(abs((bounds[1:, 0] - bounds[:-1, 1]) * 3600)) > 1):
                raise ValueError(f'{path.name}: discontinuous time bounds')
            if len(arrays['valid_dates']) != expected_days or not np.all(arrays['valid_dates'] == 1):
                raise ValueError(f'{path.name}: invalid date coverage')
            for key in FIELDS[role]:
                values = arrays[key]
                if len(values) != n or not np.all(np.isfinite(values)) or np.any(values == fills.get(key, np.nan)):
                    raise ValueError(f'{path.name}: missing/nonfinite {key}')
            expected_units = {'TA002': 'degC', 'TD002': 'degC', 'RH002': '1e-2',
                              'F010': 'm s-1', 'SWD': 'W m-2', 'LWD': 'W m-2'}
            for key in FIELDS[role]:
                if key in expected_units and units[key] != expected_units[key]:
                    raise ValueError(f'{path.name}: {key} unit-label drift: {units[key]}')
            # Explicit v1.0 exception: TA002/TD002 are labeled degC while all
            # audited values are Kelvin-scale and match lb1 K anchors.
            if role == 'meteo':
                if not (np.all((arrays['TA002'] > 230) & (arrays['TA002'] < 330))
                        and np.all((arrays['TD002'] > 220) & (arrays['TD002'] < 330))
                        and np.all((arrays['RH002'] >= 0) & (arrays['RH002'] < 120))
                        and np.all((arrays['F010'] >= 0) & (arrays['F010'] < 70))):
                    raise ValueError(f'{path.name}: v1.0 value anchor failed')
            else:
                if not (np.all((arrays['SWD'] > -10) & (arrays['SWD'] < 1500))
                        and np.all((arrays['LWD'] > 100) & (arrays['LWD'] < 600))):
                    raise ValueError(f'{path.name}: radiation value anchor failed')
            for key in FIELDS[role]:
                if key.startswith('I'):
                    values = arrays[key]
                    if not np.all(values == values.astype(int)) or not set(values.astype(int)).issubset(SOURCE_CODES):
                        raise ValueError(f'{path.name}: unknown source index {key}')
        if np.max(abs(data['meteo']['time_bnds'] - data['radiation']['time_bnds'])) * 3600 > 1:
            raise ValueError(f'{ym}: meteo/radiation time mismatch')
        if not np.array_equal(data['meteo']['SWD'], data['radiation']['SWD']):
            raise ValueError(f'{ym}: inconsistent duplicate SWD channels')
        self._month_key, self._month_data = ym, data

    def at_utc(self, when: datetime) -> WeatherForcing:
        if when.tzinfo is None or when.utcoffset() != timedelta(0):
            raise ValueError('query requires timezone-aware UTC datetime')
        ym = when.strftime('%Y%m')
        if self._month_key != ym:
            self._load_month(ym)
        start = datetime(when.year, when.month, 1, tzinfo=timezone.utc)
        elapsed = (when - start).total_seconds()
        index = math.floor(elapsed / 600)
        data = self._month_data
        met, rad = data['meteo'], data['radiation']
        if index < 0 or index >= len(met['TA002']):
            raise ValueError('query outside monthly grid')
        interval_start = start + timedelta(seconds=index * 600)
        temp_c = float(met['TA002'][index] - 273.15)
        dew_c = float(met['TD002'][index] - 273.15)
        vp_out = 610.78 * math.exp(17.2694 * dew_c / (dew_c + 238.3))
        swd = float(rad['SWD'][index]);lwd = float(rad['LWD'][index])
        if not math.isfinite(vp_out) or vp_out <= 0:
            raise ValueError('derived outdoor vapor pressure invalid')
        indices = {'TA002': int(met['ITA002'][index]),
                   'humidity': int(met['IQ002'][index]),
                   'F010': int(met['IF010'][index]),
                   'SWD': int(rad['ISWD'][index]),
                   'LWD': int(rad['ILWD'][index])}
        return WeatherForcing(interval_start, interval_start + timedelta(seconds=600),
                              temp_c, vp_out, float(met['F010'][index]),
                              (lwd / SIGMA) ** 0.25 - 273.15, max(0.0, swd),
                              swd, max(0.0, -swd), lwd,
                              float(met['RH002'][index]), indices)
