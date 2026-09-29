"""Candidate source-constrained private weather generator, not formal sites.

The generated five-channel 10-minute forcing must still pass the all-source
non-replay audit and 2019 development validation before any protocol freeze.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np

from ..v22.cabauw_weather import CabauwLc1Weather

ROWS_PER_DAY = 144
CHANNELS = ('outdoor_temperature_c', 'outdoor_dewpoint_c', 'wind_m_s',
            'shortwave_w_m2', 'downward_longwave_w_m2')


def read_source_year(cache: Path, plan: Path, year: int) -> np.ndarray:
    """Read only a declared lc1 year; this is private simulator preparation."""
    if year not in (2017, 2018, 2019, 2020):
        raise ValueError('year outside audited source partition')
    reader = CabauwLc1Weather(cache, plan)
    months = []
    for month in range(1, 13):
        reader._load_month(f'{year}{month:02d}')
        met = reader._month_data['meteo']
        rad = reader._month_data['radiation']
        matrix = np.column_stack((met['TA002'] - 273.15,
                                  met['TD002'] - 273.15,
                                  met['F010'], np.maximum(rad['SWD'], 0),
                                  rad['LWD']))
        if len(matrix) != calendar.monthrange(year, month)[1] * ROWS_PER_DAY:
            raise ValueError('source month length changed')
        months.append(matrix)
    result = np.concatenate(months)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite archived weather')
    return result


def _harmonics(days: int) -> np.ndarray:
    x = np.arange(days, dtype=np.float64) / 365.0
    return np.column_stack((np.ones(days),
                            np.sin(2 * np.pi * x), np.cos(2 * np.pi * x),
                            np.sin(4 * np.pi * x), np.cos(4 * np.pi * x)))


def _daily_features(matrix: np.ndarray) -> np.ndarray:
    daily = matrix.reshape(-1, ROWS_PER_DAY, 5)
    temp = daily[:, :, 0].mean(axis=1)
    spread = np.maximum(daily[:, :, 0] - daily[:, :, 1], 0).mean(axis=1)
    wind = daily[:, :, 2].mean(axis=1)
    solar = daily[:, :, 3].mean(axis=1)
    lwd = daily[:, :, 4].mean(axis=1)
    return np.column_stack((temp, np.log1p(spread), np.log1p(wind),
                            np.log1p(solar), lwd))


class CandidateWeatherGenerator:
    """Two-harmonic seasonal level + jointly sampled daily VAR innovations.

    High-frequency profiles are same-season analogues, transformed to new
    daily levels. No source outcome, crop state or agent feedback is used.
    """

    def __init__(self, fit_years: dict[int, np.ndarray], *, ridge: float = 0.2):
        if set(fit_years) != {2017, 2018} or not 0 < ridge < 10:
            raise ValueError('candidate fits only declared 2017-2018 weather')
        self.training = {}
        self.training_solar_daily = {}
        features = []
        for year in (2017, 2018):
            values = np.asarray(fit_years[year], dtype=np.float64)
            if (values.shape != (365 * ROWS_PER_DAY, 5) or not np.isfinite(values).all()
                    or np.any(values[:, 2:] < 0)):
                raise ValueError('invalid fitted-year weather matrix')
            self.training[year] = values.reshape(365, ROWS_PER_DAY, 5)
            self.training_solar_daily[year] = self.training[year][:, :, 3].mean(axis=1)
            features.append(_daily_features(values))
        x = np.tile(_harmonics(365), (2, 1))
        y = np.vstack(features)
        self.seasonal_coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y - x @ self.seasonal_coefficients
        self.residuals = residual
        previous = np.vstack((residual[:364], residual[365:729]))
        following = np.vstack((residual[1:365], residual[366:730]))
        self.var_coefficients = np.linalg.solve(
            previous.T @ previous + ridge * np.eye(5), previous.T @ following)
        radius = np.max(np.abs(np.linalg.eigvals(self.var_coefficients)))
        if radius > 0.95:
            self.var_coefficients *= 0.95 / radius
        innovations = following - previous @ self.var_coefficients
        self.innovations = innovations - innovations.mean(axis=0)
        self.scale = np.std(np.vstack([fit_years[year] for year in (2017, 2018)]),
                            axis=0, ddof=1)
        if np.any(self.scale <= 0) or not np.isfinite(self.scale).all():
            raise ValueError('degenerate training weather scale')

    def sample_candidate(self, seed: int) -> np.ndarray:
        """Produce one new 365-day path; caller must apply all-source audit."""
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError('private seed must be a nonnegative integer')
        rng = np.random.default_rng(seed)
        seasonal = _harmonics(365) @ self.seasonal_coefficients
        previous = self.residuals[int(rng.integers(len(self.residuals)))].copy()
        output = np.empty((365, ROWS_PER_DAY, 5), dtype=np.float64)
        for day in range(365):
            innovation = self.innovations[int(rng.integers(len(self.innovations)))]
            previous = previous @ self.var_coefficients + innovation
            target = seasonal[day] + previous
            offsets = np.arange(day - 14, day + 15) % 365
            candidates = [(year, int(index),
                           abs(np.log1p(self.training_solar_daily[year][index]) - target[3]))
                          for year in (2017, 2018) for index in offsets]
            # Condition the analogue on generated daily sunlight as well as
            # season. Unconditioned cloudy-day shapes could be multiplied into
            # impossible clear-sky peaks despite a plausible daily total.
            top = sorted(candidates, key=lambda row: row[2])[:3]
            first, second = rng.choice(len(top), size=2, replace=False)
            first_year, first_day, _ = top[int(first)]
            second_year, second_day, _ = top[int(second)]
            blend = float(rng.uniform(0.2, 0.4))
            profile = ((1 - blend) * self.training[first_year][first_day]
                       + blend * self.training[second_year][second_day])
            generated = np.empty_like(profile)
            generated[:, 0] = target[0] + profile[:, 0] - profile[:, 0].mean()
            spread = np.maximum(profile[:, 0] - profile[:, 1], 0)
            generated[:, 1] = generated[:, 0] - spread * np.expm1(target[1]) / max(spread.mean(), 1e-9)
            generated[:, 2] = profile[:, 2] * np.expm1(target[2]) / max(profile[:, 2].mean(), 1e-9)
            # The latent joint solar level conditions analogue selection, but
            # direct lognormal rescaling produced impossible 2-5 kW/m2 peaks
            # on cloudy profiles. A bounded cloudiness factor preserves the
            # selected within-day shape without post-hoc output clipping.
            sun_delta = target[3] - np.log1p(profile[:, 3].mean())
            solar_factor = 0.85 + 0.25 / (1 + np.exp(-sun_delta / 0.35))
            generated[:, 3] = profile[:, 3] * solar_factor
            generated[:, 4] = target[4] + profile[:, 4] - profile[:, 4].mean()
            if day:
                # A smooth first-hour transition avoids a discontinuous
                # non-solar forcing jump when independent analogue days join.
                for channel in (0, 1, 2, 4):
                    old = output[day - 1, -1, channel]
                    weight = np.linspace(1 / 7, 6 / 7, 6)
                    generated[:6, channel] = old * (1 - weight) + generated[:6, channel] * weight
            output[day] = generated
        result = output.reshape(-1, 5)
        violations = []
        if not np.isfinite(result).all():
            violations.append('nonfinite')
        for index, name, lo, hi in ((0, 'temperature', -30, 50),
                                    (2, 'wind', 0, 70),
                                    (3, 'shortwave', 0, 1500),
                                    (4, 'longwave', 100, 600)):
            actual_lo, actual_hi = float(np.min(result[:, index])), float(np.max(result[:, index]))
            if actual_lo < lo or actual_hi > hi:
                violations.append(f'{name}=[{actual_lo:.3f},{actual_hi:.3f}] outside [{lo},{hi}]')
        excess = float(np.max(result[:, 1] - result[:, 0]))
        if excess > 1:
            violations.append(f'dewpoint_above_temperature={excess:.3f} C')
        if violations:
            raise ValueError('candidate violates predeclared physical support: ' + '; '.join(violations))
        return result
