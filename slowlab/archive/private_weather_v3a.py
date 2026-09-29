"""Development v3 weather candidate: joint seasonal multiday blocks.

2017-2020 are now development years. No outcome, formal site seed, or untouched
2015/2025 holdout data enters this module. Every path still needs a separate
physical-support and all-source non-replay audit before simulator use.
"""
from __future__ import annotations

import numpy as np

from .private_weather import CHANNELS, ROWS_PER_DAY, _daily_features, _harmonics


def _training_calendar_year(year: int, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if year == 2020:
        if values.shape != (366 * ROWS_PER_DAY, len(CHANNELS)):
            raise ValueError('2020 training year must retain its leap day')
        # Calendar-align March through December with non-leap years. The
        # removed Feb 29 is development-only training preprocessing; actual
        # 2020 diagnostics preserve it and use real month boundaries.
        values = np.concatenate((values[:59 * ROWS_PER_DAY],
                                 values[60 * ROWS_PER_DAY:]), axis=0)
    if values.shape != (365 * ROWS_PER_DAY, len(CHANNELS)):
        raise ValueError('training year has wrong calendar length')
    if not np.isfinite(values).all() or np.any(values[:, 2:] < 0):
        raise ValueError('invalid training weather values')
    return values


class BlockWeatherGeneratorV3:
    """Seasonal joint daily anomalies sampled in same-season 14-day blocks."""

    def __init__(self, fit_years: dict[int, np.ndarray], *, block_days: int = 14,
                 jitter_days: int = 14):
        if (not set(fit_years).issubset({2017, 2018, 2019, 2020})
                or len(fit_years) < 3 or not 2 <= block_days <= 30
                or not 0 <= jitter_days <= 30):
            raise ValueError('v3 development fit requires 3-4 declared years and bounded blocks')
        self.fit_years = tuple(sorted(fit_years))
        self.block_days = block_days
        self.jitter_days = jitter_days
        self.training = {}
        self.training_solar_daily = {}
        features = []
        for year in self.fit_years:
            values = _training_calendar_year(year, fit_years[year])
            daily = values.reshape(365, ROWS_PER_DAY, len(CHANNELS))
            self.training[year] = daily
            self.training_solar_daily[year] = daily[:, :, 3].mean(axis=1)
            features.append(_daily_features(values))
        x = np.tile(_harmonics(365), (len(self.fit_years), 1))
        y = np.vstack(features)
        self.seasonal_coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        seasonal = _harmonics(365) @ self.seasonal_coefficients
        self.daily_residuals = {year: feature - seasonal
                                for year, feature in zip(self.fit_years, features)}
        self.scale = np.std(np.vstack([self.training[year].reshape(-1, len(CHANNELS))
                                       for year in self.fit_years]), axis=0, ddof=1)
        if np.any(self.scale <= 0) or not np.isfinite(self.scale).all():
            raise ValueError('degenerate training weather scale')

    def sample_candidate(self, seed: int) -> np.ndarray:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError('private seed must be a nonnegative integer')
        rng = np.random.default_rng(seed)
        seasonal = _harmonics(365) @ self.seasonal_coefficients
        output = np.empty((365, ROWS_PER_DAY, len(CHANNELS)), dtype=np.float64)
        for block_start in range(0, 365, self.block_days):
            year = self.fit_years[int(rng.integers(len(self.fit_years)))]
            source_start = block_start + int(rng.integers(-self.jitter_days,
                                                         self.jitter_days + 1))
            for day in range(block_start, min(block_start + self.block_days, 365)):
                source_day = (source_start + day - block_start) % 365
                target = seasonal[day] + self.daily_residuals[year][source_day]
                offsets = np.arange(day - 14, day + 15) % 365
                candidates = [(yr, int(index),
                               abs(np.log1p(self.training_solar_daily[yr][index]) - target[3]))
                              for yr in self.fit_years for index in offsets]
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
                generated[:, 1] = (generated[:, 0] - spread * np.expm1(target[1])
                                   / max(spread.mean(), 1e-9))
                generated[:, 2] = (profile[:, 2] * np.expm1(target[2])
                                   / max(profile[:, 2].mean(), 1e-9))
                sun_delta = target[3] - np.log1p(profile[:, 3].mean())
                solar_factor = 0.85 + 0.25 / (1 + np.exp(-sun_delta / 0.35))
                generated[:, 3] = profile[:, 3] * solar_factor
                generated[:, 4] = target[4] + profile[:, 4] - profile[:, 4].mean()
                if day:
                    for channel in (0, 1, 2, 4):
                        old = output[day - 1, -1, channel]
                        weight = np.linspace(1 / 7, 6 / 7, 6)
                        generated[:6, channel] = old * (1 - weight) + generated[:6, channel] * weight
                output[day] = generated
        result = output.reshape(-1, len(CHANNELS))
        violations = []
        if not np.isfinite(result).all():
            violations.append('nonfinite')
        for index, name, lo, hi in ((0, 'temperature', -30, 50),
                                    (2, 'wind', 0, 70),
                                    (3, 'shortwave', 0, 1500),
                                    (4, 'longwave', 100, 600)):
            minimum, maximum = float(np.min(result[:, index])), float(np.max(result[:, index]))
            if minimum < lo or maximum > hi:
                violations.append(f'{name}=[{minimum:.3f},{maximum:.3f}] outside [{lo},{hi}]')
        excess = float(np.max(result[:, 1] - result[:, 0]))
        if excess > 1:
            violations.append(f'dewpoint_above_temperature={excess:.3f} C')
        if violations:
            raise ValueError('candidate violates physical support: ' + '; '.join(violations))
        return result
