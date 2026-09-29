import numpy as np
import pytest

from slowlab.v22_private_weather import CandidateWeatherGenerator, ROWS_PER_DAY


def synthetic_year(phase):
    ticks = np.arange(365 * ROWS_PER_DAY)
    day = ticks / ROWS_PER_DAY
    hour = (ticks % ROWS_PER_DAY) / 6
    season = np.sin(2 * np.pi * (day - 80) / 365)
    sun = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    temp = 15 + 9 * season + 3 * np.sin(2 * np.pi * (hour - 7) / 24) + phase
    dew = temp - 4 - 0.5 * np.sin(2 * np.pi * hour / 24)
    wind = 2 + 0.5 * np.sin(2 * np.pi * day / 9 + phase)
    shortwave = 550 * sun * (0.7 + 0.1 * np.sin(2 * np.pi * day / 11 + phase))
    longwave = 320 + 20 * season + 5 * np.sin(2 * np.pi * hour / 24)
    return np.column_stack((temp, dew, wind, shortwave, longwave))


def test_joint_candidate_has_new_finite_365_day_path():
    a = synthetic_year(0)
    b = synthetic_year(1)
    generator = CandidateWeatherGenerator({2017: a, 2018: b})
    generated = generator.sample_candidate(12)
    assert generated.shape == (365 * ROWS_PER_DAY, 5)
    assert np.isfinite(generated).all()
    assert np.all(generated[:, 1] <= generated[:, 0] + 1)
    assert np.all(generated[:, 2:4] >= 0)
    assert not np.array_equal(generated, a)
    assert not np.array_equal(generated, b)
    assert np.array_equal(generated, generator.sample_candidate(12))


def test_reject_wrong_fit_years_and_private_seed_type():
    a = synthetic_year(0)
    with pytest.raises(ValueError):
        CandidateWeatherGenerator({2017: a, 2019: a})
    generator = CandidateWeatherGenerator({2017: a, 2018: synthetic_year(1)})
    with pytest.raises(ValueError):
        generator.sample_candidate(True)
