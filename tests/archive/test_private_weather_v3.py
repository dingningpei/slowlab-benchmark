import numpy as np
import pytest

from slowlab.archive.private_weather import ROWS_PER_DAY
from slowlab.archive.private_weather_v3 import BlockWeatherGeneratorV3


def year(phase):
    ticks = np.arange(365 * ROWS_PER_DAY)
    day = ticks / ROWS_PER_DAY
    hour = (ticks % ROWS_PER_DAY) / 6
    season = np.sin(2 * np.pi * (day - 80) / 365)
    sun = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    temp = 15 + 8 * season + 2 * np.sin(2 * np.pi * hour / 24) + phase
    dew = temp - 4
    wind = 3 + 0.5 * np.sin(2 * np.pi * day / 13 + phase)
    solar = 500 * sun * (0.8 + 0.1 * np.sin(2 * np.pi * day / 11 + phase))
    longwave = 320 + 20 * season
    return np.column_stack((temp, dew, wind, solar, longwave))


def test_v3_multiyear_block_candidate_is_deterministic_and_finite():
    inputs = {2017: year(0), 2018: year(.4), 2019: year(.8)}
    generator = BlockWeatherGeneratorV3(inputs)
    generated = generator.sample_candidate(123)
    assert generated.shape == (365 * ROWS_PER_DAY, 5)
    assert np.isfinite(generated).all()
    assert np.all(generated[:, 2:4] >= 0)
    assert np.array_equal(generated, generator.sample_candidate(123))
    assert not any(np.array_equal(generated, original) for original in inputs.values())


def test_v3_leap_day_is_explicitly_removed_only_from_training_calendar():
    base = year(1.2)
    leap_day = np.full((ROWS_PER_DAY, 5), (20., 16., 3., 0., 320.))
    leap = np.concatenate((base[:59 * ROWS_PER_DAY], leap_day,
                           base[59 * ROWS_PER_DAY:]))
    generator = BlockWeatherGeneratorV3({2017: year(0), 2018: year(.4), 2020: leap})
    assert np.array_equal(generator.training[2020][59],
                          base[59 * ROWS_PER_DAY:60 * ROWS_PER_DAY])
    assert generator.training[2020].shape == (365, ROWS_PER_DAY, 5)


def test_v3_rejects_invalid_training_years_and_seed():
    with pytest.raises(ValueError):
        BlockWeatherGeneratorV3({2017: year(0), 2018: year(.4)})
    with pytest.raises(ValueError):
        BlockWeatherGeneratorV3({2017: year(0), 2018: year(.4), 2019: year(.8), 2021: year(1)})
    generator = BlockWeatherGeneratorV3({2017: year(0), 2018: year(.4), 2019: year(.8)})
    with pytest.raises(ValueError):
        generator.sample_candidate(True)
