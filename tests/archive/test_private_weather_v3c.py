import numpy as np
import pytest
import sys
from pathlib import Path

from slowlab.archive.private_weather_v3 import BlockWeatherGeneratorV3
from slowlab.archive.private_weather_v3c import BlockWeatherGeneratorV3c

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'v22'))
from audit_weather_v3c_development import fair_energy_score


def synthetic_year(phase):
    ticks = np.arange(365 * 144)
    day = ticks / 144
    hour = (ticks % 144) / 6
    season = np.sin(2 * np.pi * day / 365 + phase)
    sun = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    temperature = 15 + 8 * season + 2 * np.sin(2 * np.pi * hour / 24)
    return np.column_stack((temperature, temperature - 4,
                            3 + 0.5 * np.sin(day / 10 + phase),
                            500 * sun * (0.8 + 0.1 * season),
                            320 + 20 * season))


def test_v3c_preserves_v3b_algorithm_on_original_training_years():
    training = {2017: synthetic_year(0), 2018: synthetic_year(.4),
                2019: synthetic_year(.8)}
    original = BlockWeatherGeneratorV3(training).sample_candidate(123)
    expanded = BlockWeatherGeneratorV3c(training).sample_candidate(123)
    assert np.array_equal(original, expanded)


def test_v3c_accepts_audited_seven_year_set_and_rejects_incomplete_or_holdout_year():
    training = {year: synthetic_year(offset / 4)
                for offset, year in enumerate((2013, 2014, 2016, 2017, 2018, 2019, 2020))}
    training[2020] = np.concatenate((training[2020][:59 * 144],
                                     training[2020][59 * 144:60 * 144],
                                     training[2020][59 * 144:]))
    generator = BlockWeatherGeneratorV3c(training)
    assert generator.fit_years == tuple(sorted(training))
    with pytest.raises(ValueError):
        BlockWeatherGeneratorV3c({**training, 2012: synthetic_year(2)})
    with pytest.raises(ValueError):
        BlockWeatherGeneratorV3c({**training, 2015: synthetic_year(2)})


def test_weather_energy_score_uses_off_diagonal_pairs():
    assert fair_energy_score(np.array([[-1.], [1.]]), np.array([0.])) == 0
    assert fair_energy_score(np.array([[1.]]), np.array([0.])) is None
