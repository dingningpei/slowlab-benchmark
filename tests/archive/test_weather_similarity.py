import numpy as np
import pytest

from slowlab.archive.weather_similarity import nearest_window_distances, audit_nonreplay


def test_duplicate_weather_window_is_rejected_against_all_sources():
    rng = np.random.default_rng(21)
    source_a = rng.normal(size=(288, 5))
    source_b = rng.normal(size=(288, 5))
    generated = source_b.copy()
    report = audit_nonreplay(generated, {2017: source_a, 2018: source_b},
                             np.ones(5), threshold=0.08)
    assert report['passed'] is False
    assert report['by_window']['24h']['near_duplicate_windows'] == 2
    assert report['by_window']['6h']['near_duplicate_windows'] == 8


def test_new_path_and_bad_scale_handling():
    source = np.zeros((144, 5))
    generated = np.ones((144, 5))
    result = nearest_window_distances(generated, source, np.ones(5), rows=36, stride=36)
    assert np.allclose(result, 1)
    with pytest.raises(ValueError):
        nearest_window_distances(generated, source, np.zeros(5), rows=36, stride=36)


def test_shifted_source_copy_escapes_aligned_windows_but_not_sliding_audit():
    rng = np.random.default_rng(22)
    source = rng.normal(size=(80, 5))
    generated = source[1:37].copy()
    aligned = nearest_window_distances(generated, source, np.ones(5),
                                       rows=36, stride=36)
    sliding = nearest_window_distances(generated, source, np.ones(5),
                                       rows=36, stride=36, source_stride=1)
    assert aligned[0] > 0.08
    assert sliding[0] < 1e-7
