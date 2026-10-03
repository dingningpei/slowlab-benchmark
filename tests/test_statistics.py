import numpy as np

from slowlab.statistics import benjamini_hochberg, bca_interval, holm, minimum_detectable, sign_flip_test, summarise


def test_sign_flip_rejects_a_clear_shift_and_not_noise():
    rng = np.random.default_rng(0)
    assert sign_flip_test(rng.normal(3, 2, 48), flips=20_000) < 0.001
    p = [sign_flip_test(rng.normal(0, 2, 48), flips=2_000, seed=s) for s in range(40)]
    assert np.mean(np.array(p) < 0.05) <= 0.15


def test_sign_flip_matches_the_exact_test_on_small_data():
    import itertools
    d = np.array([1.0, 2.0, -0.5, 3.0, 0.7, 1.2])
    exact = np.mean([abs(np.dot(s, d) / len(d)) >= abs(d.mean()) - 1e-12
                     for s in itertools.product((-1, 1), repeat=len(d))])
    assert abs(sign_flip_test(d, flips=100_000) - exact) < 0.01


def test_bca_covers_the_mean_and_is_ordered():
    rng = np.random.default_rng(1)
    d = rng.normal(2, 3, 48)
    lo, hi = bca_interval(d)
    assert lo < d.mean() < hi and hi - lo < 3


def test_holm_and_bh_adjustments():
    p = {'a': 0.01, 'b': 0.04, 'c': 0.03}
    assert holm(p) == {'a': 0.03, 'c': 0.06, 'b': 0.06}
    bh = benjamini_hochberg(p)
    assert abs(bh['a'] - 0.03) < 1e-12 and abs(bh['b'] - 0.04) < 1e-12 and abs(bh['c'] - 0.04) < 1e-12


def test_minimum_detectable_and_summary():
    assert 1.8 < minimum_detectable(3.98, 64, 0.05 / 7) < 1.9
    s = summarise([1.0, 2.0, 3.0, 2.5, 1.5], seed=0, alpha_for_mde=0.05)
    assert s['sites'] == 5 and abs(s['mean'] - 2.0) < 1e-12 and s['p_sign_flip'] <= 0.07
