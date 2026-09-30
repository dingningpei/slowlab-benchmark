import numpy as np
import pytest

from slowlab.weather_repair import MAX_REPAIRS_PER_MONTH, MAX_RUN, invalid_mask, repair_months, repair_series, runs


def test_invalid_mask_matches_reader_support():
    m = invalid_mask('SWD', np.array([0.0, -9.9, -10.0, -93.9, 1499.0, 1500.0, np.nan, -9999.0]), -9999.0)
    assert m.tolist() == [False, False, True, True, False, True, True, True]
    assert invalid_mask('F010', np.array([0.0, -0.1, 69.9, 70.0])).tolist() == [False, True, False, True]


def test_runs_are_half_open():
    assert runs(np.array([1, 1, 0, 0, 1, 0, 1], dtype=bool)) == [(0, 2), (4, 5), (6, 7)]
    assert runs(np.zeros(3, dtype=bool)) == []


def test_short_runs_are_interpolated_and_long_or_edge_runs_are_not():
    v = np.array([10.0, 20.0, -99.0, -99.0, 50.0, 60.0])
    fixed, bad = repair_series(v, v < 0)
    assert [(i, round(r, 6)) for i, _, r in fixed] == [(2, 30.0), (3, 40.0)] and bad == []
    long = np.concatenate(([1.0], np.full(MAX_RUN + 1, np.nan), [2.0]))
    fixed, bad = repair_series(long, np.isnan(long))
    assert fixed == [] and 'run of 7' in bad[0][2]
    edge = np.array([np.nan, 1.0, 2.0])
    assert repair_series(edge, np.isnan(edge))[1][0][2] == 'invalid records at series edge'
    ok = np.concatenate(([1.0], np.full(MAX_RUN, np.nan), [8.0]))
    assert len(repair_series(ok, np.isnan(ok))[0]) == MAX_RUN


def month(n, swd=None):
    base = {'TA002': np.full(n, 280.0), 'TD002': np.full(n, 275.0), 'F010': np.full(n, 5.0)}
    rad = {'SWD': np.full(n, 100.0) if swd is None else swd, 'LWD': np.full(n, 300.0)}
    return {'meteo': {'arrays': base, 'fills': {}}, 'radiation': {'arrays': rad, 'fills': {'SWD': -9999.0}}}


def test_repair_months_logs_local_indices_and_crosses_month_edges():
    a = np.full(10, 100.0)
    a[-1] = -93.9
    b = np.full(10, 50.0)
    b[0] = -9999.0
    out = repair_months([('201412', month(10, a)), ('201501', month(10, b))])
    assert out['problems'] == []
    assert [(r['month'], r['index'], r['original']) for r in out['repairs']] == [('201412', 9, -93.9), ('201501', 0, -9999.0)]
    assert [round(r['repaired'], 6) for r in out['repairs']] == [pytest.approx(100 - 50 / 3), pytest.approx(100 - 100 / 3)]
    assert out['repairs_per_month'] == {'201412': 1, '201501': 1}


def test_too_many_repairs_in_a_month_excludes_the_year():
    s = np.full(200, 100.0)
    s[1:2 * (MAX_REPAIRS_PER_MONTH + 1):2] = np.nan
    out = repair_months([('201501', month(200, s))])
    assert any('exceed' in p for p in out['problems'])
    long = np.full(50, 100.0)
    long[10:20] = np.nan
    out = repair_months([('201501', month(50, long))])
    assert out['problems'] == ['radiation.SWD: run of 10 invalid records (201501[10] to 201501[19])']
