import pytest

from slowlab.root_zone_response import li2001_ec_response


def test_published_high_transpiration_equations_are_reproduced():
    response = li2001_ec_response(2.1, "high")
    assert response.total_fresh_efficiency_g_mj_par == pytest.approx(33.03 - 1.26 * 2.1)
    assert response.marketable_fresh_efficiency_g_mj_par == pytest.approx(33.39 - 1.54 * 2.1)
    assert response.fruit_dry_matter_pct == pytest.approx(4.60 + 0.19 * 2.1)


def test_low_transpiration_has_smaller_ec_penalty_at_high_ec():
    high = li2001_ec_response(9.0, "high")
    low = li2001_ec_response(9.0, "low")
    assert low.marketable_fresh_efficiency_g_mj_par > high.marketable_fresh_efficiency_g_mj_par


@pytest.mark.parametrize("ec", [0, 2.0, 9.4, float("nan")])
def test_extrapolation_outside_source_range_fails_closed(ec):
    with pytest.raises(ValueError, match="restricted"):
        li2001_ec_response(ec, "high")


def test_unobserved_transpiration_regime_is_not_interpolated():
    with pytest.raises(ValueError, match="high.*low"):
        li2001_ec_response(4.0, "medium")
