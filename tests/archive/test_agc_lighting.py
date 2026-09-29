import pytest

from slowlab.archive.agc_lighting import (
    HPS_NAMEPLATE_W_M2_FLOOR,
    LED_FIELDS,
    agc_toplight_flux,
)


def full_led(value=1000):
    return {field: value for field in LED_FIELDS}


def test_full_commands_preserve_par_far_red_and_two_power_bases():
    flux = agc_toplight_flux(100, full_led())
    assert flux.hps_photon_flux_umol_m2_s == 100
    assert flux.led_par_photon_flux_umol_m2_s == 97
    assert flux.led_far_red_photon_flux_umol_m2_s == 12
    assert flux.par_photon_flux_umol_m2_s == 197
    assert flux.processed_hps_power_w_m2 == 81
    assert flux.processed_led_power_w_m2 == pytest.approx(61.52)
    assert flux.hps_nameplate_power_w_m2_floor == pytest.approx(HPS_NAMEPLATE_W_M2_FLOOR)
    assert flux.hps_nameplate_power_w_m2_floor != flux.processed_hps_power_w_m2


def test_hps_off_makes_missing_or_retained_led_commands_effective_zero():
    flux = agc_toplight_flux(0, {field: None for field in LED_FIELDS})
    assert flux.processed_total_power_w_m2 == 0
    assert flux.par_photon_flux_umol_m2_s == 0
    retained = agc_toplight_flux(0, full_led())
    assert retained.processed_total_power_w_m2 == 0
    assert retained.par_photon_flux_umol_m2_s == 0


def test_unknown_led_while_hps_on_fails_closed():
    with pytest.raises(ValueError, match="unknown while HPS is powered"):
        agc_toplight_flux(100, {**full_led(), "int_white_vip": None})


@pytest.mark.parametrize("hps", [-1, 101, float("nan")])
def test_invalid_hps_is_rejected(hps):
    with pytest.raises(ValueError):
        agc_toplight_flux(hps, full_led(0))


@pytest.mark.parametrize("vip", [-1, 1001, float("inf")])
def test_invalid_led_vip_is_rejected(vip):
    with pytest.raises(ValueError):
        agc_toplight_flux(100, {**full_led(0), "int_blue_vip": vip})
