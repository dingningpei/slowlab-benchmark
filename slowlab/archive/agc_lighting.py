"""Evidence-preserving AGC 2019 supplemental-light input accounting.

The AGC process computer exposes one HPS state and four ELIXIA channel VIPs.
This module keeps those sources separate and enforces the documented power
interlock.  It deliberately does not map ELIXIA to GreenLight's interlighting
state: that state has a crop-canopy geometry which is not established by the
AGC publication or archive.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


LED_FIELDS = (
    "int_blue_vip",
    "int_red_vip",
    "int_farred_vip",
    "int_white_vip",
)

# Maximum channel photon fluxes reported for AGC 2019.  Far-red is retained
# separately because it is outside the PAR total used for photosynthesis.
LED_CAPACITY_UMOL_M2_S = {
    "int_blue_vip": 11.0,
    "int_red_vip": 49.0,
    "int_farred_vip": 12.0,
    "int_white_vip": 37.0,
}

# Coefficients used by the official AGC processing ledger.  They are not an
# independent electricity measurement and must remain labelled accordingly.
HPS_LEDGER_W_M2 = 81.0
LED_LEDGER_W_M2 = {
    "int_blue_vip": 7.27,
    "int_red_vip": 25.30,
    "int_farred_vip": 6.23,
    "int_white_vip": 22.72,
}

HPS_CAPACITY_UMOL_M2_S = 100.0
HPS_NAMEPLATE_W_M2_FLOOR = 6 * 1000 / 96


@dataclass(frozen=True)
class AgcToplightFlux:
    """Observed-command light fluxes before any thermal-model calibration."""

    hps_fraction: float
    led_fraction_by_channel: Mapping[str, float]
    hps_photon_flux_umol_m2_s: float
    led_par_photon_flux_umol_m2_s: float
    led_far_red_photon_flux_umol_m2_s: float
    processed_hps_power_w_m2: float
    processed_led_power_w_m2: float
    hps_nameplate_power_w_m2_floor: float

    @property
    def par_photon_flux_umol_m2_s(self) -> float:
        return self.hps_photon_flux_umol_m2_s + self.led_par_photon_flux_umol_m2_s

    @property
    def processed_total_power_w_m2(self) -> float:
        return self.processed_hps_power_w_m2 + self.processed_led_power_w_m2


def _finite(value: float | int | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is unknown")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is unknown")
    return result


def agc_toplight_flux(
    assim_light_percent: float,
    led_vip: Mapping[str, float | int | None],
) -> AgcToplightFlux:
    """Convert one AGC command sample to separate overhead-light inputs.

    Missing LED VIPs are structural zeros only while HPS power is off.  A
    missing channel while HPS is on remains unknown and fails closed.  Values
    returned here are source accounting quantities, not emitted-light or heat
    measurements and not a GreenLight actuator vector.
    """

    hps = _finite(assim_light_percent, "AssimLight")
    if not 0 <= hps <= 100:
        raise ValueError("AssimLight must be in [0, 100]")
    hps_fraction = hps / 100

    fractions: dict[str, float] = {}
    for field in LED_FIELDS:
        value = led_vip.get(field)
        if value is None or not math.isfinite(float(value)):
            if hps == 0:
                fractions[field] = 0.0
                continue
            raise ValueError(f"{field} is unknown while HPS is powered")
        numeric = float(value)
        if not 0 <= numeric <= 1000:
            raise ValueError(f"{field} must be in [0, 1000]")
        # The process-computer VIP can retain a non-zero request while the HPS
        # supply is off.  The hardware interlock controls realised output, so
        # the effective fraction is zero regardless of that retained request.
        fractions[field] = numeric / 1000 if hps > 0 else 0.0

    led_photons = {
        field: LED_CAPACITY_UMOL_M2_S[field] * fractions[field]
        for field in LED_FIELDS
    }
    led_power = sum(
        LED_LEDGER_W_M2[field] * fractions[field] for field in LED_FIELDS
    )
    return AgcToplightFlux(
        hps_fraction=hps_fraction,
        led_fraction_by_channel=fractions,
        hps_photon_flux_umol_m2_s=HPS_CAPACITY_UMOL_M2_S * hps_fraction,
        led_par_photon_flux_umol_m2_s=sum(
            value for field, value in led_photons.items() if field != "int_farred_vip"
        ),
        led_far_red_photon_flux_umol_m2_s=led_photons["int_farred_vip"],
        processed_hps_power_w_m2=HPS_LEDGER_W_M2 * hps_fraction,
        processed_led_power_w_m2=led_power,
        hps_nameplate_power_w_m2_floor=HPS_NAMEPLATE_W_M2_FLOOR * hps_fraction,
    )
