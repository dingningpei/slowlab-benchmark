"""Auditable water and dissolved-solute balance for a rockwool root zone.

This module is intentionally independent of crop stress/yield response.  It
implements the physical accounting that must be validated first.  EC is mapped
to an effective dissolved-solute concentration under a fixed nutrient recipe;
ion-specific optimisation requires a different model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RootZoneParams:
    pore_volume_l_m2: float
    drain_threshold_fraction: float
    drain_rate_per_hour: float
    ec_to_concentration_mmol_l_per_ds_m: float
    crop_uptake_concentration_mmol_l: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.pore_volume_l_m2) or self.pore_volume_l_m2 <= 0:
            raise ValueError("pore_volume_l_m2 must be positive")
        if not 0 <= self.drain_threshold_fraction <= 1:
            raise ValueError("drain_threshold_fraction must be in [0, 1]")
        if not math.isfinite(self.drain_rate_per_hour) or self.drain_rate_per_hour < 0:
            raise ValueError("drain_rate_per_hour must be non-negative")
        if (not math.isfinite(self.ec_to_concentration_mmol_l_per_ds_m)
                or self.ec_to_concentration_mmol_l_per_ds_m <= 0):
            raise ValueError("EC conversion must be positive")
        if (not math.isfinite(self.crop_uptake_concentration_mmol_l)
                or self.crop_uptake_concentration_mmol_l < 0):
            raise ValueError("crop uptake concentration must be non-negative")


@dataclass(frozen=True)
class RootZoneState:
    water_l_m2: float
    solute_mmol_m2: float
    slab_temperature_c: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.water_l_m2) or self.water_l_m2 < 0:
            raise ValueError("water storage must be finite and non-negative")
        if not math.isfinite(self.solute_mmol_m2) or self.solute_mmol_m2 < 0:
            raise ValueError("solute storage must be finite and non-negative")
        if not math.isfinite(self.slab_temperature_c):
            raise ValueError("slab temperature must be finite")


@dataclass(frozen=True)
class RootZoneForcing:
    irrigation_l_m2: float
    feed_ec_ds_m: float
    potential_root_uptake_l_m2: float
    substrate_evaporation_l_m2: float = 0.0
    slab_temperature_c: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "irrigation_l_m2",
            "feed_ec_ds_m",
            "potential_root_uptake_l_m2",
            "substrate_evaporation_l_m2",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.slab_temperature_c is not None and not math.isfinite(self.slab_temperature_c):
            raise ValueError("slab_temperature_c must be finite when supplied")


@dataclass(frozen=True)
class RootZoneFluxes:
    irrigation_l_m2: float
    root_uptake_l_m2: float
    substrate_evaporation_l_m2: float
    drain_l_m2: float
    feed_solute_mmol_m2: float
    crop_solute_uptake_mmol_m2: float
    drained_solute_mmol_m2: float


class RockwoolRootZone:
    """A well-mixed effective rockwool store with explicit mass accounting."""

    def __init__(self, params: RootZoneParams):
        self.params = params

    def water_content_pct(self, state: RootZoneState) -> float:
        return 100.0 * state.water_l_m2 / self.params.pore_volume_l_m2

    def ec_ds_m(self, state: RootZoneState) -> float:
        if state.water_l_m2 <= 0:
            return float("nan")
        concentration = state.solute_mmol_m2 / state.water_l_m2
        return concentration / self.params.ec_to_concentration_mmol_l_per_ds_m

    def step(
        self,
        state: RootZoneState,
        forcing: RootZoneForcing,
        *,
        dt_hours: float,
    ) -> tuple[RootZoneState, RootZoneFluxes]:
        if not math.isfinite(dt_hours) or dt_hours <= 0:
            raise ValueError("dt_hours must be positive")

        p = self.params
        feed_solute = (
            forcing.irrigation_l_m2
            * forcing.feed_ec_ds_m
            * p.ec_to_concentration_mmol_l_per_ds_m
        )
        water = state.water_l_m2 + forcing.irrigation_l_m2
        solute = state.solute_mmol_m2 + feed_solute

        evaporation = min(forcing.substrate_evaporation_l_m2, water)
        water -= evaporation
        root_uptake = min(forcing.potential_root_uptake_l_m2, water)
        water -= root_uptake
        crop_solute = min(
            solute,
            root_uptake * p.crop_uptake_concentration_mmol_l,
        )
        solute -= crop_solute

        overflow = max(0.0, water - p.pore_volume_l_m2)
        water_after_overflow = water - overflow
        threshold = p.drain_threshold_fraction * p.pore_volume_l_m2
        rate_drain = min(
            max(0.0, water_after_overflow - threshold),
            p.drain_rate_per_hour
            * max(0.0, water_after_overflow - threshold)
            * dt_hours,
        )
        drain = overflow + rate_drain
        if water > 0 and drain > 0:
            drained_solute = min(solute, drain * (solute / water))
        else:
            drained_solute = 0.0
        water -= drain
        solute -= drained_solute

        next_state = RootZoneState(
            water_l_m2=max(0.0, water),
            solute_mmol_m2=max(0.0, solute),
            slab_temperature_c=(
                state.slab_temperature_c
                if forcing.slab_temperature_c is None
                else forcing.slab_temperature_c
            ),
        )
        fluxes = RootZoneFluxes(
            irrigation_l_m2=forcing.irrigation_l_m2,
            root_uptake_l_m2=root_uptake,
            substrate_evaporation_l_m2=evaporation,
            drain_l_m2=drain,
            feed_solute_mmol_m2=feed_solute,
            crop_solute_uptake_mmol_m2=crop_solute,
            drained_solute_mmol_m2=drained_solute,
        )
        return next_state, fluxes
