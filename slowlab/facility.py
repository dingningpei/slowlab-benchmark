"""The facility: chamber / loop as two levels of unit, and the control-hierarchy constraint."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Unit:
    chamber: int
    loop: int

    @property
    def id(self) -> str:
        return f"c{self.chamber}l{self.loop}"


class Facility:
    # Area of one loop. An experimental unit is *not one plant*: it is the whole
    # stand on that loop, and the observation is their mean -- so unit-level noise
    # is plant-to-plant noise divided by sqrt(N), with N = density x area. We used
    # to draw a single plant per unit, overstating unit noise by sqrt(N), which
    # buried the between-site spread (0.0066) under measurement noise (0.0165) and
    # made experimenting unprofitable in principle.
    # 12 m^2 is our setting, not a literature value: row spacing and row length in
    # commercial houses combine to loop areas from single digits to tens of square
    # metres, and we take something in the middle. It enters only through
    # plants_per_unit, i.e. unit noise as 1/sqrt(N); sensitivity is in the appendix.
    DEFAULT_LOOP_AREA_M2 = 12.0

    def __init__(self, n_chambers: int, loops_per_chamber: int,
                 loop_area_m2: float = DEFAULT_LOOP_AREA_M2):
        self.n_chambers = n_chambers
        self.loops_per_chamber = loops_per_chamber
        self.loop_area_m2 = float(loop_area_m2)
        self.units = [Unit(c, l) for c in range(n_chambers)
                      for l in range(loops_per_chamber)]
        self._by_id = {u.id: u for u in self.units}

    def __len__(self):
        return len(self.units)

    def get(self, uid: str) -> Unit:
        return self._by_id[uid]

    def draw_effects(self, rng, tau_chamber: float, tau_loop: float):
        """Position effects: chamber-level plus loop-level, drawn once and held fixed for the episode (the second noise layer)."""
        ce = rng.normal(0, tau_chamber, self.n_chambers)
        le = rng.normal(0, tau_loop, len(self.units))
        return ce, {u.id: le[i] for i, u in enumerate(self.units)}

    def plants_per_unit(self, density: float) -> int:
        """Plants on one loop. Density is both a yield decision and a *measurement precision* decision."""
        return max(1, int(round(float(density) * self.loop_area_m2)))
