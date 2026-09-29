"""Factor specifications and the control hierarchy."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class FactorSpec:
    """One actionable factor.

    control_level is the finest granularity at which it can be set, and
    therefore its ceiling on parallelism:
      'chamber' environmental (air temperature, CO2, supplemental light) -- air
                and lamps serve a whole chamber, so no variation within one
      'loop'    cultural (density, training, defoliation) -- settable per loop
    """
    name: str
    low: float
    high: float
    control_level: str = "loop"

    def denorm(self, u):
        return self.low + u * (self.high - self.low)

    def norm(self, v):
        return (v - self.low) / (self.high - self.low)
