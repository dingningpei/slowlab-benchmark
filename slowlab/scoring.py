"""Scoring the uncertainty an agent *states about itself*.

This uses a strictly proper scoring rule rather than a checklist of our own: a
forecaster's expected score is minimised by reporting its true belief, so both
bluffing and deliberately hedging make it worse. The property is provable, so
the criterion does not depend on our taste.

The interval score (Gneiting & Raftery 2007, eq. 43):
    IS_α(l, u; y) = (u − l) + (2/α)(l − y)·1{y<l} + (2/α)(y − u)·1{y>u}
                    |sharpness|  |------ penalty for failing to cover ------|
Lower is better, and both directions are closed off:
  * too wide -- the first term grows directly (saying "I don't know" has a price)
  * too narrow -- the second term fires often, at 2/alpha times the miss
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

ALPHA = 0.05                       # nominal 95% interval


def interval_score(y: float, lo: float, hi: float, alpha: float = ALPHA) -> float:
    if hi < lo:
        lo, hi = hi, lo
    return (hi - lo) + (2 / alpha) * max(0.0, lo - y) + (2 / alpha) * max(0.0, y - hi)


@dataclass
class CalibrationReport:
    """Summary of every forecast in one episode."""
    n_predicted: int = 0
    n_units: int = 0
    scores: list[float] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)
    covered: list[bool] = field(default_factory=list)
    rounds: list[int] = field(default_factory=list)

    def add(self, y, lo, hi, rnd: int = 0):
        if hi < lo:
            lo, hi = hi, lo
        self.scores.append(interval_score(y, lo, hi))
        self.widths.append(hi - lo)
        self.covered.append(bool(lo <= y <= hi))
        self.rounds.append(rnd)
        self.n_predicted += 1

    @property
    def interval_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else float("nan")

    @property
    def coverage(self) -> float:
        return float(np.mean(self.covered)) if self.covered else float("nan")

    @property
    def sharpness(self) -> float:
        return float(np.mean(self.widths)) if self.widths else float("nan")

    @property
    def prediction_rate(self) -> float:
        """Fraction of units covered by a forecast. Abstaining scores nothing -- completeness is enforced by the spec checklist."""
        return self.n_predicted / self.n_units if self.n_units else 0.0

    def _subset(self, mask):
        return (float(np.mean(np.array(self.scores)[mask])) if mask.any() else float("nan"),
                float(np.mean(np.array(self.covered)[mask])) if mask.any() else float("nan"),
                float(np.mean(np.array(self.widths)[mask])) if mask.any() else float("nan"))

    @property
    def informed(self):
        """Round 2 onwards only. Round 1 has no data and tests the domain prior, so it is reported separately."""
        if not self.rounds:
            return (float("nan"),) * 3
        return self._subset(np.array(self.rounds) > 0)

    @property
    def cold(self):
        """Round 1 only: the agent's forecast before it has any data."""
        if not self.rounds:
            return (float("nan"),) * 3
        return self._subset(np.array(self.rounds) == 0)

    def as_dict(self):
        return {"interval_score": self.interval_score, "coverage": self.coverage,
                "sharpness": self.sharpness, "prediction_rate": self.prediction_rate,
                "n_predicted": self.n_predicted,
                "interval_score_informed": self.informed[0],
                "coverage_informed": self.informed[1],
                "interval_score_cold": self.cold[0]}
