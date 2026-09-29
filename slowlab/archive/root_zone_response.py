"""Published reference curves for root-zone EC sensitivity.

These functions are evidence checks, not yet the SlowLab environment response.
They intentionally expose only the two transpiration regimes actually tested by
Li, Stanghellini and Challa (2001); interpolating between them would add an
untested modelling assumption.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math


TranspirationRegime = Literal["high", "low"]


@dataclass(frozen=True)
class PublishedECResponse:
    root_zone_ec_ds_m: float
    transpiration_regime: TranspirationRegime
    total_fresh_efficiency_g_mj_par: float
    marketable_fresh_efficiency_g_mj_par: float
    fruit_dry_matter_pct: float


_TOTAL = {
    "high": (33.03, -1.26),
    "low": (34.69, -1.08),
}
_MARKETABLE = {
    "high": (33.39, -1.54),
    "low": (34.25, -1.08),
}


def li2001_ec_response(
    root_zone_ec_ds_m: float,
    transpiration_regime: TranspirationRegime,
) -> PublishedECResponse:
    """Evaluate the published regressions inside their observed EC range."""
    ec = float(root_zone_ec_ds_m)
    if not math.isfinite(ec) or not 2.1 <= ec <= 9.3:
        raise ValueError("Li 2001 EC response is restricted to 2.1--9.3 dS/m")
    if transpiration_regime not in _TOTAL:
        raise ValueError("transpiration_regime must be 'high' or 'low'")
    total_intercept, total_slope = _TOTAL[transpiration_regime]
    market_intercept, market_slope = _MARKETABLE[transpiration_regime]
    return PublishedECResponse(
        root_zone_ec_ds_m=ec,
        transpiration_regime=transpiration_regime,
        total_fresh_efficiency_g_mj_par=total_intercept + total_slope * ec,
        marketable_fresh_efficiency_g_mj_par=market_intercept + market_slope * ec,
        fruit_dry_matter_pct=4.60 + 0.19 * ec,
    )
