"""Reduced state-variable TOMGRO -- equations from Jones, Kenig & Vallejos (1999).

The five state variables
    N    main-stem node number
    LAI  leaf area index
    W    total above-ground dry weight   g[d.w.]/m^2
    WF   fruit dry weight                g[d.w.]/m^2
    WM   mature fruit dry weight         g[d.w.]/m^2   <- the response variable (yield)

Equations 1-11 follow the paper. **J99 says of the sub-functions below only that
they are "as in the full TOMGRO", without writing them out**, so this file
approximates them with the usual literature forms and registers each one in
`APPROXIMATED`:
    fN(T) · λ(Td) · fF(Td) · DF(Td) · Pg · fR(N) · Q10

The reason this is acceptable: the skeleton is not here to predict tomato yield
accurately, it is here to produce structure a GP cannot -- thresholds,
non-monotonicity, source-sink limitation. The high-temperature cutoff of eq. (8)
and the minimum in eq. (10) already do that, and the reduced functions are all
0-1 shape functions whose error is far smaller than the between-site parameter
spread J99 itself reports (NFF differs 2.2-fold across their three sites).

When one is replaced by its original form, delete the corresponding APPROXIMATED entry.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from .factors import FactorSpec
from . import tomgro_params as TP


class PlaceholderParametersError(RuntimeError):
    """Refuses to produce paper-grade results while parameters or equations are still invented placeholders."""


# Sub-functions that are still approximations -- each must name what would replace it
APPROXIMATED = {
    "fN(T)":      "trapezoidal cardinal-temperature function (Tmin/Topt1/Topt2/Tmax) -> replace with Jones et al. (1991)",
    "lambda(Td)": "as above but steeper at high temperature (J99 states that heat suppresses leaf expansion more than node development)",
    "fF(Td)":     "saturating rise describing cold suppression of fruit partitioning -> replace with Jones et al. (1991)",
    "DF(Td)":     "linear in mean daily temperature, scaled by DFmax -> Marcelis & Koning (1995)",
    "Pg":         "Acock-type joint light/CO2 limitation with canopy interception -> replace with Acock et al. (1978)",
    "fR(N)":      "root partitioning declining with development -> replace with Jones et al. (1991)",
    "Q10":        "taken as 1.4 (TOMGRO commonly uses 1.4-2.0); J99 gives no value",
}


@dataclass
class TomgroParams:
    values: dict[str, float] = field(default_factory=dict)
    intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    provenance: str = "placeholder"     # 'placeholder' | 'j99' | 'fitted'
    site: str = ""
    source: str = ""

    @classmethod
    def from_j99(cls, site: str = "Gainesville") -> "TomgroParams":
        """J99 Table 3 plus carried-over parameters. site in {Gainesville, LakeCity, Avignon}."""
        idx = {"Gainesville": 0, "LakeCity": 1, "Avignon": 2}[site]
        v = {k: t[idx] for k, t in TP.J99_TABLE3.items() if t[idx] is not None}
        v["Nm"] = TP.J99_CARRIED_OVER["Nm"].get(site, 0.50)
        v["p1"] = TP.J99_CARRIED_OVER["p1"]
        v["rm"] = TP.J99_CARRIED_OVER["rm"]
        v.setdefault("Vmax", 8.0)                      # only Avignon has an estimate
        v.setdefault("kappaF", 5.0)
        v.setdefault("DFmax", 0.06)
        v["LAImax"] = TP.J99_LAIMAX.get(site, 4.0)
        v["rho"] = TP.J99_DENSITY.get("Gainesville1993" if site == "Gainesville" else site, 3.1)
        v["E"] = TP.VC14_NOMINAL["E"][0]                # J99 does not list it; take the VC14 nominal
        v["Q10"] = 1.4                                  # approximation, see APPROXIMATED
        return cls(values=v, provenance="j99", site=site,
                   source="Jones, Kenig & Vallejos (1999), Trans. ASAE 42(1):255–265")

    def require_real(self):
        if self.provenance == "placeholder":
            raise PlaceholderParametersError(
                "Parameters are still placeholders. Load published values with TomgroParams.from_j99(site=...).")


DEFAULT_FACTORS = [
    FactorSpec("day_temp",   18.0, 32.0, "chamber"),
    FactorSpec("night_temp", 12.0, 22.0, "chamber"),
    FactorSpec("co2",       350.0, 1200.0, "chamber"),
    FactorSpec("par",        10.0, 55.0,  "chamber"),   # mol[PAR]/(m²·d)
]


class TomgroModel:
    """The ground-truth backend interface: __call__ / oracle / response_sd."""

    def __init__(self, factors=None, params: TomgroParams | None = None,
                 days: int = 120, allow_placeholder: bool = False):
        self.factors = factors or DEFAULT_FACTORS
        self.d = len(self.factors)
        self.params = params or TomgroParams.from_j99()
        self.days = days
        if not allow_placeholder:
            self.params.require_real()
        self._oracle = None

    def _drivers(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return {f.name: f.low + X[:, i] * (f.high - f.low)
                for i, f in enumerate(self.factors)}

    # ── Sub-functions (approximations, see APPROXIMATED) ─────
    @staticmethod
    def _trapezoid(T, tmin, t1, t2, tmax):
        return np.clip(np.minimum((T - tmin) / (t1 - tmin), (tmax - T) / (tmax - t2)), 0, 1)

    def _fN(self, T):                       # node development
        return self._trapezoid(T, 9.0, 22.0, 30.0, 42.0)

    def _lambda(self, Td):                  # leaf expansion: steeper at high temperature
        return self._trapezoid(Td, 9.0, 20.0, 26.0, 34.0)

    def _fF(self, Td):                      # cold suppression of fruit partitioning
        return np.clip((Td - 8.0) / 12.0, 0.0, 1.0)

    def _DF(self, Td, p):                   # fruit maturation rate
        return p["DFmax"] * np.clip((Td - 9.0) / 20.0, 0.0, 1.0)

    def _fR(self, N):                       # root partitioning, declining with development
        return np.clip(0.20 - 0.006 * N, 0.02, 0.20)

    def _Pg(self, LAI, par, co2, p):        # Acock-type: joint light/CO2 limitation x canopy interception
        # A past bug: these three coefficients read TP.VC14_NOMINAL directly and
        # ignored the parameter dict p that was passed in. The consequence was
        # that *every site had an identical photosynthetic response*: however
        # P_Theta drew alpha / sigma / K, the optima in light and CO2 did not
        # move at all. That is the mechanism behind sites barely disagreeing in
        # the CO2 and par dimensions, and R*(none) sitting below the noise.
        # The line in Limitations claiming that "sampling the light and CO2
        # parameters over their published ranges would double the ceiling" had
        # never taken effect -- along this path it was a no-op.
        a = float(p.get("alpha", TP.VC14_NOMINAL["alpha"][0]))    # radiation use efficiency
        sig = float(p.get("sigma", TP.VC14_NOMINAL["sigma"][0])) * co2
        k = float(p.get("K_ext", 0.58))      # canopy extinction coefficient
        lightlim = a * par
        co_lim = np.where(lightlim + sig > 0, lightlim * sig / (lightlim + sig + 1e-9), 0.0)
        return co_lim * (1.0 - np.exp(-k * LAI)) * 12.0      # → g[CH2O]/(m²·d)

    # ── Equations 1-11 ──────────────────────────────────────
    def simulate(self, X, param_draw: dict | None = None) -> np.ndarray:
        p = param_draw or self.params.values
        dr = self._drivers(X)
        n = len(next(iter(dr.values())))
        Tday, Tnight = dr["day_temp"], dr["night_temp"]
        Td = 0.5 * (Tday + Tnight)                     # mean daily temperature
        par, co2 = dr["par"], dr["co2"]
        rho = p["rho"]

        N = np.full(n, 3.0); LAI = np.full(n, 0.01)
        W = np.full(n, 0.3); WF = np.zeros(n); WM = np.zeros(n)
        gT = TP.g_daytime_vec(Tday, p["TCRIT"])        # eq. (8)

        for _ in range(self.days):
            dN = p["Nm"] * self._fN(Tday)                                    # (1)
            e = np.exp(np.clip(p["beta"] * (N - p["Nb"]), -30, 30))
            dLAI = np.where(LAI <= p["LAImax"],
                            rho * p["delta"] * self._lambda(Td) * e / (1 + e) * dN, 0.0)  # (4)
            Pg = self._Pg(LAI, par, co2, p)
            Rm = p["Q10"] ** ((Td - 20.0) / 10.0) * p["rm"] * np.clip(W - WM, 0, None)    # (6)
            GRnet = np.clip(p["E"] * (Pg - Rm) * (1.0 - self._fR(N)), 0.0, None)          # (5)

            started = N > p["NFF"]
            dWF = np.where(started,
                           GRnet * p["alphaF"] * self._fF(Td)
                           * (1.0 - np.exp(-p["vartheta"] * np.clip(N - p["NFF"], 0, None)))
                           * gT, 0.0)                                                     # (7)
            dW_9  = GRnet - np.where(LAI >= p["LAImax"], p["p1"] * rho * dN, 0.0)         # (9)
            dW_10 = dWF + (p["Vmax"] - p["p1"]) * rho * dN                                # (10)
            dW = np.minimum(dW_9, dW_10)                          # source-sink limit: take the smaller

            mature = np.where(N > p["NFF"] + p["kappaF"],
                              self._DF(Td, p) * np.clip(WF - WM, 0, None), 0.0)           # (11)
            N = N + dN; LAI = LAI + dLAI
            W = np.clip(W + dW, 0, None); WF = np.clip(WF + dWF, 0, None)
            WM = np.clip(WM + mature, 0, None)
        return WM

    def __call__(self, X):
        return self.simulate(X)

    def sample_plant_params(self, rng, cv: float = 0.08) -> dict:
        """Plant-to-plant variation is parameter variation. Draw from the interval where one exists, otherwise perturb multiplicatively by cv."""
        out = {}
        for k, v in self.params.values.items():
            if k in self.params.intervals:
                lo, hi = self.params.intervals[k]; out[k] = float(rng.uniform(lo, hi))
            elif k in TP.VC14_NOMINAL:
                out[k] = float(v * np.exp(rng.normal(0, cv)))
            else:
                out[k] = float(v)
        return out

    def oracle(self, n_sample: int = 20_000, seed: int = 0):
        if self._oracle is None:
            rng = np.random.default_rng(seed)
            Xs = rng.random((n_sample, self.d)); y = self(Xs)
            i = int(np.argmax(y)); x0, best = Xs[i], float(y[i])
            for r in (0.08, 0.02):
                Xl = np.clip(x0 + rng.normal(0, r, (3000, self.d)), 0, 1)
                yl = self(Xl); j = int(np.argmax(yl))
                if yl[j] > best: x0, best = Xl[j], float(yl[j])
            self._oracle = (x0, best)
        return self._oracle

    @property
    def response_sd(self) -> float:
        rng = np.random.default_rng(12345)
        return float(np.std(self(rng.random((2000, self.d)))))
