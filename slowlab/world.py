"""Turn TOMGRO into a *distribution over tasks* rather than a single task.

The problem
----
With fixed parameters TomgroModel is deterministic: every seed sees the same
world and the same optimum. Twenty seeds would then be twenty noise draws from
one problem rather than twenty problems -- the benchmark's statistical structure
collapses and there is no held-out family to construct.

What we do
----
Each instance draws a parameter set, entirely from published material:
  * J99 gives least-squares estimates at three sites. Their central finding is
    that *vegetative parameters are stable across sites while fruit parameters
    vary with location* -- so we draw the same way: vegetative parameters get a
    small perturbation, fruit parameters are drawn across the between-site
    spread.
  * VC14 gives +/-10% intervals for 17 parameters, used for within-site
    individual variation.

This is less diverse than a GP prior, but that is a real property of
agriculture, not a defect.
"""
from __future__ import annotations
import hashlib, json, pathlib
import numpy as np
from dataclasses import dataclass

from .factors import FactorSpec
from .tomgro import TomgroModel, TomgroParams
from .economics import EconomicModel, TomgroProfitModel, sample_site_econ
from . import tomgro_params as TP


# ── The eight *management decision* factors ────────────────────────────
# All are things a greenhouse actually controls, and TOMGRO responds to each.
# The first four are environmental drivers, the last four cultural operations
# (J99 explicitly lists LAImax as a management parameter).
MANAGEMENT_FACTORS = [
    FactorSpec("day_temp",   18.0, 32.0,  "chamber"),
    FactorSpec("night_temp", 12.0, 22.0,  "chamber"),
    FactorSpec("co2",       350.0, 1200.0,"chamber"),
    # Target DLI for tomato, including natural light, is 20-30 mol/(m^2.d). The
    # upper rail used to be 50, far beyond the crop's need and certain to lose
    # money at real electricity prices. Narrowed to 18-32.
    FactorSpec("par",        18.0, 32.0,  "chamber"),
    # Commercial greenhouse tomato density is 2.5-4 plants/m^2, with a little
    # headroom at the rails.
    # Why this used to be 2.0-8.0: under the old lambda=0.5 the optimal density
    # landed at 5.1-8.0, so we widened the range to accommodate it and argued in
    # a comment that narrowing it would make the factor degenerate. That argument
    # was backwards: the real cause was that lambda=0.5 halved the true cost. With
    # lambda removed (=1) the optimal density is 3.26 over [2.3, 4.3], exactly the
    # commercial range, and the factor range needs no accommodation.
    FactorSpec("density",     2.2, 4.5,   "loop"),     # rho, plants/m^2
    # leaf_prune (p1) is *not* a decision factor. In J99's reduced model,
    # defoliation only subtracts dry matter from W (eq. 9) while leaf area is
    # capped separately by LAI_max, so removing leaves lowers maintenance
    # respiration *without costing photosynthesis* -- more is always better and the
    # optimum sits permanently on the upper rail. That is a property of the
    # published model, and the original authors themselves list pruning among the
    # possible reasons their fruit predictions did not transfer across sites.
    # Attaching a labour cost would manufacture an interior optimum, but its
    # location would be set by a coefficient we invented rather than by
    # physiology, which is exactly the difficulty-tuning this paper argues
    # against. Fixed at the midpoint of its range.
    FactorSpec("lai_max",     2.0, 5.0,   "loop"),     # pruning target
    # cycle_days is *no longer a decision factor*: over this parameter range
    # TOMGRO does not reproduce late-season decline, so margin rate increases
    # monotonically with cycle length and the optimum always sits on the upper
    # rail (still true when measured out to 330 days). Rather than keep a trivial
    # dimension, we fix it and say so.
]
# Default cycle length. The source of truth is Task.cycle_days; this is only the
# fallback when there is no task.
# 210 days (30 weeks). Chosen because at that length the model gives an
# annualised 49 kg/m^2, at the bottom of the 50-70 kg/(m^2.yr) range for Dutch
# and Spanish greenhouse tomato; 120 days gives only 32 kg and 330 days gives
# 104 kg, the latter being the model run outside its valid range (the reduced
# model has no late-season decline). Mediterranean practice uses cycles of this
# magnitude too.
DEFAULT_CYCLE_DAYS = 210.0
FIXED_LEAF_PRUNE = 2.25          # midpoint of p1's range, see above

# The three-site spread of J99 Table 3 -- the real variation in fruit parameters
_FRUIT_SPREAD = {
    "alphaF":   (0.80, 0.95),
    "vartheta": (0.135, 0.200),
    "NFF":      (10.0, 22.0),
    "TCRIT":    (22.0, 24.4),
    "DFmax":    (0.04, 0.08),
}
# Vegetative parameters are stable across sites, so only perturb them slightly (VC14's +/-10%)
_VEG_STABLE = ("beta", "Nb", "delta", "Nm")


def sample_instance_params(seed: int, veg_cv: float = 0.06) -> TomgroParams:
    """Draw one site: vegetative parameters perturbed slightly, fruit parameters drawn across the between-site spread."""
    rng = np.random.default_rng(20_000 + seed)
    base = TomgroParams.from_j99("Gainesville")
    v = dict(base.values)
    for k in _VEG_STABLE:
        if k in v:
            v[k] = float(v[k] * np.exp(rng.normal(0, veg_cv)))
    for k, (lo, hi) in _FRUIT_SPREAD.items():
        v[k] = float(rng.uniform(lo, hi))
    v["E"] = float(rng.uniform(*TP.VC14_NOMINAL["E"][1:3]))
    return TomgroParams(values=v, intervals=TP.perturbation_intervals("VC14"),
                        provenance="j99", site=f"sampled#{seed}",
                        source="parameter distribution from the J99 three-site spread plus VC14 +/-10% intervals")



_ORACLE_CACHE = pathlib.Path(__file__).resolve().parents[1] / "results" / "_oracle_cache"


def _oracle_key(seed, factors, cycle_days, energy_shock=1.0):
    sig = "|".join(f"{f.name}:{f.low}:{f.high}" for f in factors)
    return f"{seed}|{cycle_days}|{energy_shock}|{hashlib.md5(sig.encode()).hexdigest()[:10]}"


def _oracle_disk(key):
    """On-disk cache for the oracle. With multi-start search one call costs about
    0.5 s, so building an M=320 atom set takes three minutes -- over the sandbox's
    per-call limit, and an in-process cache does not survive across processes."""
    _ORACLE_CACHE.mkdir(parents=True, exist_ok=True)
    return _ORACLE_CACHE / f"{key}.json"


def _maximise(f, d, rng, n_sample=6000, n_start=8,
              stages=(0.20, 0.08, 0.03, 0.01), n_local=600):
    """Maximisation by multi-start local refinement.

    An earlier version refined from the *single* best uniform sample, which is
    not enough in 4-6 dimensions: uniform coverage degrades as n^(-1/d) and a
    single start gets stuck locally. Measured, 20-24% of instances had their true
    optimum understated as a result, by an amount comparable to Screen's entire
    spread. The oracle appears in the definition of regret, so understating it
    understates every agent's regret together.

    tests/test_env.py::test_oracle_is_converged guards this with an independent,
    stronger search.
    """
    Xs = rng.random((n_sample, d))
    y = np.asarray(f(Xs), float)
    order = np.argsort(-y)[:n_start]
    bx, bv = Xs[order[0]], float(y[order[0]])
    for i in order:
        x0, v = Xs[i], float(y[i])
        for r in stages:
            Xl = np.clip(x0 + rng.normal(0, r, (n_local, d)), 0.0, 1.0)
            yl = np.asarray(f(Xl), float)
            j = int(np.argmax(yl))
            if yl[j] > v:
                x0, v = Xl[j], float(yl[j])
        if v > bv:
            bx, bv = x0, v
    return bx, bv


class ManagedTomgro:
    """Eight management factors and a gross margin, with every seed a different site.

    The ground-truth backend interface.
    """

    def __init__(self, seed: int = 0, econ: EconomicModel | None = None,
                 factors: list[FactorSpec] | None = None,
                 cycle_days: float = DEFAULT_CYCLE_DAYS):
        self.seed = seed
        self.cycle_days = float(cycle_days)
        self.factors = factors or MANAGEMENT_FACTORS
        self.d = len(self.factors)
        self.params = sample_instance_params(seed)
        # A site is a place: climate and prices vary with the seed rather than
        # being shared by every site. Passing econ overrides this (used by scan
        # scripts and unit tests).
        self.econ = econ if econ is not None else sample_site_econ(seed)
        self._oracle = None
        # Factors the task does not expose are fixed at their midpoint (the site's routine management)
        self._fixed = {f.name: 0.5 for f in MANAGEMENT_FACTORS
                       if f.name not in {g.name for g in self.factors}}

    def _split(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        f = self.factors
        return {f[i].name: f[i].low + X[:, i] * (f[i].high - f[i].low)
                for i in range(self.d)}

    def __call__(self, X, plant_params: dict | None = None, components: bool = False):
        """Fully vectorised: rho, LAImax, p1 and cycle length all vary per candidate point.

        Variable cycles are handled by running to the longest one and freezing WM
        for each point on its own end day.
        """
        d = self._split(X)
        n = len(next(iter(d.values())))
        for name, u in self._fixed.items():          # fill in the factors the task does not expose
            f = next(g for g in MANAGEMENT_FACTORS if g.name == name)
            d[name] = np.full(n, f.denorm(u))
        # Cycle length is not a decision variable (see Limitations). This must
        # *assert* rather than silently override: anyone who adds cycle_days back
        # to the factor set should get an error immediately, not discover later
        # that their whole scan ran at one cycle length.
        assert "cycle_days" not in {f.name for f in self.factors}, (
            "cycle_days is declared by the task, not chosen by the agent; "
            "set Task.cycle_days instead of adding it as a factor")
        d["cycle_days"] = np.full(n, self.cycle_days)
        assert "leaf_prune" not in {f.name for f in self.factors}, (
            "leaf_prune is fixed: the reduced model gives it no photosynthetic "
            "penalty, so its optimum is always at the upper rail")
        d["leaf_prune"] = np.full(n, FIXED_LEAF_PRUNE)
        p = dict(self.params.values) if plant_params is None else plant_params
        Tday, Tnight = d["day_temp"], d["night_temp"]
        Td = 0.5 * (Tday + Tnight)
        par, co2 = d["par"], d["co2"]
        rho, laimax, p1 = d["density"], d["lai_max"], d["leaf_prune"]
        end = np.round(d["cycle_days"]).astype(int)

        crop = TomgroModel(params=TomgroParams(values=dict(p), provenance="j99"),
                           factors=self.factors[:4], days=1)
        N = np.full(n, 3.0); LAI = np.full(n, 0.01)
        W = np.full(n, 0.3); WF = np.zeros(n); WM = np.zeros(n)
        WM_final = np.zeros(n)
        gT = TP.g_daytime_vec(Tday, p["TCRIT"])

        for day in range(1, int(end.max()) + 1):
            live = day <= end
            dN = p["Nm"] * crop._fN(Tday)
            e = np.exp(np.clip(p["beta"] * (N - p["Nb"]), -30, 30))
            dLAI = np.where(LAI <= laimax,
                            rho * p["delta"] * crop._lambda(Td) * e / (1 + e) * dN, 0.0)
            Pg = crop._Pg(LAI, par, co2, p)
            Rm = p["Q10"] ** ((Td - 20.0) / 10.0) * p["rm"] * np.clip(W - WM, 0, None)
            GRnet = np.clip(p["E"] * (Pg - Rm) * (1.0 - crop._fR(N)), 0.0, None)
            started = N > p["NFF"]
            dWF = np.where(started,
                           GRnet * p["alphaF"] * crop._fF(Td)
                           * (1.0 - np.exp(-p["vartheta"] * np.clip(N - p["NFF"], 0, None)))
                           * gT, 0.0)
            dW = np.minimum(GRnet - np.where(LAI >= laimax, p1 * rho * dN, 0.0),
                            dWF + (p["Vmax"] - p1) * rho * dN)
            mat = np.where(N > p["NFF"] + p["kappaF"],
                           crop._DF(Td, p) * np.clip(WF - WM, 0, None), 0.0)
            N = np.where(live, N + dN, N)
            LAI = np.where(live, LAI + dLAI, LAI)
            W = np.where(live, np.clip(W + dW, 0, None), W)
            WF = np.where(live, np.clip(WF + dWF, 0, None), WF)
            WM = np.where(live, np.clip(WM + mat, 0, None), WM)
            WM_final = np.where(day == end, WM, WM_final)

        if components:
            # Decomposable observation. The price shock acts on electricity and
            # gas only, so cost is returned split into energy and other -- an agent
            # that learned a single scalar cost_rate cannot re-solve after an
            # energy price rise, because it does not know how much of it was
            # energy. This is both harder and more realistic than the old version,
            # where lambda scaled all cost and cost_rate was scalable as a whole.
            days = end.astype(float)
            return {"rev_rate": self.econ.revenue(WM_final) / days,
                    "energy_cost_rate": self.econ.energy_cost(Tday, Tnight, par),
                    "other_cost_rate": self.econ.other_cost(co2, Tday, Tnight, rho, days),
                    "cost_rate": self.econ.daily_cost(Tday, Tnight, co2, par, rho, days),
                    "profit": self.econ.profit_rate(WM_final, Tday, Tnight, co2, par,
                                                    end, rho)}
        return self.econ.profit_rate(WM_final, Tday, Tnight, co2, par, end, rho)

    def sample_plant_params(self, n: int, rng, cv: float) -> dict:
        """Draw a parameter set per plant -- plant-to-plant variation is modelled as
        *parameter* variation, not additive noise.

        This is more faithful: plants differ because of transplant vigour and
        genetics, not because of measurement error invented at harvest. As a
        by-product it produces heteroscedasticity and temporal correlation for
        free (a large plant stays large).
        """
        out = {}
        for k, v in self.params.values.items():
            if k in TP.VC14_NOMINAL:                 # perturb only parameters with a published interval
                out[k] = np.asarray(v) * np.exp(rng.normal(0, cv, n))
            else:
                out[k] = v
        return out

    def oracle_at(self, energy_shock: float, n_sample: int = 6000, seed: int = 1):
        """The true optimum at a *different energy price*. Used by the economic
        transfer task.

        This interface used to take cost_weight, which scaled *all* cost -- the
        tomato price, transplants and labour moved too, which is not an energy
        price rise but a different objective function. It now moves electricity
        and gas only; see EconomicModel.energy_shock for the source.
        """
        old = self.econ.energy_shock
        self.econ.energy_shock = float(energy_shock)
        try:
            return _maximise(self, self.d, np.random.default_rng(seed),
                             n_sample=n_sample)
        finally:
            self.econ.energy_shock = old

    def profit_at(self, X, energy_shock: float):
        old = self.econ.energy_shock
        self.econ.energy_shock = float(energy_shock)
        try:
            return self(X)
        finally:
            self.econ.energy_shock = old

    def oracle(self, n_sample: int = 6000, seed: int = 0):
        if self._oracle is None:
            fp = _oracle_disk(_oracle_key(self.seed, self.factors,
                                          self.cycle_days,
                                          self.econ.energy_shock))
            if fp.exists():
                d = json.loads(fp.read_text())
                self._oracle = (np.asarray(d["x"], float), float(d["v"]))
            else:
                x, v = _maximise(self, self.d, np.random.default_rng(seed),
                                 n_sample=n_sample)
                fp.write_text(json.dumps({"x": list(map(float, x)), "v": v}))
                self._oracle = (x, v)
        return self._oracle

    @property
    def response_sd(self) -> float:
        rng = np.random.default_rng(999)
        return float(np.std(self(rng.random((800, self.d)))))
