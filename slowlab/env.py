"""The SLOWLAB environment (the core of W1).

Design principle: the interface guarantees timing and feasibility, never
scientific correctness -- the latter is the capability under test.
"""
from __future__ import annotations
import copy
import numpy as np
from dataclasses import dataclass, field

from .factors import FactorSpec
from .facility import Facility
from .world import ManagedTomgro, MANAGEMENT_FACTORS
from .design import Design, Rejection, RejectCode, OK
from .validity import score_validity, ValidityReport
from .scoring import CalibrationReport


@dataclass
class Observation:
    design_id: int
    treatment: str
    unit_id: str
    value: float
    rev_rate: float = float("nan")   # revenue rate EUR/(m^2.d), including plant-to-plant noise
    cost_rate: float = float("nan")  # total cost rate EUR/(m^2.d), determined by the setpoints
    energy_cost_rate: float = float("nan")   # the electricity and gas part of it
    other_cost_rate: float = float("nan")    # purchased CO2, transplants, labour
    # value = rev_rate - cost_rate, where cost_rate = energy + other.
    # An energy price rise multiplies the energy term only, so to re-solve after a
    # price change without running anything an agent must model all *three*
    # components separately; learning only the value, or only a merged cost, is
    # not enough.


@dataclass
class EpisodeResult:
    task: str
    agent: str
    seed: int
    simple_regret: float
    recommended: np.ndarray
    n_designs: int
    n_valid: int
    validity_rate: float
    unit_days_used: int
    parallel_utilisation: float
    # ── The scoreboard ───────────────────────────────────
    # Outcome : simple_regret / regret_trace (how good the recommendation is)
    #           campaign_cash (the campaign's own net profit or loss, EUR/m^2)
    # Process : churn (how much of the trajectory was wasted motion), see _churn below
    regret_trace: list = field(default_factory=list)
    cumulative_regret: float = float("nan")   # Δ Σ_r Σ_u [f(x*) − f(a_r(u))]
    gain: float = float("nan")          # G = Σ max(0, R(r) − R(r+1))
    backslide: float = float("nan")     # B = Σ max(0, R(r+1) − R(r))
    churn: float = float("nan")         # kappa = B / (G + B); an internal diagnostic, not in the paper
    campaign_cash: float = 0.0
    wasted_rounds: int = 0
    committed_rounds: int = 0
    wasted_fraction: float = float("nan")
    lost_unit_days: float = 0.0
    transfer_regret: float = float("nan")
    probe_score: float = float("nan")
    probe_coverage: float = float("nan")
    probe_sharpness: float = float("nan")
    interval_score: float = float("nan")
    interval_score_informed: float = float("nan")
    coverage_informed: float = float("nan")
    coverage: float = float("nan")
    sharpness: float = float("nan")
    prediction_rate: float = 0.0
    rejections: list[str] = field(default_factory=list)
    validity_detail: list[dict] = field(default_factory=list)


def _churn(trace) -> tuple[float, float, float]:
    """Split the regret trajectory into forward motion G and backward motion B,
    returning (G, B, kappa).

    The identity R(1) - R(R) = G - B holds. The endpoint score is the *net*, the
    process measure is the *gross loss*.

        κ = B / (G + B) ∈ [0, 1]

    kappa=0 is monotone descent; kappa~0.5 is a random walk. We use a ratio rather
    than the efficiency (G-B)/G because the latter explodes as G approaches zero
    (measured mean -0.24, minimum -7.66, unusable).

    When G+B=0 -- the recommendation never moved -- this returns nan; the endpoint
    regret speaks for those episodes.
    """
    import numpy as _np
    t = _np.asarray(trace, float)
    if t.size < 2:
        return float("nan"), float("nan"), float("nan")
    d = t[:-1] - t[1:]
    G = float(_np.clip(d, 0, None).sum())
    B = float(_np.clip(-d, 0, None).sum())
    return G, B, (B / (G + B) if (G + B) > 1e-12 else float("nan"))


class SlowLabEnv:
    def __init__(self, task, seed: int = 0):
        self.task = copy.deepcopy(task)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Each seed is a different site: TOMGRO parameters are drawn across the J99 three-site spread
        self.truth = ManagedTomgro(seed=seed, factors=self.task.factors,
                                   cycle_days=self.task.cycle_days)
        if getattr(self.task, "train_energy_shock", None) is not None:
            self.truth.econ.energy_shock = float(self.task.train_energy_shock)
            self.truth._oracle = None            # prices changed, so the true optimum must be recomputed
        self.task.noise_sd = self.task.plant_cv * self.truth.response_sd
        self.facility = Facility(task.n_chambers, task.loops_per_chamber)
        self.chamber_eff, self.loop_eff = self.facility.draw_effects(
            self.rng, task.tau_chamber * self.truth.response_sd,
            task.tau_loop * self.truth.response_sd)
        self.round = 0
        self._designs: list[Design] = []
        self._obs: list[Observation] = []
        self._reports: list[ValidityReport] = []
        self._rejections: list[str] = []
        self.calib = CalibrationReport()
        # Irreversibility: a facility clock plus a per-unit "available again at" time, in days
        self.clock = 0.0
        self._available_at = {u.id: 0.0 for u in self.facility.units}
        self.lost_unit_days = 0.0
        self._trace: list[float] = []
        self._cash = 0.0
        self._occupied: set[str] = set()
        self.terminated = False

    # ---------- Common probe set: every agent faces the same forecasting question ----------
    def probe_set(self, k: int = 12) -> np.ndarray:
        """k recipes chosen by the environment rather than the agent, identical for
        every agent at a given seed.

        Why this is needed: if each agent forecasts only the points it chose, the
        forecasting task is not fixed, and a method that spreads points at random
        gains on calibration simply because its own points are easy to predict. A
        fixed probe set turns it into a common question -- you have just spent your
        budget, how well do you actually know this response surface?
        """
        rng = np.random.default_rng(90_000 + self.seed)
        return rng.random((k, self.task.d))

    # ---------- Free and read-only ----------
    @property
    def rounds_left(self) -> int:
        return self.task.n_rounds - self.round

    def interim_recommendation(self, x) -> None:
        """Record, at the end of each round, what the agent would recommend if it
        stopped now. Does not end the episode and does not consume budget.

        This is where the Irreversible axis is read: one round is one irrevocable
        commitment, and we want to know what that commitment bought. It asks for
        the same recommendation, only earlier.
        """
        x = np.clip(np.asarray(x, float).ravel(), 0, 1)
        _, best = self.truth.oracle()
        self._trace.append(best - float(self.truth(x.reshape(1, -1))[0]))

    def available_units(self) -> list[str]:
        """Units currently available -- those inside a heat-damage recovery period
        are excluded. Free and read-only.

        Without this interface an agent cannot replan after damage and
        "irreversible" degrades into "one mistake and you are out", which measures
        fragility rather than adaptation.
        """
        return [u.id for u in self.facility.units
                if self._available_at.get(u.id, 0.0) <= self.clock + 1e-9]

    def observations(self) -> list[Observation]:
        return list(self._obs)

    def as_arrays(self):
        """Completed observations as (X, y), where X is the factor vector of each observation's treatment."""
        X, y = [], []
        for o in self._obs:
            d = self._designs[o.design_id]
            X.append([d.treatments[o.treatment][f.name] for f in self.task.factors])
            y.append(o.value)
        return (np.array(X).reshape(-1, self.task.d), np.array(y)) if X else \
               (np.zeros((0, self.task.d)), np.zeros(0))

    def validate_design(self, design: Design) -> Rejection:
        """Mechanical feasibility only. Free, unlimited, and contains no statistical judgement."""
        t = self.task
        if not design.treatments or not design.allocation:
            return Rejection(RejectCode.SCHEMA, "treatments or allocation is empty")
        for tid in design.allocation:
            if tid not in design.treatments:
                return Rejection(RejectCode.SCHEMA, f"allocation refers to undefined treatment {tid}", [tid])
        for tid, fv in design.treatments.items():
            for f in t.factors:
                if f.name not in fv:
                    return Rejection(RejectCode.SCHEMA, f"treatment {tid} is missing factor {f.name}", [tid])
                if not (0.0 - 1e-9 <= fv[f.name] <= 1.0 + 1e-9):
                    return Rejection(RejectCode.FACTOR_OUT_OF_RANGE,
                                     f"treatment {tid} has {f.name}={fv[f.name]} out of range", [tid])
        uids = design.unit_ids
        for u in uids:
            if u not in self.facility._by_id:
                return Rejection(RejectCode.UNKNOWN_UNIT, f"no such unit {u}", [u])
            if self._available_at.get(u, 0.0) > self.clock + 1e-9:
                return Rejection(
                    RejectCode.UNIT_RECOVERING,
                    f"unit {u} is still in heat-damage recovery, "
                    f"{self._available_at[u] - self.clock:.0f} days remaining", [u])
        if len(set(uids)) != len(uids):
            return Rejection(RejectCode.UNIT_CONFLICT, "one unit assigned to several treatments")
        if len(uids) > t.units_per_round:
            return Rejection(RejectCode.BUDGET_EXCEEDED,
                             f"{len(uids)} units this round exceeds the cap of {t.units_per_round}")
        # Control hierarchy: a chamber-level factor may not take different values inside one chamber
        chamber_factors = [f.name for f in t.factors if f.control_level == "chamber"]
        if chamber_factors:
            byc = {}
            for tid, us in design.allocation.items():
                for u in us:
                    c = self.facility.get(u).chamber
                    key = tuple(round(design.treatments[tid][f], 9) for f in chamber_factors)
                    if c in byc and byc[c] != key:
                        return Rejection(
                            RejectCode.INFEASIBLE_GRANULARITY,
                            f"chamber-level factors take different values inside chamber {c}: {byc[c]} vs {key}")
                    byc[c] = key
        return OK

    # ---------- Consumes resources ----------
    def submit_design(self, design: Design) -> Rejection | int:
        if self.terminated:
            return Rejection(RejectCode.SCHEMA, "the episode has ended")
        if self.rounds_left <= 0:
            return Rejection(RejectCode.BUDGET_EXCEEDED, "no rounds left")
        rej = self.validate_design(design)
        if not rej:
            self._rejections.append(rej.code)      # a rejection costs no budget but is counted
            return rej
        did = len(self._designs)
        self._designs.append(design)
        self._reports.append(score_validity(design, self.facility, self.task))
        self._pending = did
        return did

    def forfeit_round(self) -> None:
        """No executable design was submitted this round -- consume it, but produce no observations.

        Without this, the harness could only break out of the whole episode when
        retries ran out, so *one failed submission ended the entire campaign*. That
        penalty is the wrong severity: a grower who ruins one planting loses that
        season, not every season that remains.
        """
        if hasattr(self, "_pending"):
            del self._pending
        self.round += 1
        self.clock += float(self.task.duration_days)

    def advance(self) -> list[Observation]:
        """Advance to the next event: run the submitted design to completion and return its observations."""
        if not hasattr(self, "_pending"):
            return []
        did = self._pending
        del self._pending
        d = self._designs[did]
        batch_eff = self.rng.normal(0, self.task.tau_batch * self.truth.response_sd)

        # Flatten into a per-unit table and simulate once, vectorised
        rows = [(tid, u) for tid, uids in d.allocation.items() for u in uids]
        if not rows:
            self.round += 1
            return []
        X = np.array([[d.treatments[tid][f.name] for f in self.task.factors]
                      for tid, _ in rows])
        # Plant-to-plant variation is per-plant parameter perturbation rather than
        # noise added to the output: more faithful, and it produces
        # heteroscedasticity and temporal correlation automatically. See
        # world.sample_plant_params.
        # *One unit is the whole stand on one loop* and the observation is their
        # mean, so each unit draws n_p plants and averages them; n_p follows from
        # that unit's density and the loop area.
        dens_name = "density" if "density" in {f.name for f in self.task.factors} else None
        n_p = [self.facility.plants_per_unit(
                   d.treatments[tid][dens_name] if dens_name else 3.0) for tid, _ in rows]
        big = np.repeat(X, n_p, axis=0)
        pp = self.truth.sample_plant_params(len(big), self.rng, self.task.plant_cv)
        comp_all = self.truth(big, plant_params=pp, components=True)
        cut = np.cumsum([0] + list(n_p))
        comp = {k: np.array([v[cut[i]:cut[i + 1]].mean() for i in range(len(rows))])
                for k, v in comp_all.items()}
        base = comp["profit"]

        out = []
        for k, (tid, u) in enumerate(rows):
            c = self.facility.get(u).chamber
            val = float(base[k] + self.chamber_eff[c] + self.loop_eff[u] + batch_eff)
            o = Observation(did, tid, u, val,
                            rev_rate=float(comp["rev_rate"][k]),
                            cost_rate=float(comp["cost_rate"][k]),
                            energy_cost_rate=float(comp["energy_cost_rate"][k]),
                            other_cost_rate=float(comp["other_cost_rate"][k]))
            out.append(o); self._obs.append(o)
            self._occupied.add(u)
            # Process measure: score the *pre-registered* interval forecasts. The
            # forecasts were locked in at submit_design and the observations are
            # only revealed here, which matches P1's information ordering.
            self._cash += val * float(self.task.duration_days)
            self._settle_damage(d, tid, u)
            self.calib.n_units += 1
            pred = d.predictions.get(tid)
            if pred is not None:
                lo, _pt, hi = pred
                self.calib.add(val, float(lo), float(hi), rnd=self.round)
        self.clock += float(self.task.duration_days)
        self.round += 1
        return out

    def _settle_damage(self, design, tid, unit_id):
        """Settle heat damage: units above TCRIT + margin enter a recovery period,
        graded by how far above they went.

        Decoupled from rounds: recovery is counted in *days* from the end of this
        round. A recovery shorter than one cycle (duration_days) blocks only the
        next round; one longer than a cycle blocks two.
        """
        rd = float(getattr(self.task, "recovery_days", 0.0))
        if rd <= 0:
            return
        names = [f.name for f in self.task.factors]
        if "day_temp" in names:
            f = next(g for g in self.task.factors if g.name == "day_temp")
            t_day = f.denorm(design.treatments[tid]["day_temp"])
        else:
            f = next(g for g in MANAGEMENT_FACTORS if g.name == "day_temp")
            t_day = f.denorm(self.truth._fixed.get("day_temp", 0.5))
        tcrit = float(self.truth.params.values["TCRIT"])
        excess = t_day - (tcrit + self.task.damage_margin)
        if excess <= 0:
            return
        days = rd * min(1.0, excess / max(self.task.damage_scale, 1e-9))
        end = self.clock + float(self.task.duration_days)
        self._available_at[unit_id] = max(self._available_at[unit_id], end + days)
        self.lost_unit_days += days

    # ---------- Termination ----------
    def transfer_query(self):
        """At the end of the episode the environment announces an *energy price
        rise*. No further experimental budget is granted.

        What this tests: did the agent learn what this profit surface looks like,
        or how yield, energy cost and other cost each respond to the factors? Only
        the latter can be re-solved after prices move.
        """
        return {"energy_shock": float(self.task.transfer_energy_shock),
                "old_energy_shock": float(self.truth.econ.energy_shock)}

    def submit_recommendation(self, x, agent_name="", probes=None, x_transfer=None) -> EpisodeResult:
        """probes: a (k,3) array of (lower, point, upper) matching the k rows of probe_set()."""
        x = np.clip(np.asarray(x, float).ravel(), 0, 1)
        pc = CalibrationReport()
        if probes is not None:
            P = self.probe_set()
            truth = self.truth(P)
            for (lo, _pt, hi), yv in zip(np.asarray(probes, float), truth):
                pc.n_units += 1
                pc.add(float(yv), float(lo), float(hi))
        _, best = self.truth.oracle()
        regret = best - float(self.truth(x.reshape(1, -1))[0])
        # Cumulative regret: the same reference x* and the same currency as simple
        # regret. It measures how much profit *the experimenting itself* forwent,
        # not how good the final recommendation is.
        cum = 0.0
        for d in self._designs:
            names = [f.name for f in self.task.factors]
            A = np.array([[d.treatments[tid][n] for n in names]
                          for tid, uids in d.allocation.items() for _ in uids], float)
            if len(A):
                cum += float(np.sum(best - self.truth(A))) * float(self.task.duration_days)
        transfer_r = float("nan")
        if x_transfer is not None and getattr(self.task, "transfer_energy_shock", None):
            s = float(self.task.transfer_energy_shock)
            _, tb = self.truth.oracle_at(s)
            xt = np.clip(np.asarray(x_transfer, float).ravel(), 0, 1)
            transfer_r = tb - float(self.truth.profit_at(xt.reshape(1, -1), s)[0])
        tr = np.asarray(self._trace, float)
        if len(tr) >= 2:
            thr = 0.05 * tr[0]
            gains = tr[:-1] - tr[1:]
            wasted = int((gains < thr).sum())
            committed = int(len(gains))
        else:
            wasted, committed = 0, 0
        n_valid = sum(r.passed for r in self._reports)
        used = sum(len(d.unit_ids) for d in self._designs) * self.task.duration_days
        cap = self.task.budget_units * self.task.duration_days
        self.terminated = True
        return EpisodeResult(
            task=self.task.name, agent=agent_name, seed=self.seed,
            simple_regret=regret, recommended=x,
            n_designs=len(self._designs), n_valid=n_valid,
            validity_rate=(n_valid / len(self._reports)) if self._reports else 0.0,
            unit_days_used=used, parallel_utilisation=used / cap if cap else 0.0,
            rejections=list(self._rejections),
            regret_trace=[float(v) for v in tr],
            cumulative_regret=cum,
            gain=_churn(tr)[0], backslide=_churn(tr)[1], churn=_churn(tr)[2],
            campaign_cash=self._cash,
            wasted_rounds=wasted, committed_rounds=committed,
            wasted_fraction=(wasted / committed) if committed else float("nan"),
            lost_unit_days=self.lost_unit_days,
            transfer_regret=float(transfer_r),
            probe_score=pc.interval_score, probe_coverage=pc.coverage,
            probe_sharpness=pc.sharpness,
            interval_score=self.calib.interval_score,
            interval_score_informed=self.calib.informed[0],
            coverage_informed=self.calib.informed[1],
            coverage=self.calib.coverage,
            sharpness=self.calib.sharpness,
            prediction_rate=self.calib.prediction_rate,
            validity_detail=[{"checks": r.checks, "notes": r.notes} for r in self._reports],
        )
