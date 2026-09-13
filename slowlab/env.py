"""The SLOWLAB environment (the core of W1).

Design principle: the interface guarantees timing and feasibility, never
scientific correctness -- the latter is the capability under test.
"""
from __future__ import annotations
import copy
import hashlib
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


@dataclass(frozen=True)
class Measurement:
    """One agent-visible, timestamped record from a running crop cycle."""

    design_id: int
    treatment: str
    unit_id: str
    round: int
    day: int
    absolute_day: float
    modality: str
    value: float
    unit: str
    method: str
    cost: float = 0.0
    kind: str = "measured"
    available: bool = True


@dataclass(frozen=True)
class RecommendationUpdate:
    """A recommendation made with the history available at one decision time."""

    round: int
    day: int
    absolute_day: float
    point: tuple[float, ...]
    phase: str = "interim"


@dataclass(frozen=True)
class VisibleEvent:
    """One fact available to every reader at a particular simulation time."""

    seq: int
    kind: str
    round: int
    day: int
    absolute_day: float
    design_id: int | None
    payload: dict


@dataclass
class _ActiveRun:
    """Private execution state for the one design currently in the facility."""

    design_id: int
    start_clock: float
    rows: list[tuple[str, str]]
    plant_counts: list[int]
    cuts: np.ndarray
    state: ManagedTomgro.State
    batch_effect: float


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
        self._measurements: list[Measurement] = []
        self._measurement_cache: dict[tuple[int, int, str, str], Measurement] = {}
        self._recommendation_updates: list[RecommendationUpdate] = []
        self._history: list[VisibleEvent] = []
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
        """Record what the agent recommends with the currently visible history.

        This may be called more than once within a round. It does not change the
        submitted treatment, advance time or consume experimental resources.

        This is where the Irreversible axis is read: one round is one irrevocable
        commitment, and we want to know what that commitment bought. It asks for
        the same recommendation, only earlier.
        """
        x = np.clip(np.asarray(x, float).ravel(), 0, 1)
        _, best = self.truth.oracle()
        self._trace.append(best - float(self.truth(x.reshape(1, -1))[0]))
        self.record_recommendation(x, phase="interim")

    def record_recommendation(self, x, *, phase="final") -> None:
        """Add a visible recommendation without changing the legacy score trace."""
        x = np.clip(np.asarray(x, float).ravel(), 0, 1)
        day = self._active.state.day if hasattr(self, "_active") else 0
        self._recommendation_updates.append(RecommendationUpdate(
            round=self.round, day=int(day), absolute_day=float(self.clock),
            point=tuple(float(v) for v in x), phase=str(phase)))
        self._emit_visible("recommended", design_id=(
            self._active.design_id if hasattr(self, "_active") else None),
            payload={"point": [float(v) for v in x], "phase": str(phase)}, day=int(day))

    def available_units(self) -> list[str]:
        """Units currently available -- those inside a heat-damage recovery period
        are excluded. Free and read-only.

        Without this interface an agent cannot replan after damage and
        "irreversible" degrades into "one mistake and you are out", which measures
        fragility rather than adaptation.
        """
        occupied = set()
        if hasattr(self, "_active"):
            occupied.update(u for _, u in self._active.rows)
        elif hasattr(self, "_pending"):
            occupied.update(self._designs[self._pending].unit_ids)
        return [u.id for u in self.facility.units
                if u.id not in occupied
                and self._available_at.get(u.id, 0.0) <= self.clock + 1e-9]

    def observations(self) -> list[Observation]:
        return list(self._obs)

    def measurements(self) -> list[Measurement]:
        """All measurements created so far, in acquisition order."""
        return list(self._measurements)

    def recommendation_updates(self) -> list[RecommendationUpdate]:
        return list(self._recommendation_updates)

    def history(self, through_day: float | None = None) -> list[VisibleEvent]:
        """Return the common, ordered information set visible to all readers.

        ``through_day`` is an absolute simulation day and is useful for an
        evaluator reconstructing an earlier information set. Future events are
        never returned. Payloads are copied so callers cannot mutate history.
        """
        limit = self.clock if through_day is None else min(float(through_day), self.clock)
        return [copy.deepcopy(event) for event in self._history
                if event.absolute_day <= limit + 1e-9]

    def _emit_visible(self, kind: str, *, design_id: int | None, payload: dict,
                      day: int | None = None, absolute_day: float | None = None) -> None:
        relative_day = (self._active.state.day if day is None and hasattr(self, "_active")
                        else (0 if day is None else int(day)))
        self._history.append(VisibleEvent(
            seq=len(self._history), kind=kind, round=self.round,
            day=int(relative_day),
            absolute_day=float(self.clock if absolute_day is None else absolute_day),
            design_id=design_id, payload=copy.deepcopy(payload)))

    def as_arrays(self):
        """Visible completed history as ``(X, margin)``.

        Scripted readers deliberately consume the same public event history as
        LLMs and evaluators instead of reading private simulator collections.
        """
        X, y = [], []
        designs = {e.design_id: e.payload for e in self.history()
                   if e.kind == "submitted"}
        for event in self.history():
            if event.kind != "completed":
                continue
            d = designs[event.design_id]
            treatment = event.payload["treatment"]
            X.append([d["treatments"][treatment][f.name] for f in self.task.factors])
            y.append(event.payload["value"])
        return (np.array(X).reshape(-1, self.task.d), np.array(y)) if X else \
               (np.zeros((0, self.task.d)), np.zeros(0))

    def component_arrays(self):
        """Visible completed history as factor rows and public response fields."""
        designs = {e.design_id: e.payload for e in self.history()
                   if e.kind == "submitted"}
        X, fields = [], {name: [] for name in (
            "value", "rev_rate", "cost_rate", "energy_cost_rate", "other_cost_rate")}
        for event in self.history():
            if event.kind != "completed":
                continue
            d = designs[event.design_id]
            treatment = event.payload["treatment"]
            X.append([d["treatments"][treatment][f.name] for f in self.task.factors])
            for name in fields:
                fields[name].append(event.payload[name])
        return (np.asarray(X, float).reshape(-1, self.task.d),
                {name: np.asarray(values, float) for name, values in fields.items()})

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
        if hasattr(self, "_pending") or hasattr(self, "_active"):
            return Rejection(RejectCode.SCHEMA, "a design is already running in this round")
        rej = self.validate_design(design)
        if not rej:
            self._rejections.append(rej.code)      # a rejection costs no budget but is counted
            return rej
        did = len(self._designs)
        self._designs.append(design)
        self._reports.append(score_validity(design, self.facility, self.task))
        self._pending = did
        self._emit_visible("submitted", design_id=did, payload={
            "treatments": copy.deepcopy(design.treatments),
            "allocation": copy.deepcopy(design.allocation),
            "randomization_seed": design.randomization_seed,
        })
        return did

    def forfeit_round(self) -> None:
        """No executable design was submitted this round -- consume it, but produce no observations.

        Without this, the harness could only break out of the whole episode when
        retries ran out, so *one failed submission ended the entire campaign*. That
        penalty is the wrong severity: a grower who ruins one planting loses that
        season, not every season that remains.
        """
        if hasattr(self, "_active"):
            raise RuntimeError("a running crop cannot be forfeited; early termination is not implemented")
        if hasattr(self, "_pending"):
            del self._pending
        self._emit_visible(
            "forfeited", design_id=None, payload={"reason": "no executable design"},
            day=self.task.duration_days,
            absolute_day=self.clock + float(self.task.duration_days))
        self.round += 1
        self.clock += float(self.task.duration_days)

    def _start_pending(self) -> _ActiveRun | None:
        """Instantiate plants once and retain their state for the whole round."""
        if hasattr(self, "_active"):
            return self._active
        if not hasattr(self, "_pending"):
            return None
        did = self._pending
        d = self._designs[did]
        batch_eff = self.rng.normal(0, self.task.tau_batch * self.truth.response_sd)
        rows = [(tid, u) for tid, uids in d.allocation.items() for u in uids]
        if not rows:
            del self._pending
            self.round += 1
            return None
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
        cut = np.cumsum([0] + list(n_p))
        self._active = _ActiveRun(
            design_id=did, start_clock=float(self.clock), rows=rows,
            plant_counts=n_p, cuts=cut,
            state=self.truth.start(big, plant_params=pp), batch_effect=float(batch_eff))
        return self._active

    @staticmethod
    def _unit_means(values, active: _ActiveRun) -> np.ndarray:
        values = np.asarray(values, float)
        return np.array([
            values[active.cuts[i]:active.cuts[i + 1]].mean()
            for i in range(len(active.rows))])

    def advance_to(self, day: int) -> list[Observation]:
        """Advance the running round to a relative integer day.

        Days are the simulator's native resolution.  Moving to the terminal day
        completes the round; earlier calls leave final observations unavailable.
        """
        active = self._start_pending()
        if active is None:
            return []
        day = int(day)
        if day < active.state.day:
            raise ValueError(f"cannot move round backwards from day {active.state.day} to {day}")
        if day > self.task.duration_days:
            raise ValueError(
                f"day {day} exceeds round duration {self.task.duration_days}")
        self.truth.advance_state(active.state, day)
        self.clock = active.start_clock + float(day)
        if day < self.task.duration_days:
            return []
        return self._complete_active()

    def _complete_active(self) -> list[Observation]:
        active = self._active
        did = active.design_id
        d = self._designs[did]
        comp_all = self.truth.components(active.state, final=True)
        cut = active.cuts
        comp = {k: np.array([v[cut[i]:cut[i + 1]].mean() for i in range(len(active.rows))])
                for k, v in comp_all.items()}
        base = comp["profit"]

        out = []
        for k, (tid, u) in enumerate(active.rows):
            c = self.facility.get(u).chamber
            nuisance = float(self.chamber_eff[c] + self.loop_eff[u] + active.batch_effect)
            val = float(base[k] + nuisance)
            # Nuisance effects are yield/revenue effects.  Reporting them only in
            # ``value`` made the public fields algebraically inconsistent and let
            # an agent reconstruct a cleaner target by subtracting components.
            rev_rate = float(comp["rev_rate"][k] + nuisance)
            energy_rate = float(comp["energy_cost_rate"][k])
            other_rate = float(comp["other_cost_rate"][k])
            o = Observation(did, tid, u, val,
                            rev_rate=rev_rate,
                            cost_rate=energy_rate + other_rate,
                            energy_cost_rate=energy_rate,
                            other_cost_rate=other_rate)
            out.append(o); self._obs.append(o)
            self._emit_visible("completed", design_id=did, payload={
                "treatment": tid, "unit_id": u, "value": o.value,
                "rev_rate": o.rev_rate, "cost_rate": o.cost_rate,
                "energy_cost_rate": o.energy_cost_rate,
                "other_cost_rate": o.other_cost_rate,
            }, day=self.task.duration_days)
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
        self.round += 1
        del self._pending
        del self._active
        return out

    def advance(self) -> list[Observation]:
        """Run the submitted design to completion (the v1 terminal-only API)."""
        return self.advance_to(self.task.duration_days)

    _MEASUREMENT_SPECS = {
        # bias_sd is persistent within unit/modality; record_sd is specific to a
        # day.  Relative noise is used for cumulative nonnegative quantities.
        "canopy_lai": {"unit": "m2_leaf/m2_ground", "method": "non_destructive_canopy", "bias_sd": .03, "record_sd": .02, "decimals": 3},
        "harvested_fresh_mass": {"unit": "kg/m2", "method": "harvest_ledger", "bias_sd": .015, "record_sd": .01, "decimals": 3},
        "energy_cost_to_date": {"unit": "EUR/m2", "method": "utility_meter", "bias_sd": .003, "record_sd": .002, "decimals": 4},
        "other_cost_to_date": {"unit": "EUR/m2", "method": "cost_ledger", "bias_sd": 0.0, "record_sd": 0.0, "decimals": 4},
    }

    def _stable_normal(self, *parts) -> float:
        raw = "|".join(map(str, (self.seed,) + parts)).encode()
        seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
        return float(np.random.default_rng(seed).normal())

    def observe(self, units=None, modality: str = "canopy_lai") -> list[Measurement]:
        """Read a supported measurement at the current day of a running round.

        Re-reading the same unit, modality and day returns the cached record.  It
        therefore cannot manufacture independent samples by repeated calls.
        """
        active = self._start_pending()
        if active is None:
            raise RuntimeError("there is no running design to observe")
        if modality.startswith("setpoint:"):
            factor_name = modality.split(":", 1)[1]
            factor = next((f for f in self.task.factors if f.name == factor_name), None)
            if factor is None:
                raise ValueError(f"unknown setpoint {factor_name!r}")
            factor_units = {"day_temp": "degC", "night_temp": "degC", "co2": "ppm",
                            "par": "mol_PAR/(m2*d)", "density": "plants/m2",
                            "lai_max": "m2_leaf/m2_ground"}
            spec = {"unit": factor_units.get(factor_name, "factor_unit"),
                    "method": "environment_sensor",
                    "bias_sd": 0.0, "record_sd": 0.0, "decimals": 2}
        else:
            if modality not in self._MEASUREMENT_SPECS:
                raise ValueError(f"unsupported modality {modality!r}")
            spec = self._MEASUREMENT_SPECS[modality]

        requested = list(units) if units is not None else [u for _, u in active.rows]
        row_index = {u: i for i, (_, u) in enumerate(active.rows)}
        unknown = [u for u in requested if u not in row_index]
        if unknown:
            raise ValueError(f"units are not in the running design: {unknown}")

        day = active.state.day
        if modality == "canopy_lai":
            truth = self._unit_means(active.state.LAI, active)
        elif modality == "harvested_fresh_mass":
            wm = self._unit_means(active.state.WM, active)
            truth = np.asarray(self.truth.econ.marketable_kg(wm), float)
        elif modality in {"energy_cost_to_date", "other_cost_to_date"}:
            comp = self.truth.components(active.state)
            truth = self._unit_means(comp[modality], active)
        else:  # exact treatment setpoint in physical units
            factor_name = modality.split(":", 1)[1]
            factor = next(f for f in self.task.factors if f.name == factor_name)
            truth = np.array([
                factor.denorm(self._designs[active.design_id].treatments[tid][factor_name])
                for tid, _ in active.rows], float)

        result = []
        for unit_id in requested:
            key = (active.design_id, day, unit_id, modality)
            if key not in self._measurement_cache:
                i = row_index[unit_id]
                bias = spec["bias_sd"] * self._stable_normal(active.design_id, unit_id, modality, "bias")
                error = spec["record_sd"] * self._stable_normal(active.design_id, unit_id, modality, day)
                value = round(float(max(0.0, truth[i] * (1.0 + bias + error))),
                              int(spec["decimals"]))
                tid = active.rows[i][0]
                m = Measurement(
                    design_id=active.design_id, treatment=tid, unit_id=unit_id,
                    round=self.round, day=day,
                    absolute_day=active.start_clock + day, modality=modality,
                    value=value, unit=spec["unit"], method=spec["method"], cost=0.0)
                self._measurement_cache[key] = m
                self._measurements.append(m)
                self._emit_visible("measured", design_id=active.design_id, payload={
                    "treatment": tid, "unit_id": unit_id, "modality": modality,
                    "value": value, "unit": spec["unit"], "method": spec["method"],
                    "cost": 0.0, "available": True,
                }, day=day)
            result.append(self._measurement_cache[key])
        return result

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
        end = (self._active.start_clock + float(self.task.duration_days)
               if hasattr(self, "_active") else self.clock + float(self.task.duration_days))
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
