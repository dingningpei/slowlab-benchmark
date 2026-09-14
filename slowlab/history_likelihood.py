"""Joint likelihood for every non-redundant field visible in a v2 history."""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from .env import SlowLabEnv
from .facility import Facility


@dataclass(frozen=True)
class LikelihoodRecord:
    kind: str
    modality: str
    design_id: int
    treatment: str
    unit_id: str
    chamber: str
    day: int
    value: float


def visible_likelihood_records(events):
    """Return an independent basis for all numeric observations in ``H_t``.

    Terminal margin is omitted because it is exactly revenue-energy-other.
    Repeated exact cost fields for replicates of one treatment are also
    deduplicated; they contain no additional information.
    """
    records, exact_seen = [], set()
    for event in events:
        p = event.payload
        if event.kind == "measured":
            modality = str(p["modality"])
            if modality.startswith("setpoint:"):
                continue                       # a known action, not evidence about theta
            uid = str(p["unit_id"])
            records.append(LikelihoodRecord(
                "measured", modality, int(event.design_id), str(p["treatment"]),
                uid, uid.split("l", 1)[0], int(event.day), float(p["value"])))
        elif event.kind == "completed":
            uid, tid, did = str(p["unit_id"]), str(p["treatment"]), int(event.design_id)
            records.append(LikelihoodRecord(
                "completed", "terminal_revenue_rate", did, tid, uid,
                uid.split("l", 1)[0], int(event.day), float(p["rev_rate"])))
            for modality, field in (("terminal_energy_cost_rate", "energy_cost_rate"),
                                    ("terminal_other_cost_rate", "other_cost_rate")):
                key = did, tid, modality
                if key not in exact_seen:
                    exact_seen.add(key)
                    records.append(LikelihoodRecord(
                        "completed", modality, did, tid, uid,
                        uid.split("l", 1)[0], int(event.day), float(p[field])))
    return records


def _trajectory_samples(world, X, task, days_and_modalities, n_mc, rng):
    X = np.repeat(np.asarray(X, float).reshape(1, -1), n_mc, axis=0)
    params = world.sample_plant_params(n_mc, rng, task.plant_cv)
    state = world.start(X, plant_params=params)
    values = {}
    for day in sorted({day for day, _ in days_and_modalities}):
        world.advance_state(state, day)
        comp = None
        for _, modality in [x for x in days_and_modalities if x[0] == day]:
            if modality == "canopy_lai":
                value = state.LAI
            elif modality == "harvested_fresh_mass":
                value = world.econ.marketable_kg(state.WM)
            else:
                comp = world.components(state, final=(day == task.duration_days)) if comp is None else comp
                lookup = {
                    "energy_cost_to_date": "energy_cost_to_date",
                    "other_cost_to_date": "other_cost_to_date",
                    "terminal_revenue_rate": "rev_rate",
                    "terminal_energy_cost_rate": "energy_cost_rate",
                    "terminal_other_cost_rate": "other_cost_rate",
                }
                value = comp[lookup[modality]]
            values[(day, modality)] = np.asarray(value, float)
    return values


def atom_history_moments(atoms, events, task, *, n_mc_plant=64, seed=0):
    """Return observed vector and atom-specific Gaussian moments.

    Plant trajectories are Monte Carlo integrated per distinct treatment. Their
    covariance is divided by the number of plants in an experimental unit.
    Measurement errors then use the same persistent-bias and per-day SDs as the
    environment. This is an approximation to the clipped multiplicative model;
    calibration must report sensitivity to ``n_mc_plant``.
    """
    events = list(events)
    records = visible_likelihood_records(events)
    y = np.asarray([record.value for record in records], float)
    n = len(records)
    if n == 0:
        return y, np.zeros((atoms.M, 0)), np.zeros((atoms.M, 0, 0)), records
    designs = {e.design_id: e.payload for e in events if e.kind == "submitted"}
    facility = Facility(task.n_chambers, task.loops_per_chamber)
    by_treatment = defaultdict(list)
    for i, record in enumerate(records):
        by_treatment[(record.design_id, record.treatment)].append(i)

    means = np.zeros((atoms.M, n))
    covariances = np.zeros((atoms.M, n, n))
    for m, world in enumerate(atoms.worlds):
        rng = np.random.default_rng(seed + 104729 * (m + 1))
        biological = np.zeros((n, n))
        for (did, tid), indices in by_treatment.items():
            treatment = designs[did]["treatments"][tid]
            X = [treatment[f.name] for f in task.factors]
            keys = {(records[i].day, records[i].modality) for i in indices}
            samples = _trajectory_samples(world, X, task, keys, n_mc_plant, rng)
            for i in indices:
                means[m, i] = float(samples[(records[i].day, records[i].modality)].mean())
            for i in indices:
                for j in indices:
                    if records[i].unit_id != records[j].unit_id:
                        continue
                    a = samples[(records[i].day, records[i].modality)]
                    b = samples[(records[j].day, records[j].modality)]
                    density_factor = next(
                        (factor for factor in task.factors
                         if factor.name == "density"), None)
                    density = (density_factor.denorm(treatment["density"])
                               if density_factor is not None else 3.0)
                    n_plants = facility.plants_per_unit(density)
                    biological[i, j] = float(np.cov(a, b, ddof=1)[0, 1] / n_plants)

        S = biological.copy()
        # Measurement noise: a unit/modality bias persists across dates, while
        # record error is independent between dates. Both are multiplicative.
        for i, ri in enumerate(records):
            if ri.kind != "measured":
                continue
            spec = SlowLabEnv._MEASUREMENT_SPECS[ri.modality]
            for j, rj in enumerate(records):
                if (rj.kind != "measured" or ri.design_id != rj.design_id
                        or ri.unit_id != rj.unit_id or ri.modality != rj.modality):
                    continue
                second_moment = biological[i, j] + means[m, i] * means[m, j]
                S[i, j] += spec["bias_sd"] ** 2 * second_moment
                if i == j:
                    S[i, i] += spec["record_sd"] ** 2 * second_moment
                    resolution = 10.0 ** (-int(spec["decimals"]))
                    S[i, i] += resolution ** 2 / 12.0

        scale = world.response_sd
        for i, ri in enumerate(records):
            if ri.modality != "terminal_revenue_rate":
                continue
            for j, rj in enumerate(records):
                if rj.modality != "terminal_revenue_rate":
                    continue
                if ri.chamber == rj.chamber:
                    S[i, j] += (task.tau_chamber * scale) ** 2
                if ri.unit_id == rj.unit_id:
                    S[i, j] += (task.tau_loop * scale) ** 2
                if ri.design_id == rj.design_id:
                    S[i, j] += (task.tau_batch * scale) ** 2
        # Completed rates are displayed accounting records. Treat their last
        # reported digit as an interval rather than an exact equality on the
        # continuous site parameters. Margin is omitted from the independent
        # basis, so the three retained components receive independent rounding
        # variances here.
        terminal_resolution = 10.0 ** (-SlowLabEnv.TERMINAL_RATE_DECIMALS)
        for i, record in enumerate(records):
            if record.kind == "completed":
                S[i, i] += terminal_resolution ** 2 / 12.0
        # Exact ledger fields can make S singular. Jitter is numerical only and
        # far below displayed precision; duplicate exact readings were removed.
        covariances[m] = S + 1e-12 * np.eye(n)
    return y, means, covariances, records


def posterior_weights_full_history(atoms, events, task, *, prior=None,
                                   temperature=1.0, n_mc_plant=64, seed=0):
    y, means, covariances, records = atom_history_moments(
        atoms, events, task, n_mc_plant=n_mc_plant, seed=seed)
    weights = (np.full(atoms.M, 1.0 / atoms.M) if prior is None
               else np.asarray(prior, float).copy())
    if len(y) == 0:
        return weights / weights.sum(), records
    return posterior_weights_from_moments(
        y, means, covariances, prior=weights, temperature=temperature), records


def posterior_weights_from_moments(y, means, covariances, *, prior=None,
                                   temperature=1.0):
    """Posterior weights for an already constructed atom-specific Gaussian."""
    y, means, covariances = (np.asarray(y, float), np.asarray(means, float),
                             np.asarray(covariances, float))
    M = len(means)
    weights = (np.full(M, 1.0 / M) if prior is None
               else np.asarray(prior, float).copy())
    if len(y) == 0:
        return weights / weights.sum()
    log_likelihood = np.empty(M)
    for m in range(M):
        S = covariances[m]
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            raise np.linalg.LinAlgError("history covariance is not positive definite")
        residual = y - means[m]
        log_likelihood[m] = (-0.5 * (residual @ np.linalg.solve(S, residual) + logdet)
                             / float(temperature))
    z = log_likelihood - log_likelihood.max()
    weights *= np.exp(z)
    return weights / weights.sum()
