"""Risk quantities conditioned on one common, timestamped visible history.

The terminal-only estimator uses the complete joint covariance across rounds,
including persistent chamber and loop effects. The full-field estimator also
uses the joint biological and measurement likelihood in history_likelihood.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .achievable import candidate_set, response_table
from .history_likelihood import (atom_history_moments,
                                 posterior_weights_from_moments,
                                 posterior_weights_full_history,
                                 visible_likelihood_records)


ESTIMATOR_VERSION = "history-risk-0.1-terminal"


class IncompleteLikelihoodError(RuntimeError):
    pass


@dataclass
class HistoryTable:
    X: np.ndarray
    y: np.ndarray
    chamber: np.ndarray
    unit: np.ndarray
    batch: np.ndarray

    def __len__(self):
        return len(self.y)


@dataclass
class RiskDecomposition:
    estimator_version: str
    n_observations: int
    prior_bayes_risk: float
    bayes_risk: float
    agent_posterior_risk: float
    posterior_excess_risk: float
    bayes_action: np.ndarray
    posterior_weights: np.ndarray
    temperature: float
    likelihood_seed: int | None = None
    n_mc_plant: int | None = None
    mc_error: float = float("nan")
    failure_flags: tuple[str, ...] = ()


@dataclass
class ConditionalDesignValue:
    estimator_version: str
    bayes_risk_before: float
    expected_bayes_risk_after: float
    conditional_value: float
    mc_error: float
    n_existing_records: int
    n_new_records: int
    n_outer: int
    likelihood_seed: int
    simulation_seed: int
    failure_flags: tuple[str, ...] = ()


@dataclass
class RiskTrajectoryPoint:
    event_seq: int
    round: int
    day: int
    absolute_day: float
    phase: str
    bayes_risk: float
    agent_posterior_risk: float
    posterior_excess_risk: float
    conditional_value_since_previous: float
    conditional_value_mc_error: float
    failure_flags: tuple[str, ...] = ()


def terminal_history(events, task, *, reject_measurements=True) -> HistoryTable:
    events = list(events)
    if reject_measurements and any(e.kind == "measured" for e in events):
        raise IncompleteLikelihoodError(
            "within-cycle measurements require the calibrated temporal likelihood")
    designs = {e.design_id: e.payload for e in events if e.kind == "submitted"}
    X, y, chamber, unit, batch = [], [], [], [], []
    for event in events:
        if event.kind != "completed":
            continue
        payload = event.payload
        design = designs[event.design_id]
        treatment = payload["treatment"]
        X.append([design["treatments"][treatment][f.name] for f in task.factors])
        y.append(payload["value"])
        uid = str(payload["unit_id"])
        chamber.append(uid.split("l", 1)[0])
        unit.append(uid)
        batch.append(str(event.design_id))
    return HistoryTable(
        X=np.asarray(X, float).reshape(-1, task.d), y=np.asarray(y, float),
        chamber=np.asarray(chamber, object), unit=np.asarray(unit, object),
        batch=np.asarray(batch, object))


def _membership(labels):
    labels = np.asarray(labels, object)
    if len(labels) == 0:
        return np.zeros((0, 0))
    _, inv = np.unique(labels, return_inverse=True)
    return np.eye(int(inv.max()) + 1)[inv]


def history_covariance(table, plant_variance, tau_chamber, tau_loop, tau_batch):
    """Joint covariance with chamber/loop persistence and round-level batches."""
    n = len(table)
    Zc, Zu, Zb = (_membership(table.chamber), _membership(table.unit),
                  _membership(table.batch))
    return (np.eye(n) * float(plant_variance)
            + float(tau_chamber) ** 2 * Zc @ Zc.T
            + float(tau_loop) ** 2 * Zu @ Zu.T
            + float(tau_batch) ** 2 * Zb @ Zb.T)


def posterior_weights_history(atoms, table, tau_chamber, tau_loop, tau_batch,
                              *, prior=None, temperature=1.0):
    prior = (np.full(atoms.M, 1.0 / atoms.M) if prior is None
             else np.asarray(prior, float))
    if len(table) == 0:
        return prior / prior.sum()
    mu = atoms.mean_response(table.X)
    S = history_covariance(table, atoms.s2_ref, tau_chamber, tau_loop, tau_batch)
    Si = np.linalg.inv(S + 1e-12 * np.eye(len(table)))
    residual = table.y[None, :] - mu
    ll = -0.5 * np.einsum("mi,ij,mj->m", residual, Si, residual) / float(temperature)
    weights = np.exp(ll - ll.max()) * prior
    return weights / weights.sum()


def decompose_history_risk(atoms, events, task, agent_point, *,
                           tau_chamber, tau_loop, tau_batch, cand=None,
                           temperature=1.0, candidate_seed=0):
    """Compute ``b(H)`` and agent excess risk from exactly the same history.

    The agent point is inserted into the action set before minimisation. Hence
    posterior excess risk is nonnegative by construction of the estimand, apart
    from floating-point tolerance; it is never clipped after calculation.
    """
    table = terminal_history(events, task)
    agent_point = np.asarray(agent_point, float).reshape(1, -1)
    if cand is None:
        cand = candidate_set(atoms, rng=np.random.default_rng(candidate_seed))
    cand = np.vstack([np.asarray(cand, float), agent_point])
    F = response_table(atoms, cand)
    best = np.asarray([world.oracle()[1] for world in atoms.worlds], float)
    regret = best[:, None] - F
    prior = np.full(atoms.M, 1.0 / atoms.M)
    weights = posterior_weights_history(
        atoms, table, tau_chamber, tau_loop, tau_batch,
        prior=prior, temperature=temperature)
    posterior_risk = weights @ regret
    bayes_index = int(np.argmin(posterior_risk))
    agent_risk = float(posterior_risk[-1])
    bayes_risk = float(posterior_risk[bayes_index])
    excess = agent_risk - bayes_risk
    if excess < -1e-10:
        raise ArithmeticError(f"posterior excess risk is negative: {excess}")
    return RiskDecomposition(
        estimator_version=ESTIMATOR_VERSION, n_observations=len(table),
        prior_bayes_risk=float(np.min(prior @ regret)),
        bayes_risk=bayes_risk, agent_posterior_risk=agent_risk,
        posterior_excess_risk=float(excess), bayes_action=cand[bayes_index].copy(),
        posterior_weights=weights.copy(), temperature=float(temperature))


def decompose_full_history_risk(atoms, events, task, agent_point, *, cand=None,
                                temperature=1.0, candidate_seed=0,
                                n_mc_plant=64, likelihood_seed=0):
    """Risk decomposition using every independent visible numeric field."""
    events = list(events)
    agent_point = np.asarray(agent_point, float).reshape(1, -1)
    if cand is None:
        cand = candidate_set(atoms, rng=np.random.default_rng(candidate_seed))
    cand = np.vstack([np.asarray(cand, float), agent_point])
    response = response_table(atoms, cand)
    best = np.asarray([world.oracle()[1] for world in atoms.worlds], float)
    regret = best[:, None] - response
    prior = np.full(atoms.M, 1.0 / atoms.M)
    weights, records = posterior_weights_full_history(
        atoms, events, task, prior=prior, temperature=temperature,
        n_mc_plant=n_mc_plant, seed=likelihood_seed)
    risks = weights @ regret
    bayes_index = int(np.argmin(risks))
    excess = float(risks[-1] - risks[bayes_index])
    if excess < -1e-10:
        raise ArithmeticError(f"posterior excess risk is negative: {excess}")
    return RiskDecomposition(
        estimator_version="history-risk-0.2-full-fields",
        n_observations=len(records), prior_bayes_risk=float(np.min(prior @ regret)),
        bayes_risk=float(risks[bayes_index]), agent_posterior_risk=float(risks[-1]),
        posterior_excess_risk=excess, bayes_action=cand[bayes_index].copy(),
        posterior_weights=weights.copy(), temperature=float(temperature),
        likelihood_seed=int(likelihood_seed), n_mc_plant=int(n_mc_plant),
        failure_flags=("mc_error_not_estimated",))


def conditional_design_value_full_history(
        atoms, before_events, after_events, task, *, cand=None,
        temperature=1.0, candidate_seed=0, n_mc_plant=64,
        likelihood_seed=0, n_outer=200, simulation_seed=0):
    """Expected reduction in ``b(H)`` from the newly acquired record geometry.

    ``after_events`` must contain ``before_events`` as a prefix. Values in its
    new records define only which fields were acquired; the expectation samples
    replacement values from the posterior predictive distribution conditional
    on the old history. For adaptively selected future measurement schedules,
    this evaluates the realised acquisition set rather than the selection policy.
    """
    before_events, after_events = list(before_events), list(after_events)
    if after_events[:len(before_events)] != before_events:
        raise ValueError("before_events must be a prefix of after_events")
    old_records = visible_likelihood_records(before_events)
    y, means, covariances, records = atom_history_moments(
        atoms, after_events, task, n_mc_plant=n_mc_plant, seed=likelihood_seed)
    n_old = len(old_records)
    if records[:n_old] != old_records:
        raise ValueError("likelihood records from before_events are not a stable prefix")
    n_new = len(records) - n_old
    if n_new <= 0:
        return ConditionalDesignValue(
            "history-risk-0.2-full-fields", float("nan"), float("nan"), 0.0,
            0.0, n_old, 0, int(n_outer), int(likelihood_seed),
            int(simulation_seed), ("no_new_records",))

    if cand is None:
        cand = candidate_set(atoms, rng=np.random.default_rng(candidate_seed))
    cand = np.asarray(cand, float)
    response = response_table(atoms, cand)
    best = np.asarray([world.oracle()[1] for world in atoms.worlds], float)
    regret = best[:, None] - response
    prior = np.full(atoms.M, 1.0 / atoms.M)
    y_old = y[:n_old]
    weights_before = posterior_weights_from_moments(
        y_old, means[:, :n_old], covariances[:, :n_old, :n_old],
        prior=prior, temperature=temperature)
    risk_before = float(np.min(weights_before @ regret))

    rng = np.random.default_rng(simulation_seed)
    truth_atoms = rng.choice(atoms.M, size=n_outer, p=weights_before)
    after_risks = np.empty(n_outer)
    for k, truth_index in enumerate(truth_atoms):
        mu = means[truth_index]
        S = covariances[truth_index]
        if n_old:
            Soo, Son = S[:n_old, :n_old], S[:n_old, n_old:]
            Sno, Snn = S[n_old:, :n_old], S[n_old:, n_old:]
            gain = np.linalg.solve(Soo, Son).T
            conditional_mean = mu[n_old:] + gain @ (y_old - mu[:n_old])
            conditional_cov = Snn - gain @ Son
        else:
            conditional_mean, conditional_cov = mu, S
        L = np.linalg.cholesky(conditional_cov + 1e-12 * np.eye(n_new))
        y_new = conditional_mean + L @ rng.standard_normal(n_new)
        y_full = np.concatenate([y_old, y_new])
        weights_after = posterior_weights_from_moments(
            y_full, means, covariances, prior=prior, temperature=temperature)
        after_risks[k] = float(np.min(weights_after @ regret))
    expected_after = float(after_risks.mean())
    value = risk_before - expected_after
    mc_error = float(after_risks.std(ddof=1) / np.sqrt(n_outer)) if n_outer > 1 else float("nan")
    flags = []
    if value < -2.0 * mc_error:
        flags.append("negative_conditional_value")
    return ConditionalDesignValue(
        estimator_version="history-risk-0.2-full-fields",
        bayes_risk_before=risk_before,
        expected_bayes_risk_after=expected_after, conditional_value=float(value),
        mc_error=mc_error, n_existing_records=n_old, n_new_records=n_new,
        n_outer=int(n_outer), likelihood_seed=int(likelihood_seed),
        simulation_seed=int(simulation_seed), failure_flags=tuple(flags))


def evaluate_history_trajectory(
        atoms, events, task, *, cand=None, temperature=1.0,
        candidate_seed=0, n_mc_plant=64, likelihood_seed=0,
        n_outer=200, simulation_seed=0):
    """Evaluate every visible recommendation against its exact history prefix."""
    events = list(events)
    recommendations = [event for event in events if event.kind == "recommended"
                       and event.payload.get("phase", "interim") != "transfer"]
    if not recommendations:
        return []
    if cand is None:
        cand = candidate_set(atoms, rng=np.random.default_rng(candidate_seed))
    points = np.asarray([event.payload["point"] for event in recommendations], float)
    shared_cand = np.vstack([np.asarray(cand, float), points])
    out, previous = [], []
    for index, event in enumerate(recommendations):
        prefix = events[:event.seq + 1]
        risk = decompose_full_history_risk(
            atoms, prefix, task, np.asarray(event.payload["point"], float),
            cand=shared_cand, temperature=temperature,
            n_mc_plant=n_mc_plant, likelihood_seed=likelihood_seed)
        value = conditional_design_value_full_history(
            atoms, previous, prefix, task, cand=shared_cand,
            temperature=temperature, n_mc_plant=n_mc_plant,
            likelihood_seed=likelihood_seed, n_outer=n_outer,
            simulation_seed=simulation_seed + index)
        flags = tuple(sorted(set(risk.failure_flags + value.failure_flags)))
        out.append(RiskTrajectoryPoint(
            event_seq=event.seq, round=event.round, day=event.day,
            absolute_day=event.absolute_day,
            phase=str(event.payload.get("phase", "interim")),
            bayes_risk=risk.bayes_risk,
            agent_posterior_risk=risk.agent_posterior_risk,
            posterior_excess_risk=risk.posterior_excess_risk,
            conditional_value_since_previous=value.conditional_value,
            conditional_value_mc_error=value.mc_error,
            failure_flags=flags))
        previous = prefix
    return out
