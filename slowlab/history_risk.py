"""Risk quantities conditioned on one common, timestamped visible history.

This module is the terminal-observation foundation for the v2 evaluator. It
uses the complete joint covariance across rounds, including persistent chamber
and loop effects. Within-cycle measurements are rejected until their
time-correlated biological likelihood is calibrated; silently treating them as
independent terminal observations would recreate the problem Phase 2 fixes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .achievable import candidate_set, response_table


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
