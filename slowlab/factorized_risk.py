"""Risk evaluation on independent continuous posterior samples."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .factorized_history import FactorizedHistoryPosterior


@dataclass
class FactorizedRiskDecomposition:
    estimator_version: str
    bayes_action: np.ndarray
    evaluation_bayes_action: np.ndarray
    bayes_risk: float
    selected_bayes_action_risk: float
    posterior_selection_gap: float
    agent_posterior_risk: float
    posterior_excess_risk: float
    search_joint_ess: float
    evaluation_joint_ess: float
    n_action_candidates: int
    n_best_candidates: int
    failure_flags: tuple[str, ...]


def decompose_factorized_history_risk(
        search: FactorizedHistoryPosterior,
        evaluation: FactorizedHistoryPosterior,
        agent_point, *, action_candidates, best_candidates):
    """Select with one posterior sample and score with an independent sample.

    ``best_candidates`` is disjoint from the action-search set and supplies a
    feasible lower bound on each world's true optimum. The reported flag keeps
    that numerical limitation explicit. Agent excess is measured against the
    evaluation posterior's own Bayes action and is nonnegative because the agent
    point is inserted into both action sets before minimisation.
    """
    agent_point = np.asarray(agent_point, float).reshape(1, -1)
    action_candidates = np.vstack([
        np.atleast_2d(np.asarray(action_candidates, float)), agent_point])
    best_candidates = np.atleast_2d(np.asarray(best_candidates, float))
    if action_candidates.shape[1] != agent_point.shape[1]:
        raise ValueError("action candidate dimension does not match agent point")
    if best_candidates.shape[1] != agent_point.shape[1]:
        raise ValueError("best candidate dimension does not match agent point")

    search_response = search.mean_response(action_candidates)
    search_action = int(np.argmax(search.weights @ search_response))
    bayes_action = action_candidates[search_action]

    evaluation_response = evaluation.mean_response(action_candidates)
    independent_best = evaluation.mean_response(best_candidates).max(axis=1)
    best = np.maximum(independent_best, evaluation_response.max(axis=1))
    regret = best[:, None] - evaluation_response
    risks = evaluation.weights @ regret
    evaluation_action = int(np.argmin(risks))
    bayes_risk = float(risks[evaluation_action])
    selected_risk = float(risks[search_action])
    agent_risk = float(risks[-1])
    excess = agent_risk - bayes_risk
    selection_gap = selected_risk - bayes_risk
    if excess < -1e-10 or selection_gap < -1e-10:
        raise ArithmeticError("held-out posterior risk difference is negative")
    return FactorizedRiskDecomposition(
        estimator_version="history-risk-0.3-factorized-heldout",
        bayes_action=bayes_action.copy(),
        evaluation_bayes_action=action_candidates[evaluation_action].copy(),
        bayes_risk=bayes_risk,
        selected_bayes_action_risk=selected_risk,
        posterior_selection_gap=selection_gap,
        agent_posterior_risk=agent_risk,
        posterior_excess_risk=excess,
        search_joint_ess=search.effective_sample_size,
        evaluation_joint_ess=evaluation.effective_sample_size,
        n_action_candidates=len(action_candidates),
        n_best_candidates=len(best_candidates),
        failure_flags=("finite_best_lower_bound",))
