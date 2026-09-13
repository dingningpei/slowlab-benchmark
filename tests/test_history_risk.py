from dataclasses import dataclass

import numpy as np
import pytest

from slowlab.env import VisibleEvent
from slowlab.history_risk import (
    IncompleteLikelihoodError, decompose_history_risk, history_covariance,
    terminal_history,
)
from slowlab.tasks import TASKS


class _World:
    def __init__(self, optimum):
        self.optimum = float(optimum)

    def __call__(self, X):
        x = np.asarray(X, float)[:, 0]
        return -(x - self.optimum) ** 2

    def oracle(self):
        return np.array([self.optimum]), 0.0


@dataclass
class _Atoms:
    worlds: list
    xstar: np.ndarray
    s2_ref: float = 0.01

    @property
    def M(self):
        return len(self.worlds)

    def mean_response(self, X):
        return np.stack([world(X) for world in self.worlds])


def _events(value=-0.01):
    return [
        VisibleEvent(0, "submitted", 0, 0, 0.0, 0, {
            "treatments": {"a": {"day_temp": 0.2}},
            "allocation": {"a": ["c0l0"]}, "randomization_seed": 1}),
        VisibleEvent(1, "completed", 0, 210, 210.0, 0, {
            "treatment": "a", "unit_id": "c0l0", "value": value,
            "rev_rate": 0.1, "cost_rate": 0.11,
            "energy_cost_rate": 0.05, "other_cost_rate": 0.06}),
    ]


def test_excess_risk_uses_same_history_and_is_nonnegative_without_clipping():
    atoms = _Atoms([_World(0.2), _World(0.8)], np.array([[0.2], [0.8]]))
    result = decompose_history_risk(
        atoms, _events(), TASKS["T1"], np.array([0.8]),
        tau_chamber=0.01, tau_loop=0.01, tau_batch=0.01,
        cand=np.array([[0.2], [0.8]]))
    assert result.n_observations == 1
    assert result.posterior_excess_risk >= 0
    assert result.agent_posterior_risk == pytest.approx(
        result.bayes_risk + result.posterior_excess_risk)


def test_joint_covariance_retains_chamber_and_loop_effects_across_rounds():
    task = TASKS["T1"]
    events = _events() + [
        VisibleEvent(2, "submitted", 1, 0, 210.0, 1, {
            "treatments": {"a": {"day_temp": 0.2}},
            "allocation": {"a": ["c0l0"]}, "randomization_seed": 2}),
        VisibleEvent(3, "completed", 1, 210, 420.0, 1, {
            "treatment": "a", "unit_id": "c0l0", "value": -0.02,
            "rev_rate": 0.1, "cost_rate": 0.12,
            "energy_cost_rate": 0.05, "other_cost_rate": 0.07}),
    ]
    table = terminal_history(events, task)
    S = history_covariance(table, 0.01, 0.2, 0.1, 0.3)
    assert S[0, 1] == pytest.approx(0.2 ** 2 + 0.1 ** 2)
    assert S[0, 0] == pytest.approx(0.01 + 0.2 ** 2 + 0.1 ** 2 + 0.3 ** 2)


def test_measurements_are_rejected_until_temporal_likelihood_is_implemented():
    events = _events()
    events.insert(1, VisibleEvent(1, "measured", 0, 60, 60.0, 0, {
        "treatment": "a", "unit_id": "c0l0", "modality": "canopy_lai",
        "value": 1.2, "unit": "m2/m2", "method": "sensor"}))
    with pytest.raises(IncompleteLikelihoodError):
        terminal_history(events, TASKS["T1"])
