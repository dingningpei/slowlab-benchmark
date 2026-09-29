import copy

import numpy as np
import pytest

from slowlab.design import Design
from slowlab.eig import AtomSet
from slowlab.env import SlowLabEnv
from slowlab.history_likelihood import atom_history_moments, posterior_weights_full_history
from slowlab.history_risk import (conditional_design_value_full_history,
                                  decompose_full_history_risk,
                                  evaluate_history_trajectory)
from slowlab.tasks import TASKS
from slowlab.world import ManagedTomgro


def test_full_history_likelihood_covers_measurements_and_terminal_components():
    task = copy.deepcopy(TASKS["T1"])
    task.cycle_days = 12
    env = SlowLabEnv(task, seed=0)
    treatments = {"a": {"day_temp": 0.5}}
    allocation = {"a": ["c0l0", "c0l1"]}
    env.submit_design(Design(treatments, allocation, randomization_seed=1))
    env.advance_to(5)
    env.observe(modality="canopy_lai")
    env.observe(modality="energy_cost_to_date")
    env.advance()

    worlds = [ManagedTomgro(seed=s, factors=task.factors, cycle_days=task.cycle_days)
              for s in (100, 101)]
    atoms = AtomSet(worlds, np.array([[0.5], [0.5]]), np.array([0, 0]),
                    n_cells=1, nbins=2, task_name=task.name)
    y, means, covariances, records = atom_history_moments(
        atoms, env.history(), task, n_mc_plant=8)
    modalities = [record.modality for record in records]
    assert "canopy_lai" in modalities and "energy_cost_to_date" in modalities
    assert "terminal_revenue_rate" in modalities
    assert "terminal_energy_cost_rate" in modalities
    assert "terminal_other_cost_rate" in modalities
    assert len(y) == means.shape[1] == covariances.shape[1] == covariances.shape[2]
    assert means.shape[0] == covariances.shape[0] == atoms.M
    assert all(np.linalg.eigvalsh(S).min() > 0 for S in covariances)

    weights, used = posterior_weights_full_history(
        atoms, env.history(), task, n_mc_plant=8)
    assert len(used) == len(records)
    assert np.isfinite(weights).all() and weights.sum() == 1.0

    for world in worlds:
        point = np.array([0.5])
        world._oracle = (point, float(world(point.reshape(1, -1))[0]))
    result = decompose_full_history_risk(
        atoms, env.history(), task, np.array([0.5]), cand=np.array([[0.5]]),
        n_mc_plant=8, likelihood_seed=3)
    assert result.posterior_excess_risk >= 0
    assert result.n_observations == len(records)
    assert result.likelihood_seed == 3 and result.n_mc_plant == 8
    assert "mc_error_not_estimated" in result.failure_flags

    before = env.history()[:1]       # submitted design, no outcome records yet
    value = conditional_design_value_full_history(
        atoms, before, env.history(), task, cand=np.array([[0.2], [0.5], [0.8]]),
        n_mc_plant=8, likelihood_seed=3, n_outer=80, simulation_seed=4)
    assert value.n_existing_records == 0
    assert value.n_new_records == len(records)
    assert np.isfinite(value.conditional_value) and value.mc_error >= 0

    env.record_recommendation([0.5], phase="final")
    trajectory = evaluate_history_trajectory(
        atoms, env.history(), task, cand=np.array([[0.2], [0.5], [0.8]]),
        n_mc_plant=8, likelihood_seed=3, n_outer=30, simulation_seed=4)
    assert len(trajectory) == 1 and trajectory[0].phase == "final"
    assert trajectory[0].posterior_excess_risk >= 0


def test_terminal_nuisance_covariance_uses_each_sites_response_scale():
    task = copy.deepcopy(TASKS["Sanity"])
    task.plant_cv = 0.0
    task.tau_chamber, task.tau_loop, task.tau_batch = 0.03, 0.02, 0.01
    env = SlowLabEnv(task, seed=4)
    env.submit_design(Design(
        {"a": {"day_temp": .5}}, {"a": ["c0l0"]}, randomization_seed=1))
    env.advance()
    worlds = [ManagedTomgro(seed=seed, factors=task.factors,
                            cycle_days=task.cycle_days)
              for seed in (101, 109)]
    atoms = AtomSet(worlds, np.array([[.5], [.5]]), np.zeros(2, int),
                    n_cells=1, nbins=2, task_name=task.name)
    _, _, covariances, records = atom_history_moments(
        atoms, env.history(), task, n_mc_plant=2)
    revenue_index = next(i for i, record in enumerate(records)
                         if record.modality == "terminal_revenue_rate")
    resolution_variance = 10.0 ** (-2 * env.TERMINAL_RATE_DECIMALS) / 12.0
    tau2 = task.tau_chamber ** 2 + task.tau_loop ** 2 + task.tau_batch ** 2
    for world, covariance in zip(worlds, covariances):
        expected = tau2 * world.response_sd ** 2 + resolution_variance + 1e-12
        assert covariance[revenue_index, revenue_index] == pytest.approx(
            expected, rel=1e-7)
    assert worlds[0].response_sd != pytest.approx(worlds[1].response_sd)
