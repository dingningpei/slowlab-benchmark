import copy

import numpy as np

from slowlab.design import Design
from slowlab.eig import AtomSet
from slowlab.env import SlowLabEnv
from slowlab.history_likelihood import atom_history_moments, posterior_weights_full_history
from slowlab.history_risk import decompose_full_history_risk
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
