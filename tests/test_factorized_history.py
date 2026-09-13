import copy

import numpy as np

from slowlab.design import Design
from slowlab.env import SlowLabEnv
from slowlab.factorized_history import factorized_history_posterior
from slowlab.tasks import TASKS


def _history(cost_only):
    task = copy.deepcopy(TASKS["Sanity"])
    task.cycle_days = 20
    env = SlowLabEnv(task, seed=7)
    env.submit_design(Design(
        {"cool": {"day_temp": .3}, "warm": {"day_temp": .7}},
        {"cool": ["c0l0"], "warm": ["c1l0"]}, randomization_seed=3))
    env.advance_to(10)
    env.observe(modality="energy_cost_to_date")
    if not cost_only:
        env.observe(modality="canopy_lai")
    return task, env.history()


def test_cost_evidence_preserves_independent_crop_prior():
    task, history = _history(cost_only=True)
    posterior = factorized_history_posterior(
        history, task, crop_seeds=(100, 101, 102),
        n_economic_particles=32, n_economic_product=8,
        n_mc_plant=4, mcmc_steps=2)
    np.testing.assert_allclose(
        posterior.crop_marginal, np.full(3, 1 / 3), atol=1e-12)


def test_non_cost_records_update_product_with_finite_normalised_weights():
    task, history = _history(cost_only=False)
    posterior = factorized_history_posterior(
        history, task, crop_seeds=(100, 101, 102),
        n_economic_particles=32, n_economic_product=8,
        n_mc_plant=4, mcmc_steps=2)
    assert np.isfinite(posterior.weights).all()
    assert np.isclose(posterior.weights.sum(), 1.0)
    assert np.isclose(posterior.crop_marginal.sum(), 1.0)
    assert np.isclose(posterior.economic_marginal.sum(), 1.0)
