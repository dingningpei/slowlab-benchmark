import copy

import numpy as np

from slowlab.design import Design
from slowlab.economic_posterior import (economic_history_moments,
                                        sample_economic_posterior)
from slowlab.economics import sample_site_econ, site_econ_from_unit
from slowlab.env import SlowLabEnv
from slowlab.tasks import TASKS


def _cost_history(seed=7):
    task = copy.deepcopy(TASKS["Sanity"])
    env = SlowLabEnv(task, seed=seed)
    treatments = {"cool": {"day_temp": .3}, "warm": {"day_temp": .7}}
    allocation = {"cool": ["c0l0"], "warm": ["c1l0"]}
    env.submit_design(Design(treatments, allocation, randomization_seed=3))
    env.advance_to(60)
    env.observe(modality="energy_cost_to_date")
    env.advance()
    return task, env.history()


def test_explicit_unit_map_preserves_seeded_economic_prior():
    seed = 42
    rng = np.random.default_rng(70_000 + seed)
    mapped = site_econ_from_unit(rng.random(6))
    sampled = sample_site_econ(seed)
    assert vars(mapped) == vars(sampled)


def test_economic_smc_reaches_narrow_cost_likelihood_without_atom_collapse():
    task, history = _cost_history()
    posterior = sample_economic_posterior(
        history, task, n_particles=96, seed=11, mcmc_steps=4)
    y, means, _, records = economic_history_moments(
        posterior.unit_coordinates, history, task)
    predictive = posterior.weights @ means
    assert posterior.n_records == len(records) == len(y)
    assert posterior.n_stages > 1
    assert np.isfinite(posterior.log_likelihood).all()
    # The two interim utility readings contain independent meter error and need
    # not agree with one deterministic rate. The much narrower rounded terminal
    # ledger fields should be matched at their reporting resolution.
    terminal = np.array([record.kind == "completed" for record in records])
    assert np.max(np.abs(predictive[terminal] - y[terminal])) < 5e-5
    assert np.max(np.abs(predictive[~terminal] - y[~terminal])) < 0.05
    assert np.unique(np.round(posterior.unit_coordinates, 8), axis=0).shape[0] > 24
