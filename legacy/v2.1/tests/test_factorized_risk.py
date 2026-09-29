import copy
import numpy as np

from slowlab.design import Design
from slowlab.env import SlowLabEnv
from slowlab.factorized_history import factorized_history_posterior
from slowlab.factorized_risk import decompose_factorized_history_risk
from slowlab.tasks import TASKS


def test_heldout_factorized_risk_is_nonnegative_by_action_set_construction():
    task = copy.deepcopy(TASKS["Sanity"])
    task.cycle_days = 20
    env = SlowLabEnv(task, seed=3)
    env.submit_design(Design(
        {"a": {"day_temp": .4}}, {"a": ["c0l0"]}, randomization_seed=2))
    env.advance()
    kwargs = dict(
        n_crop_particles=24, n_crop_product=6,
        n_economic_particles=24, n_economic_product=6,
        n_mc_plant=4, mcmc_steps=2)
    search = factorized_history_posterior(
        env.history(), task, crop_seed=1, economic_seed=1, **kwargs)
    evaluation = factorized_history_posterior(
        env.history(), task, crop_seed=2, economic_seed=2, **kwargs)
    result = decompose_factorized_history_risk(
        search, evaluation, np.array([.9]),
        action_candidates=np.linspace(0, 1, 21)[:, None],
        best_candidates=np.linspace(.0125, .9875, 40)[:, None])
    assert result.posterior_excess_risk >= 0
    assert result.posterior_selection_gap >= 0
    np.testing.assert_allclose(
        result.agent_posterior_risk,
        result.bayes_risk + result.posterior_excess_risk)
    assert "finite_best_lower_bound" in result.failure_flags
