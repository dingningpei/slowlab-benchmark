from dataclasses import replace

import numpy as np

from slowlab import SlowLabEnv, TASKS
from slowlab.registry import describe, make_agent
from slowlab.strong_baselines import (BlockAwareGP, ConstraintAwareBatchBOAgent,
                                      completed_history_arrays)


def test_adaptive_replication_switches_with_noise_regime():
    agent = ConstraintAwareBatchBOAgent()
    low = replace(TASKS["T1"], plant_cv=0.03, tau_loop=0.01, target_mde=0.5)
    high = replace(TASKS["T1"], plant_cv=0.8, tau_loop=0.2, target_mde=0.25)
    assert agent.replication_count(low) == 1
    assert agent.replication_count(high) == 4


def test_constraint_aware_batch_bo_uses_full_budget_and_valid_designs():
    env = SlowLabEnv(TASKS["T1"], seed=3)
    agent = ConstraintAwareBatchBOAgent()
    point = agent.run(env)
    result = env.submit_recommendation(point, agent.name)
    assert result.n_designs == TASKS["T1"].n_rounds
    assert result.parallel_utilisation == 1.0
    assert result.rejections == []
    assert len(agent.replication_history) == TASKS["T1"].n_rounds


def test_block_aware_reader_tracks_chamber_and_batch_covariance():
    X = np.array([[0.1], [0.1], [0.9], [0.9]])
    y = np.array([0.0, 0.1, 1.0, 1.1])
    chambers = np.array([0, 0, 1, 1])
    batches = np.array([0, 1, 0, 1])
    noise = np.full(4, 0.05)
    model = BlockAwareGP(chamber_scale=0.3, batch_scale=0.2).fit(
        X, y, chambers, batches, noise)
    mean, variance = model.predict(np.array([[0.2], [0.8]]))
    assert mean.shape == variance.shape == (2,)
    assert np.all(variance > 0)


def test_history_adapter_uses_completed_public_events():
    env = SlowLabEnv(TASKS["T1"], seed=2)
    make_agent("constraint_aware_batch_bo").run(env)
    X, y, chambers, batches, noise = completed_history_arrays(env)
    assert len(X) == len(y) == len(chambers) == len(batches) == len(noise)
    assert len(y) == len([e for e in env.history() if e.kind == "completed"])


def test_phase3_registry_exposes_all_initial_baselines():
    for name in ("split_plot_doe", "constraint_aware_batch_bo",
                 "component_reconstruction", "prior_optimal_fixed", "site_oracle"):
        assert make_agent(name) is not None


def test_privileged_diagnostics_are_machine_readable():
    deployable = describe("constraint_aware_batch_bo")
    assert deployable["protocol_version"] == "phase3-baseline-0.1"
    assert deployable["privileged"] is False
    assert describe("prior_optimal_fixed")["privileged"] is True
    assert describe("site_oracle")["privileged"] is True
