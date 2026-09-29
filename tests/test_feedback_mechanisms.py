import copy

import numpy as np

from slowlab import SlowLabEnv, TASKS
from slowlab.strong_baselines import (ConstraintAwareBatchBOAgent,
                                      ScheduledObservationBatchBOAgent)


def test_shared_control_counterfactual_changes_factor_granularity_only():
    task = copy.deepcopy(TASKS["T3"])
    original = [factor.control_level for factor in task.factors]
    task.shared_control = False
    assert original == ["chamber", "loop"]
    assert [factor.control_level for factor in task.factors] == ["loop", "loop"]
    assert TASKS["T3"].shared_control is True


def test_scheduled_observation_preserves_terminal_policy_path():
    task = copy.deepcopy(TASKS["T1"])
    terminal = SlowLabEnv(task, seed=91)
    observed = SlowLabEnv(task, seed=91)
    a = ConstraintAwareBatchBOAgent(fixed_replicates=1)
    b = ScheduledObservationBatchBOAgent(active=True, fixed_replicates=1)
    xa, xb = a.run(terminal), b.run(observed)
    assert np.allclose(xa, xb)
    assert len(b.visit_records) == 2 * task.n_rounds
    assert all(record["cost"] == 0 for record in b.visit_records)
