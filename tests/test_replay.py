import numpy as np

from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent
from slowlab.replay import (ComponentGPReader, OrdinaryGPReader,
                            PrivilegedSiteReader)


def test_readers_replay_one_fixed_history_without_new_designs():
    env = SlowLabEnv(TASKS["T1"], seed=5)
    make_agent("split_plot_doe").run(env)
    before = [(event.seq, event.kind, event.payload) for event in env.history()]
    points = np.linspace(0, 1, 101)[:, None]
    for reader in (OrdinaryGPReader(), ComponentGPReader(), PrivilegedSiteReader()):
        point = reader.recommend(env, candidates=points)
        assert point.shape == (1,)
    after = [(event.seq, event.kind, event.payload) for event in env.history()]
    assert after == before
    assert len(env._designs) == TASKS["T1"].n_rounds


def test_privileged_reader_is_explicit():
    assert PrivilegedSiteReader.privileged is True
    assert OrdinaryGPReader.privileged is False
