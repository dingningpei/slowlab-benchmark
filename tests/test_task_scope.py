from slowlab import (
    CORE_TASK_KEYS,
    CORE_TASK_NAMES,
    CORE_TASKS,
    EXPERIMENTAL_TASK_KEYS,
    EXPERIMENTAL_TASK_NAMES,
    EXPERIMENTAL_TASKS,
    TASKS,
)


def test_version_2_1_core_and_experimental_task_scope():
    assert CORE_TASK_KEYS == ("T1", "T3", "T4")
    assert CORE_TASK_NAMES == ("Sanity", "Optimise", "Transfer")
    assert tuple(CORE_TASKS) == CORE_TASK_NAMES
    assert EXPERIMENTAL_TASK_KEYS == ("T2",)
    assert EXPERIMENTAL_TASK_NAMES == ("Screen",)
    assert tuple(EXPERIMENTAL_TASKS) == EXPERIMENTAL_TASK_NAMES
    assert set(CORE_TASKS).isdisjoint(EXPERIMENTAL_TASKS)
    for name, task in CORE_TASKS.items():
        assert task is TASKS[name]
    assert EXPERIMENTAL_TASKS["Screen"] is TASKS["Screen"]
