import copy

import numpy as np
import pytest

from slowlab.design import Design
from slowlab.env import SlowLabEnv
from slowlab.tasks import TASKS


def _design(env):
    treatments = {
        "low": {f.name: 0.3 for f in env.task.factors},
        "high": {f.name: 0.7 for f in env.task.factors},
    }
    allocation = {
        "low": [u.id for u in env.facility.units if u.chamber == 0][:2],
        "high": [u.id for u in env.facility.units if u.chamber == 1][:2],
    }
    return Design(treatments, allocation, randomization_seed=17)


def _final_vector(env):
    return np.array([
        [o.value, o.rev_rate, o.cost_rate, o.energy_cost_rate, o.other_cost_rate]
        for o in env.observations()
    ])


def test_segmented_observation_does_not_change_terminal_path():
    one_shot = SlowLabEnv(TASKS["T3"], seed=41)
    segmented = SlowLabEnv(TASKS["T3"], seed=41)
    d = _design(one_shot)
    one_shot.submit_design(copy.deepcopy(d))
    segmented.submit_design(copy.deepcopy(d))

    one_shot.advance()
    for day in (0, 30, 75, 140):
        segmented.advance_to(day)
        segmented.observe(modality="canopy_lai")
        segmented.observe(modality="energy_cost_to_date")
    segmented.advance_to(segmented.task.duration_days)

    np.testing.assert_array_equal(_final_vector(one_shot), _final_vector(segmented))


def test_terminal_result_is_hidden_until_terminal_day():
    env = SlowLabEnv(TASKS["T3"], seed=2)
    env.submit_design(_design(env))
    env.advance_to(env.task.duration_days - 1)
    assert env.observations() == []
    assert env.round == 0
    out = env.advance_to(env.task.duration_days)
    assert out == env.observations()
    assert env.round == 1


def test_running_units_remain_occupied_until_terminal_day():
    env = SlowLabEnv(TASKS["T3"], seed=2)
    design = _design(env)
    env.submit_design(design)
    assert set(design.unit_ids).isdisjoint(env.available_units())
    env.advance_to(80)
    assert set(design.unit_ids).isdisjoint(env.available_units())
    env.advance()
    assert set(design.unit_ids).issubset(env.available_units())


def test_recommendations_can_be_updated_within_a_round_without_changing_treatment():
    env = SlowLabEnv(TASKS["T3"], seed=19)
    # The update API records timing; this test does not need to run the expensive
    # disk-cached oracle used only to score the private regret trace.
    env.truth._oracle = (np.full(env.task.d, 0.5), 0.0)
    env.submit_design(_design(env))
    env.advance_to(40)
    env.interim_recommendation([0.2, 0.8])
    env.advance_to(90)
    env.interim_recommendation([0.3, 0.7])
    updates = env.recommendation_updates()
    assert [(u.round, u.day) for u in updates] == [(0, 40), (0, 90)]
    assert updates[0].point == (0.2, 0.8)
    assert env.observations() == []


def test_repeated_read_returns_cached_measurement():
    env = SlowLabEnv(TASKS["T3"], seed=3)
    env.submit_design(_design(env))
    env.advance_to(60)
    first = env.observe(modality="canopy_lai")
    second = env.observe(modality="canopy_lai")
    assert first == second
    assert all(a is b for a, b in zip(first, second))
    assert len(env.measurements()) == len(first)
    assert all(m.kind == "measured" and m.day == 60 for m in first)


def test_past_measurement_is_immutable_after_future_advance():
    env = SlowLabEnv(TASKS["T3"], seed=5)
    env.submit_design(_design(env))
    env.advance_to(50)
    old = env.observe(modality="harvested_fresh_mass")
    env.advance_to(100)
    new = env.observe(modality="harvested_fresh_mass")
    assert old == [m for m in env.measurements() if m.day == 50]
    assert all(m.day == 100 for m in new)
    assert all(m.absolute_day <= env.clock for m in env.measurements())


def test_day_zero_has_no_accrued_cost_and_setpoint_uses_physical_units():
    env = SlowLabEnv(TASKS["T3"], seed=7)
    env.submit_design(_design(env))
    costs = env.observe(modality="energy_cost_to_date")
    assert all(m.value == 0.0 for m in costs)
    temperatures = env.observe(modality="setpoint:day_temp")
    assert {round(m.value, 6) for m in temperatures} == {22.2, 27.8}
    assert all(m.unit == "degC" for m in temperatures)


def test_observation_api_rejects_time_travel_and_hidden_state_names():
    env = SlowLabEnv(TASKS["T3"], seed=11)
    with pytest.raises(RuntimeError):
        env.observe()
    env.submit_design(_design(env))
    env.advance_to(20)
    with pytest.raises(ValueError):
        env.advance_to(19)
    for hidden in ("profit", "final_profit", "internal_dry_mass", "future_trajectory"):
        with pytest.raises(ValueError):
            env.observe(modality=hidden)


def test_completed_fields_are_algebraically_consistent():
    env = SlowLabEnv(TASKS["T3"], seed=13)
    env.submit_design(_design(env))
    for observation in env.advance():
        assert observation.cost_rate == pytest.approx(
            observation.energy_cost_rate + observation.other_cost_rate)
        assert observation.value == pytest.approx(
            observation.rev_rate - observation.energy_cost_rate - observation.other_cost_rate)
        assert observation.rev_rate == round(
            observation.rev_rate, env.TERMINAL_RATE_DECIMALS)
        assert observation.energy_cost_rate == round(
            observation.energy_cost_rate, env.TERMINAL_RATE_DECIMALS)
        assert observation.other_cost_rate == round(
            observation.other_cost_rate, env.TERMINAL_RATE_DECIMALS)


def test_common_history_is_ordered_copied_and_censored_at_current_time():
    env = SlowLabEnv(TASKS["T3"], seed=23)
    env.submit_design(_design(env))
    assert [e.kind for e in env.history()] == ["submitted"]
    env.advance_to(50)
    env.observe(units=[_design(env).unit_ids[0]], modality="canopy_lai")
    at_50 = env.history()
    assert [e.kind for e in at_50] == ["submitted", "measured"]
    assert [e.seq for e in at_50] == list(range(len(at_50)))
    at_50[-1].payload["value"] = -999
    assert env.history()[-1].payload["value"] != -999

    env.advance()
    assert len(env.history(through_day=50)) == 2
    completed = [e for e in env.history() if e.kind == "completed"]
    assert len(completed) == len(env.observations())
    assert all(e.absolute_day == env.task.duration_days for e in completed)
