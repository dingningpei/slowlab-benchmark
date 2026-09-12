"""End-to-end tests of the LLM harness -- no network, no SDK."""
import json
import numpy as np
import pytest

from slowlab import SlowLabEnv, TASKS
from slowlab.llm import LLMAgent, scripted_completer, _extract_json


def test_json_extraction_survives_fences_and_prose():
    assert _extract_json('here you go:\n```json\n{"a": 1}\n```\nhope that helps')["a"] == 1
    assert _extract_json('{"a": {"b": 2}} trailing text')["a"]["b"] == 2
    with pytest.raises(ValueError):
        _extract_json("no object here")


def test_end_to_end_all_rounds_submitted():
    env = SlowLabEnv(TASKS["T3"], seed=0)
    ag = LLMAgent(complete=scripted_completer())
    r = env.submit_recommendation(ag.run(env), "fake")
    assert r.n_designs == TASKS["T3"].n_rounds
    assert ag.format_failures == 0 and ag.infeasible_submissions == 0
    assert len(r.regret_trace) == TASKS["T3"].n_rounds


def test_final_recommendation_is_asked_after_last_round():
    """The model must be asked once more after the final round's observations are revealed, or that round's data is wasted."""
    env = SlowLabEnv(TASKS["T3"], seed=0)
    ag = LLMAgent(complete=scripted_completer())
    ag.run(env)
    assert len(ag.transcript) == TASKS["T3"].n_rounds + 1
    last = ag.transcript[-1]["user"]
    assert "No further" in last and "final recommendation" in last
    assert "round 3" in last          # the final round's observations really are in the last prompt


def test_malformed_reply_forfeits_the_round_not_the_campaign():
    calls = {"n": 0}

    def broken(messages):
        calls["n"] += 1
        return "I am afraid I cannot do that."

    env = SlowLabEnv(TASKS["T3"], seed=0)
    ag = LLMAgent(complete=broken, max_retries=3)
    ag.run(env)
    # Three retries per round; once exhausted, *only that round* is voided and the
    # campaign continues to the last round, plus three more retries for the final
    # recommendation. This used to break out of the whole episode (one failure
    # ended everything), which was the wrong severity of penalty.
    n_rounds = TASKS["T3"].n_rounds
    assert calls["n"] == 3 * n_rounds + 3
    assert ag.forfeited_rounds == n_rounds
    assert ag.format_failures == 3 * n_rounds + 3
    # No budget was spent: a voided round produces no observations and occupies zero unit-days
    assert env.unit_days_used() == 0 if callable(getattr(env, 'unit_days_used', None)) \
        else len(env.observations()) == 0                  # budget untouched


def test_hierarchy_violation_is_fed_back_not_charged():
    """A chamber-level factor taking two values in one chamber: rejected, the reason fed back, no budget spent."""
    def violator(messages):
        return json.dumps({
            "reasoning": "ignores the chamber constraint on purpose",
            "treatments": {"A": {"day_temp": 20.0, "density": 3.0},
                           "B": {"day_temp": 30.0, "density": 3.0}},
            "allocation": {"A": ["c0l0"], "B": ["c0l1"]},   # one chamber, two temperatures
            "randomization_seed": 1,
            "current_best": {"day_temp": 23.0, "density": 3.0},
        })

    env = SlowLabEnv(TASKS["T3"], seed=0)
    ag = LLMAgent(complete=violator, max_retries=2)
    ag.run(env)
    # Twice per round over n_rounds rounds -- a violation costs no budget and does not end the campaign early
    assert ag.infeasible_submissions == 2 * TASKS["T3"].n_rounds
    # No budget spent: a voided round produces no observations and zero unit-days
    assert env.unit_days_used() == 0 if callable(getattr(env, 'unit_days_used', None)) \
        else len(env.observations()) == 0
    assert any("INFEASIBLE_GRANULARITY" in c for c in env._rejections)
    # The reason was fed back (the last message is the final question; the retry after rejection is the one before it)
    assert any("INFEASIBLE_GRANULARITY" in c["user"] for c in ag.transcript)


def test_prompt_states_constraints_but_gives_no_advice():
    env = SlowLabEnv(TASKS["T3"], seed=0)
    ag = LLMAgent(complete=scripted_completer())
    ag.run(env)
    prompt = ag.transcript[0]["user"]
    for must in ["FACILITY", "CHAMBER", "BUDGET", "day_temp", "c0l0",
                 "choose at most"]:
        assert must in prompt
    # No hints about the capability under test. "randomization_seed" is a field name
    # on Design rather than advice, so it is excluded before checking.
    body = prompt.lower().replace("randomization_seed", "")
    for must_not in ["replicat", "randomis", "randomiz", "block", "split-plot",
                     "pure error", "confound"]:
        assert must_not not in body


def test_agent_may_stop_early():
    """Abstaining is a legal action: submit no design, occupy no units, finish on
    current_best.

    The results show that on a short budget not experimenting may be the right
    move. If the agent had no such action we could observe the phenomenon but not
    measure whether it finds it.
    """
    import json as _json
    from slowlab import SlowLabEnv, TASKS
    import slowlab.llm as L

    class _Fake:
        def __init__(self):
            self.n = 0
            self.base = L.scripted_completer()

        def __call__(self, msgs):
            self.n += 1
            if self.n == 2 and "No further" not in msgs[-1]["content"]:
                return _json.dumps({"reasoning": "enough", "stop": True,
                                    "current_best": {"day_temp": 23.0, "density": 3.0}})
            return self.base(msgs)

    env = SlowLabEnv(TASKS["Optimise"], seed=0)
    ag = L.LLMAgent(complete=_Fake())
    r = env.submit_recommendation(ag.run(env), "fake")
    assert ag.stopped_early_at == 1
    assert r.n_designs == 1                      # nothing was submitted in round two
    assert r.unit_days_used < TASKS["Optimise"].budget_units * 120


def test_stop_option_is_offered_in_the_prompt():
    from slowlab import SlowLabEnv, TASKS
    from slowlab.llm import _schema_text
    txt = _schema_text(SlowLabEnv(TASKS["Optimise"], seed=0))
    assert chr(34)+"stop"+chr(34) in txt and "Stopping is a legitimate" in txt


def test_tool_text_stays_inside_factor_ranges():
    """The tool's design must land inside the factor ranges. We once scaled +/-1 as
    0/1, which produced 14.5 C (range 18-32) and 0.5 plants/m^2 (range 2-8)."""
    import re
    from slowlab import SlowLabEnv, TASKS
    from slowlab.llm import tool_text
    for cfg in ("Optimise", "Screen"):
        env = SlowLabEnv(TASKS[cfg], seed=0)
        txt = tool_text(env)
        lo = {f.name: f.low for f in env.task.factors}
        hi = {f.name: f.high for f in env.task.factors}
        for name, val in re.findall(r"(\w+)=([\d.]+)", txt):
            if name in lo:
                assert lo[name] - 1e-6 <= float(val) <= hi[name] + 1e-6, (cfg, name, val)


def test_tools_flag_changes_the_prompt():
    from slowlab import SlowLabEnv, TASKS
    from slowlab.llm import LLMAgent, scripted_completer
    env = SlowLabEnv(TASKS["Optimise"], seed=0)
    a0 = LLMAgent(complete=scripted_completer(), tools=False)
    a1 = LLMAgent(complete=scripted_completer(), tools=True)
    a0.run(env)
    env2 = SlowLabEnv(TASKS["Optimise"], seed=0)
    a1.run(env2)
    assert "TOOL 1" not in a0.transcript[0]["user"]
    assert "TOOL 1" in a1.transcript[0]["user"]
