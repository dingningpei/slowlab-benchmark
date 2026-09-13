"""Regression tests for transcript-to-execution-event recovery."""
from __future__ import annotations

import json

from slowlab import TASKS
from slowlab.eventlog import (
    completed_design_payloads,
    design_arrays_from_events,
    executed_designs,
    reconcile_format_failures,
    recover_transcript,
)


def _turn(reply, *, rejected=None, final=False):
    if final:
        user = "All rounds are finished. No further experiments are possible."
    else:
        user = "ROUNDS REMAINING: 3"
        if rejected:
            user += f"\nYOUR PREVIOUS REPLY WAS REJECTED: {rejected}"
    return {"user": user, "assistant": json.dumps(reply)}


def _valid(temp, density=3.6):
    return {
        "reasoning": "regression fixture",
        "treatments": {"A": {"day_temp": temp, "density": density}},
        "allocation": {"A": ["c0l0", "c0l1"]},
        "randomization_seed": 1,
        "stop": False,
        "current_best": {"day_temp": temp, "density": density},
    }


def _invalid():
    blob = _valid(20.5)
    blob["treatments"]["B"] = {"day_temp": 24.0, "density": 3.6}
    blob["allocation"] = {"A": ["c0l0"], "B": ["c0l1"]}
    return blob


def test_known_optimise_retry_shape_recovers_two_of_six_designs():
    """Mirrors GPT-5.6-luna+tools/Optimise/s4: six design replies,
    four hierarchy rejections, and only two completed experiments.
    """
    rejected = "INFEASIBLE_GRANULARITY"
    transcript = [
        _turn(_valid(28.5)),
        _turn(_invalid()),
        _turn(_valid(20.5), rejected=rejected),
        _turn(_invalid()),
        _turn(_invalid(), rejected=rejected),
        _turn(_invalid(), rejected=rejected),
        _turn({"reasoning": "done",
               "recommendation": {"day_temp": 20.5, "density": 3.6}},
              final=True),
    ]

    result = recover_transcript(TASKS["Optimise"], 4, transcript)

    assert result.errors == []
    assert result.counts["proposed"] == 6
    assert result.counts["rejected"] == 4
    assert result.counts["accepted"] == 2
    assert result.counts["completed"] == 2
    assert result.counts["forfeited"] == 1
    assert len(executed_designs(result)) == 2


def test_transfer_prompt_is_not_misclassified_as_final():
    transcript = [{
        "user": ("Energy prices have now changed. No further experiments are "
                 "possible."),
        "assistant": json.dumps({
            "recommendation": {
                "day_temp": 22.0, "night_temp": 17.0,
                "co2": 500.0, "par": 20.0}}),
    }]
    result = recover_transcript(TASKS["Transfer"], 0, transcript)
    phases = [event.payload.get("phase") for event in result.events]
    assert phases == ["transfer"]


def test_summary_only_parse_failures_are_marked_inferred():
    result = recover_transcript(TASKS["Sanity"], 0, [])

    added = reconcile_format_failures(result, 2)

    assert added == 2
    assert result.counts["parse_failed"] == 2
    assert all(event.payload["inferred"] for event in result.events)
    assert all("raw_reply" not in event.payload for event in result.events)


def test_event_loader_excludes_accepted_design_without_completion(tmp_path):
    result = recover_transcript(TASKS["Sanity"], 0, [_turn(_valid(22.0))])
    document = result.to_dict()
    document["events"] = [
        event for event in document["events"] if event["kind"] != "completed"
    ]
    assert completed_design_payloads(document) == []

    path = tmp_path / "events.json"
    path.write_text(json.dumps(result.to_dict()))
    designs = design_arrays_from_events(path, __import__("slowlab").SlowLabEnv(
        TASKS["Sanity"], seed=0))
    assert len(designs) == 1
    assert designs[0][0].shape == (2, 1)
