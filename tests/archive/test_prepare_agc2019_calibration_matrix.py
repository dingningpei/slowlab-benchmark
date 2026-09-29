import pytest

from scripts.agc.prepare_agc2019_calibration_matrix import consecutive_sequences, selected_identities


def coverage():
    return {
        "summary": {"A": {"input_eligible_dates": ["2020-01-01"]},
                    "B": {"input_eligible_dates": ["2020-01-01", "2020-01-02"]}},
        "daily": {"A": {"2020-01-01": {"input_eligible": True}},
                  "B": {"2020-01-01": {"input_eligible": True},
                        "2020-01-02": {"input_eligible": True}}},
    }


def test_selection_is_exactly_frozen_input_eligibility():
    result = selected_identities(coverage())
    assert [(c, d.isoformat()) for c, d in result] == [
        ("A", "2020-01-01"), ("B", "2020-01-01"), ("B", "2020-01-02")]


def test_selection_rejects_summary_daily_disagreement():
    value = coverage(); value["daily"]["B"]["2020-01-02"]["input_eligible"] = False
    with pytest.raises(ValueError, match="coverage summary mismatch"):
        selected_identities(value)


def test_consecutive_days_form_one_state_propagation_sequence():
    records = [
        {"compartment": "A", "day": "2020-01-01"},
        {"compartment": "A", "day": "2020-01-02"},
        {"compartment": "A", "day": "2020-01-04"},
        {"compartment": "B", "day": "2020-01-02"},
    ]
    result = consecutive_sequences(records)
    assert [(x["compartment"], x["day_count"]) for x in result] == [("A", 2), ("A", 1), ("B", 1)]
    assert result[0]["sequence_id"] == "A:2020-01-01:2d"
