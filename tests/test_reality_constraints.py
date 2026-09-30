import hashlib
import json
from pathlib import Path

import pytest

from slowlab.reality_constraints import audit_reality_support, load_audit_policy, load_reality_constraints


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "reality_constraints_v2_2.json"
POLICY_PATH = ROOT / "configs" / "reality_audit_policy_v1.json"
POLICY = load_audit_policy(POLICY_PATH, MANIFEST)


def test_reality_manifest_is_structurally_valid_and_has_all_evidence_classes():
    manifest = load_reality_constraints(MANIFEST)
    assert manifest["status"] == "frozen_phase0_evidence_roles"
    assert {item["evidence_class"] for item in manifest["constraints"]} == {
        "observed", "literature_constrained", "assumed", "unsupported"
    }
    assert manifest["phase0_decision"]["greenlight_v8_result"] == "fail"


def audit(records):
    return audit_reality_support(records, load_reality_constraints(MANIFEST), POLICY)


def categories(report):
    return [(item["position"], name) for name, items in report["records"].items() for item in items]


def test_policy_is_bound_to_the_exact_manifest():
    assert POLICY["applies_to_manifest"]["sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="different manifest"):
        load_audit_policy(POLICY_PATH, ROOT / "configs" / "task_contract_v4.json")


def test_extrapolation_is_reported_but_never_fails_or_removes_a_site():
    report = audit([
        {"constraint_id": "agc2023.indoor_temperature", "value": 21.0, "site": "s1"},
        {"constraint_id": "agc2023.indoor_temperature", "value": 50.0, "site": "s1"},
        {"constraint_id": "agc2023.indoor_co2", "value": 1150.0, "site": "s2"},
        {"constraint_id": "agc2023.sampling_interval", "value": 600, "site": "s2"},
    ])
    assert report["status"] == "pass" and report["excluded_records"] == []
    assert report["counts"]["within_support"] == 1 and report["counts"]["source_extrapolation"] == 3
    high = report["records"]["source_extrapolation"][0]
    assert high["side"] == "above" and high["distance"] == pytest.approx(50.0 - 35.1999987125314)
    assert report["records"]["source_extrapolation"][2]["side"] == "not_an_observed_value"
    assert report["per_site"]["s1"]["source_extrapolation"] == 1
    assert report["per_site"]["s2"]["source_extrapolation"] == 2
    assert sorted(position for position, _ in categories(report)) == [0, 1, 2, 3]


@pytest.mark.parametrize("cid,value,reason", [
    ("agc2023.indoor_rh", 101.0, "above_law_bound"),
    ("agc2023.indoor_co2", -5.0, "below_law_bound"),
    ("agc2023.indoor_temperature", 75.0, "above_model_domain_bound"),
    ("agc2024.heating_capacity", 130.0, "above_capacity_bound"),
    ("agc2024.co2_capacity", 7.6, "above_capacity_bound"),
    ("agc2023.lamp_state", 0.5, "not_in_law_values"),
    ("agc2023.indoor_temperature", float("nan"), "not_a_finite_number"),
    ("agc2023.indoor_temperature", True, "not_a_finite_number"),
])
def test_physical_errors_fail(cid, value, reason):
    report = audit([{"constraint_id": cid, "value": value}])
    assert report["status"] == "fail" and report["failing_categories"] == ["physical_error"]
    assert report["records"]["physical_error"][0]["reason"] == reason


def test_physical_error_beats_a_declaration():
    report = audit([{"constraint_id": "unsupported.sensor_error_calibration", "value": float("inf"),
                     "declared_in": "configs/sensor_noise_v0.json"}])
    assert report["failing_categories"] == ["physical_error"]


def test_assumptions_need_an_existing_declaring_file():
    declared = audit([
        {"constraint_id": "unsupported.sensor_error_calibration", "value": 0.2,
         "declared_in": "configs/sensor_noise_v0.json"},
        {"constraint_id": "unsupported.heating_water_flow", "value": 1.2,
         "declared_in": "configs/greenlight_model_family.json"},
        # an older flat path recorded in a frozen artifact still resolves
        {"constraint_id": "benchmark.cycle_length", "value": 180.0,
         "declared_in": "configs/v22_task_contract_v3.json"},
    ])
    assert declared["status"] == "pass" and declared["counts"]["declared_assumption"] == 3
    missing = audit([
        {"constraint_id": "unsupported.sensor_error_calibration", "value": 0.2},
        {"constraint_id": "benchmark.cycle_length", "value": 180.0, "declared_in": "configs/nope.json"},
    ])
    assert missing["failing_categories"] == ["undeclared_assumption"]
    assert missing["counts"]["undeclared_assumption"] == 2


def test_sensitivity_only_quantities_fail_in_the_main_analysis():
    main = audit([{"constraint_id": "unsupported.executor_delay", "value": 60.0,
                   "declared_in": "configs/task_contract_v4.json"}])
    assert main["failing_categories"] == ["undeclared_assumption"]
    assert "sensitivity" in main["records"]["undeclared_assumption"][0]["reason"]
    sens = audit([{"constraint_id": "unsupported.executor_delay", "value": 60.0,
                   "declared_in": "configs/task_contract_v4.json", "analysis_role": "sensitivity"}])
    assert sens["status"] == "pass" and sens["counts"]["declared_assumption"] == 1


def test_unknown_constraints_and_roles_fail_and_formal_is_an_alias_of_main():
    report = audit([
        {"constraint_id": "does.not.exist", "value": 1.0},
        {"constraint_id": "agc2023.indoor_temperature", "value": 21.0, "analysis_role": "exploratory"},
        {"constraint_id": "agc2023.indoor_temperature", "value": 21.0, "analysis_role": "formal"},
    ])
    assert report["counts"]["unknown_constraint"] == 1
    assert report["counts"]["undeclared_assumption"] == 1
    assert report["records"]["within_support"][0]["analysis_role"] == "main"
    assert set(report["failing_categories"]) == {"unknown_constraint", "undeclared_assumption"}


def test_phase0_numbers_match_frozen_holdout_artifact():
    manifest = json.loads(MANIFEST.read_text())
    result = json.loads((ROOT / "configs" / "agc" / "agc2019_holdout_result_v8.json").read_text())
    pooled = result["result"]["aggregate"]["pooled"]
    frozen = manifest["phase0_decision"]["pooled_holdout"]
    assert frozen["hourly_samples"] == pooled["samples"]
    assert frozen["temperature_rmse_c"] == pooled["metrics"]["tAir"]["rmse"]
    assert frozen["rh_rmse_percentage_points"] == pooled["metrics"]["rhIn"]["rmse"]
    assert frozen["co2_rmse_ppm"] == pooled["metrics"]["co2InPpm"]["rmse"]


def test_reality_contract_matches_frozen_hash():
    expected = (ROOT / "configs" / "reality_constraints_v2_2.sha256").read_text().split()[0]
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == expected
