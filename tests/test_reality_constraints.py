import hashlib
import json
from pathlib import Path

from slowlab.reality_constraints import audit_reality_support, load_reality_constraints


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "reality_constraints_v2_2.json"


def test_reality_manifest_is_structurally_valid_and_has_all_evidence_classes():
    manifest = load_reality_constraints(MANIFEST)
    assert manifest["status"] == "frozen_phase0_evidence_roles"
    assert {item["evidence_class"] for item in manifest["constraints"]} == {
        "observed", "literature_constrained", "assumed", "unsupported"
    }
    assert manifest["phase0_decision"]["greenlight_v8_result"] == "fail"


def test_formal_audit_fails_closed_on_range_ineligible_and_unknown_values():
    manifest = load_reality_constraints(MANIFEST)
    report = audit_reality_support([
        {"constraint_id": "agc2023.indoor_temperature", "value": 21.0},
        {"constraint_id": "agc2023.indoor_temperature", "value": 50.0},
        {"constraint_id": "unsupported.fogging_realised_flow", "value": 10.0},
        {"constraint_id": "does.not.exist", "value": 1.0},
    ], manifest)
    assert report["status"] == "fail"
    assert [item["issue"] for item in report["violations"]] == [
        "out_of_support", "not_formal_eligible", "unknown_constraint"
    ]


def test_predeclared_sensitivity_is_visible_but_does_not_fail_formal_audit():
    manifest = load_reality_constraints(MANIFEST)
    report = audit_reality_support([{
        "constraint_id": "unsupported.fogging_realised_flow",
        "value": 10.0,
        "analysis_role": "sensitivity",
        "justification": "Predeclared capacity-only stress test",
    }], manifest)
    assert report["status"] == "pass"
    assert report["warnings"][0]["issue"] == "not_formal_eligible"


def test_phase0_numbers_match_frozen_holdout_artifact():
    manifest = json.loads(MANIFEST.read_text())
    result = json.loads((ROOT / "configs" / "agc2019_holdout_result_v8.json").read_text())
    pooled = result["result"]["aggregate"]["pooled"]
    frozen = manifest["phase0_decision"]["pooled_holdout"]
    assert frozen["hourly_samples"] == pooled["samples"]
    assert frozen["temperature_rmse_c"] == pooled["metrics"]["tAir"]["rmse"]
    assert frozen["rh_rmse_percentage_points"] == pooled["metrics"]["rhIn"]["rmse"]
    assert frozen["co2_rmse_ppm"] == pooled["metrics"]["co2InPpm"]["rmse"]


def test_reality_contract_matches_frozen_hash():
    expected = (ROOT / "configs" / "reality_constraints_v2_2.sha256").read_text().split()[0]
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == expected
