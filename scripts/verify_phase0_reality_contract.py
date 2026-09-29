#!/usr/bin/env python3
"""Reproduce the frozen Phase 0 pass/fail conclusion from committed artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = json.loads((ROOT / "configs" / "agc" / "agc2019_holdout_result_v8.json").read_text())
    gate = json.loads((ROOT / "configs" / "agc" / "agc2019_climate_validation_gate_v0.json").read_text())
    reality_path = ROOT / "configs" / "reality_constraints_v2_2.json"
    reality_bytes = reality_path.read_bytes()
    reality = json.loads(reality_bytes)
    expected_hash = (ROOT / "configs" / "reality_constraints_v2_2.sha256").read_text().split()[0]
    actual_hash = hashlib.sha256(reality_bytes).hexdigest()
    assert actual_hash == expected_hash
    pooled = result["result"]["aggregate"]["pooled"]
    limits = gate["minimum_reality_gate"]["maximum_rmse"]
    metrics = {
        "temperature": {
            "rmse": pooled["metrics"]["tAir"]["rmse"],
            "limit": limits["tAir_c"],
        },
        "relative_humidity": {
            "rmse": pooled["metrics"]["rhIn"]["rmse"],
            "limit": limits["rhIn_percentage_points"],
        },
        "co2": {
            "rmse": pooled["metrics"]["co2InPpm"]["rmse"],
            "limit": limits["co2InPpm"],
        },
    }
    for metric in metrics.values():
        metric["pass"] = metric["rmse"] <= metric["limit"]
    decision = "pass" if all(item["pass"] for item in metrics.values()) else "fail"
    assert decision == result["decision"] == reality["phase0_decision"]["greenlight_v8_result"]
    report = {
        "phase0_status": reality["phase0_decision"]["status"],
        "trajectory_gate": decision,
        "reality_contract_sha256": actual_hash,
        "sequences": pooled["sequence_count"],
        "samples": pooled["samples"],
        "metrics": metrics,
        "claim_boundary": reality["claim_boundary"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
