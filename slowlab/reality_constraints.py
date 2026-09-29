"""Load and enforce the Version 2.2 reality-support contract.

The contract separates an observed envelope from causal validation.  Passing this
audit only means that a value stays inside a declared support boundary and that
the boundary has an auditable source.  It does not turn the simulator into a
validated digital twin.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable


EVIDENCE_CLASSES = {"observed", "literature_constrained", "assumed", "unsupported"}


def load_reality_constraints(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a reality-constraint manifest."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text())
    required = {"protocol_id", "status", "claim_boundary", "evidence_classes", "constraints"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"reality manifest missing keys: {sorted(missing)}")

    ids: set[str] = set()
    for item in payload["constraints"]:
        needed = {
            "constraint_id", "quantity", "evidence_class", "unit", "support",
            "source", "claim_use", "formal_eligible",
        }
        absent = needed.difference(item)
        if absent:
            raise ValueError(
                f"constraint {item.get('constraint_id', '<unknown>')} missing {sorted(absent)}"
            )
        cid = str(item["constraint_id"])
        if cid in ids:
            raise ValueError(f"duplicate constraint_id: {cid}")
        ids.add(cid)
        if item["evidence_class"] not in EVIDENCE_CLASSES:
            raise ValueError(f"unknown evidence_class for {cid}: {item['evidence_class']}")
        support = item["support"]
        if item["evidence_class"] == "unsupported":
            if support is not None or item["formal_eligible"]:
                raise ValueError(f"unsupported constraint {cid} cannot be formally eligible")
        elif support is None:
            raise ValueError(f"supported constraint {cid} requires a support definition")
        elif not ({"min", "max"} <= set(support) or "values" in support):
            raise ValueError(f"constraint {cid} support needs min/max or values")
    return payload


def _in_support(value: Any, support: dict[str, Any]) -> bool:
    if "values" in support:
        return any(
            (isinstance(value, (int, float)) and isinstance(candidate, (int, float))
             and math.isclose(float(value), float(candidate), rel_tol=0.0, abs_tol=1e-12))
            or value == candidate
            for candidate in support["values"]
        )
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    return float(support["min"]) <= float(value) <= float(support["max"])


def audit_reality_support(
    records: Iterable[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Audit formal or sensitivity records against declared reality support.

    Each record contains ``constraint_id`` and ``value``.  ``analysis_role``
    defaults to ``formal``.  An out-of-support or ineligible record is allowed
    only when ``analysis_role`` is ``sensitivity`` and ``justification`` is a
    nonempty string; it is then returned as a warning rather than disappearing.
    """
    index = {item["constraint_id"]: item for item in manifest["constraints"]}
    violations: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    checked = 0

    for position, record in enumerate(records):
        checked += 1
        cid = record.get("constraint_id")
        value = record.get("value")
        role = record.get("analysis_role", "formal")
        justification = record.get("justification", "")
        issue: str | None = None

        if cid not in index:
            issue = "unknown_constraint"
        else:
            constraint = index[cid]
            if not constraint["formal_eligible"]:
                issue = "not_formal_eligible"
            elif not _in_support(value, constraint["support"]):
                issue = "out_of_support"

        detail = {
            "position": position,
            "constraint_id": cid,
            "value": value,
            "issue": issue,
        }
        if issue is None:
            continue
        if role == "sensitivity" and isinstance(justification, str) and justification.strip():
            detail["justification"] = justification.strip()
            warnings.append(detail)
        else:
            violations.append(detail)

    return {
        "protocol_id": manifest["protocol_id"],
        "checked": checked,
        "status": "pass" if not violations else "fail",
        "violations": violations,
        "warnings": warnings,
        "interpretation": (
            "Passing establishes declared range support only; it does not establish "
            "causal or counterfactual validity."
        ),
    }
