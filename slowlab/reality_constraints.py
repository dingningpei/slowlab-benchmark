"""Load the reality-support manifest and classify values against it.

The manifest (``configs/reality_constraints_v2_2.json``) records where each quantity's
evidence comes from. The audit policy (``configs/reality_audit_policy_v1.json``) says
how a value is judged: every record falls into exactly one category -- within support,
source extrapolation, declared assumption, physical error, undeclared assumption or
unknown constraint. An evidence envelope describes data coverage, not a physical limit,
so extrapolation is reported but never fails the audit or removes a site. Passing does
not turn the simulator into a validated digital twin.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .frozen_paths import frozen_path

ROOT = Path(__file__).resolve().parents[1]


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


def load_audit_policy(path: str | Path, manifest_path: str | Path | None = None) -> dict[str, Any]:
    """Load the audit policy; optionally check it names the exact manifest file."""
    policy = json.loads(Path(path).read_text())
    for key in ("policy_id", "categories", "assumption_rules", "physical_bounds", "analysis_roles"):
        if key not in policy:
            raise ValueError(f"audit policy missing {key}")
    if manifest_path is not None:
        digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
        if digest != policy["applies_to_manifest"]["sha256"]:
            raise ValueError("audit policy was written for a different manifest")
    return policy


def _finite(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _physical_issue(value: Any, constraint: dict[str, Any], policy: dict[str, Any]) -> str | None:
    bounds = policy["physical_bounds"]
    rule = bounds["constraints"].get(constraint["constraint_id"]) or bounds["unit_defaults"].get(constraint["unit"])
    if not _finite(value):
        return "not_a_finite_number"
    if rule is None:
        return None
    if "values" in rule:
        return None if any(float(value) == float(v) for v in rule["values"]) else f"not_in_{rule['basis']}_values"
    if "min" in rule and float(value) < float(rule["min"]):
        return f"below_{rule['basis']}_bound"
    if "max" in rule and float(value) > float(rule["max"]):
        return f"above_{rule['basis']}_bound"
    return None


def _needs_declaration(constraint: dict[str, Any]) -> bool:
    return constraint["evidence_class"] in ("assumed", "unsupported") or not constraint["formal_eligible"]


def _extrapolation(value: float, support: dict[str, Any]) -> dict[str, Any]:
    if "values" in support:
        return {"side": "not_an_observed_value", "observed_values": support["values"]}
    low, high = float(support["min"]), float(support["max"])
    if float(value) < low:
        return {"side": "below", "support": [low, high], "distance": low - float(value)}
    return {"side": "above", "support": [low, high], "distance": float(value) - high}


def classify_record(record: dict[str, Any], index: dict[str, dict], policy: dict[str, Any],
                    root: Path = ROOT) -> tuple[str, dict[str, Any]]:
    """Return (category, detail) for one record; see the audit policy for categories."""
    cid = record.get("constraint_id")
    value = record.get("value")
    roles = policy["analysis_roles"]
    role = roles["aliases"].get(record.get("analysis_role", roles["default"]),
                                record.get("analysis_role", roles["default"]))
    detail: dict[str, Any] = {"constraint_id": cid, "value": value, "analysis_role": role}
    for key in ("site", "context"):
        if key in record:
            detail[key] = record[key]
    if cid not in index:
        return "unknown_constraint", detail
    if role not in roles["allowed"]:
        detail["reason"] = f"unknown analysis role {role!r}"
        return "undeclared_assumption", detail
    constraint = index[cid]
    issue = _physical_issue(value, constraint, policy)
    if issue is not None:
        detail["reason"] = issue
        return "physical_error", detail
    if _needs_declaration(constraint):
        declared = record.get("declared_in")
        detail["declared_in"] = declared
        sensitivity_only = policy["assumption_rules"]["sensitivity_only"]
        if not isinstance(declared, str) or not declared or not frozen_path(root, declared).is_file():
            detail["reason"] = "no existing declaring file"
            return "undeclared_assumption", detail
        if cid in sensitivity_only and role != "sensitivity":
            detail["reason"] = "sensitivity-only quantity used in the main analysis: " + sensitivity_only[cid]
            return "undeclared_assumption", detail
        return "declared_assumption", detail
    if _in_support(value, constraint["support"]):
        return "within_support", detail
    detail.update(_extrapolation(value, constraint["support"]))
    return "source_extrapolation", detail


def audit_reality_support(records: Iterable[dict[str, Any]], manifest: dict[str, Any],
                          policy: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    """Classify every record; fail only on physical errors, undeclared assumptions or unknown ids.

    Nothing is removed: every input record appears once, in input order, under its
    category, and per-site counts show where extrapolation or errors occur.
    """
    index = {item["constraint_id"]: item for item in manifest["constraints"]}
    categories = {name: [] for name in policy["categories"]}
    per_site: dict[str, dict[str, int]] = {}
    checked = 0
    for position, record in enumerate(records):
        checked += 1
        category, detail = classify_record(record, index, policy, root)
        detail["position"] = position
        categories[category].append(detail)
        if "site" in record:
            site = per_site.setdefault(str(record["site"]), {name: 0 for name in policy["categories"]})
            site[category] += 1
    failing = [name for name, spec in policy["categories"].items() if spec["fails"] and categories[name]]
    return {
        "protocol_id": manifest["protocol_id"],
        "policy_id": policy["policy_id"],
        "checked": checked,
        "status": "fail" if failing else "pass",
        "failing_categories": failing,
        "counts": {name: len(items) for name, items in categories.items()},
        "records": categories,
        "per_site": per_site,
        "excluded_records": [],
        "interpretation": (
            "Classification only. Extrapolation beyond an evidence envelope is disclosed, not "
            "removed. Passing establishes neither causal nor counterfactual validity."
        ),
    }
