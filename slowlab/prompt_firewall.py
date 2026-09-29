"""Fail-closed audit for information sent to a Version 2.2 model endpoint."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable


def load_blinding_policy(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    required = {
        "protocol_id", "status", "threat_model", "allowed_model_information",
        "forbidden_model_information", "forbidden_patterns", "site_generation_rules",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"blinding policy missing keys: {sorted(missing)}")
    for pattern in payload["forbidden_patterns"]:
        re.compile(pattern, re.IGNORECASE)
    return payload


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def audit_outbound_messages(messages: list[dict[str, Any]], policy: dict[str, Any]) -> dict:
    """Return every forbidden identity or hidden-state marker in outbound messages."""
    findings = []
    for index, message in enumerate(messages):
        extra_keys = sorted(set(message).difference({"role", "content"}))
        if extra_keys:
            findings.append({
                "message": index, "issue": "unexpected_message_keys", "keys": extra_keys,
            })
        text = _content_text(message.get("content", ""))
        for pattern in policy["forbidden_patterns"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                findings.append({
                    "message": index,
                    "issue": "forbidden_pattern",
                    "pattern": pattern,
                    "match": match.group(0)[:120],
                })
    return {
        "protocol_id": policy["protocol_id"],
        "status": "pass" if not findings else "fail",
        "messages_checked": len(messages),
        "findings": findings,
    }


def assert_outbound_messages_safe(
    messages: list[dict[str, Any]], policy: dict[str, Any]
) -> None:
    report = audit_outbound_messages(messages, policy)
    if report["status"] != "pass":
        raise RuntimeError("outbound prompt blocked by simulation-blinding firewall: " +
                           json.dumps(report["findings"], ensure_ascii=False))


class BlindedCompleter:
    """Validate every message list immediately before calling a provider."""

    def __init__(self, complete: Callable, policy: dict[str, Any]):
        self._complete = complete
        self.policy = policy
        self.audit_reports: list[dict[str, Any]] = []

    def __call__(self, messages):
        report = audit_outbound_messages(messages, self.policy)
        self.audit_reports.append(report)
        if report["status"] != "pass":
            raise RuntimeError("outbound prompt blocked by simulation-blinding firewall: " +
                               json.dumps(report["findings"], ensure_ascii=False))
        return self._complete(messages)

    def __getattr__(self, name):
        return getattr(self._complete, name)


def blinded_completer(complete: Callable, policy_path: str | Path) -> BlindedCompleter:
    return BlindedCompleter(complete, load_blinding_policy(policy_path))
