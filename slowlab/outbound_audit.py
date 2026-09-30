"""Audited, firewalled model calls.

``AuditedCompleter`` wraps ``complete(messages) -> str``. Before every call it
runs the simulation-blinding firewall on every harness-generated message; a block writes an audit record and
raises ``FirewallBlock``, an infrastructure failure that must abort the
campaign and must never be retried into a result. Every permitted call
appends one JSON line holding the complete outbound messages, their SHA-256,
the reply and the provider's call record. The audit log holds only data that
was sent to or received from the provider.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .agent_protocol import canonical, digest
from .prompt_firewall import audit_outbound_messages

AUDIT_PROTOCOL = 'outbound-audit-v1'


class FirewallBlock(RuntimeError):
    """An outbound payload failed the blinding firewall; abort, record, never retry."""


class AuditedCompleter:
    def __init__(self, complete: Callable, policy: dict, audit_path: Path, *, label: str = ''):
        self._complete = complete
        self._policy = policy
        self._path = Path(audit_path)
        self._label = label
        self.calls = 0

    def _write(self, record: dict) -> None:
        with self._path.open('a', encoding='utf-8') as stream:
            stream.write(canonical(record) + '\n')

    def __call__(self, messages: list) -> str:
        self.calls += 1
        payload = canonical(messages)
        record = {'protocol': AUDIT_PROTOCOL, 'label': self._label, 'seq': self.calls,
                  'messages_sha256': digest(payload), 'messages': messages}
        # The firewall checks what the harness generates. A model's own earlier
        # replies, echoed back as assistant turns, are its words, not a leak.
        report = audit_outbound_messages([m for m in messages if m.get('role') != 'assistant'], self._policy)
        if report['status'] != 'pass':
            self._write({**record, 'firewall': 'blocked', 'findings': report['findings']})
            raise FirewallBlock('outbound payload blocked by the blinding firewall')
        reply = self._complete(messages)
        if not isinstance(reply, str):
            raise TypeError('completer must return text')
        records = getattr(self._complete, 'call_records', None)
        self._write({**record, 'firewall': 'pass', 'reply': reply, 'reply_sha256': digest(reply),
                     'provider_record': records[-1] if records else None})
        return reply


def read_audit(path: Path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
