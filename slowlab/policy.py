"""Immutable, contract-validated management policy.

Agents and tools exchange policies as JSON objects. The executor converts each
accepted payload once into a ``Policy``: validated against the contract's
declared fields, bounds and constraints, frozen for as long as the crop runs
(contract ``policy.immutable_while_running``), and identified by a canonical
digest. ``Policy`` is a read-only mapping, so controller code that reads
``policy['day_temperature_c']`` works unchanged. Numeric values are kept exactly
as submitted (no int/float coercion) so records and traces stay byte-stable.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from .task_contract import validate_policy


class Policy(Mapping):
    __slots__ = ('_contract_id', '_items')

    def __init__(self, contract_id: str, items: tuple):
        object.__setattr__(self, '_contract_id', contract_id)
        object.__setattr__(self, '_items', items)

    def __setattr__(self, name, value):
        raise AttributeError('Policy is immutable')

    @classmethod
    def from_payload(cls, contract: dict, payload) -> 'Policy':
        """Validate a JSON policy payload (or an existing Policy) against ``contract``."""
        if isinstance(payload, Policy):
            payload.require_contract(contract)
            return payload
        validate_policy(contract, payload)
        order = tuple(contract['policy']['fields'])
        return cls(contract['contract_id'], tuple((name, payload[name]) for name in order))

    @property
    def contract_id(self) -> str:
        return self._contract_id

    def require_contract(self, contract: dict) -> None:
        if self._contract_id != contract['contract_id']:
            raise ValueError('policy was validated against a different contract')

    def as_dict(self) -> dict:
        """A fresh JSON-ready copy in the contract's field order."""
        return dict(self._items)

    def digest(self) -> str:
        canonical = json.dumps({'contract_id': self._contract_id, 'policy': self.as_dict()},
                               sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def __getitem__(self, name):
        for key, value in self._items:
            if key == name:
                return value
        raise KeyError(name)

    def __iter__(self):
        return (key for key, _ in self._items)

    def __len__(self):
        return len(self._items)

    def __eq__(self, other):
        if isinstance(other, Policy):
            return self._contract_id == other._contract_id and self._items == other._items
        return NotImplemented

    def __hash__(self):
        return hash((self._contract_id, self._items))

    def __repr__(self):
        return f'Policy({self._contract_id!r}, {self.as_dict()!r})'
