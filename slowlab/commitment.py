"""Salted commitments to private seeds (commit before execution, reveal after lock).

``make_commitment`` returns a public record (safe to commit to Git) and a
private opening (salt plus the secret; keep under the gitignored
``output/private/`` until the formal matrix is locked). Anyone holding both
can check them with ``verify``. The salt is 32 random bytes, so the public
digest reveals nothing about small seed spaces.
"""
from __future__ import annotations

import hashlib
import json
import secrets as _secrets

SCHEME = 'sha256(domain \\0 label \\0 salt \\0 canonical-json(secret))'
DOMAIN = b'slowlab-seed-commitment-v1'


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def _digest(label: str, salt: bytes, secret) -> str:
    return hashlib.sha256(b'\0'.join((DOMAIN, label.encode(), salt, canonical(secret)))).hexdigest()


def make_commitment(secret, *, label: str, salt: bytes | None = None) -> tuple[dict, dict]:
    if salt is None:
        salt = _secrets.token_bytes(32)
    if len(salt) < 16:
        raise ValueError('salt must be at least 16 bytes')
    public = {'scheme': SCHEME, 'domain': DOMAIN.decode(), 'label': label, 'commitment': _digest(label, salt, secret)}
    opening = {'label': label, 'salt_hex': salt.hex(), 'secret': secret}
    return public, opening


def verify(public: dict, opening: dict) -> bool:
    if public.get('scheme') != SCHEME or public.get('label') != opening.get('label'):
        return False
    return _secrets.compare_digest(public['commitment'],
                                   _digest(opening['label'], bytes.fromhex(opening['salt_hex']), opening['secret']))
