"""Locate the pinned GreenLight code and model definitions.

Two ways to supply GreenLight, both checked against the pinned file hashes in
``configs/greenlight_model_family.json`` (repeated in the task contract):

* ``source``: a checkout directory at the pinned commit, as used on the remote
  host (``--source /path/to/GreenLight``). It is put first on ``sys.path``.
* nothing: the installed package, from ``pip install -e '.[greenlight]'``, whose
  model definitions ship as package data.

The engine that is imported and the definition files that are read must come
from the same tree; mixing a checkout's definitions with a different installed
engine is refused.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = Path('greenlight/models/katzin_2021/definition')
INSTALL_HINT = ("install the pinned engine with  pip install -e '.[greenlight]'  "
                "or pass --source <GreenLight checkout at the pinned commit>")


def pinned_files(contract: dict | None = None) -> dict[str, str]:
    if contract is not None:
        return dict(contract['model']['files'])
    family = json.loads((ROOT / 'configs/greenlight_model_family.json').read_text())
    return dict(family['files'])


def verify_definitions(source: Path, files: dict[str, str]) -> None:
    definitions = Path(source) / DEFINITIONS
    if not definitions.is_dir():
        raise FileNotFoundError(f'no GreenLight model definitions under {definitions}; ' + INSTALL_HINT)
    for name, expected in files.items():
        path = definitions / name
        if not path.is_file():
            raise FileNotFoundError(f'missing pinned GreenLight definition {name}; ' + INSTALL_HINT)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'GreenLight definition {name} does not match the pinned commit')


def _engine_root(module) -> Path:
    return Path(module.__file__).resolve().parent.parent


def resolve_greenlight_source(source: Path | str | None = None, contract: dict | None = None) -> Path:
    """Return the tree holding both the engine and the pinned definitions."""
    files = pinned_files(contract)
    if source is not None:
        root = Path(source).resolve()
        verify_definitions(root, files)
        loaded = sys.modules.get('greenlight')
        if loaded is not None and _engine_root(loaded) != root:
            raise RuntimeError(f'greenlight already imported from {_engine_root(loaded)}, '
                               f'not from the requested source {root}')
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        engine = importlib.import_module('greenlight')
    else:
        try:
            engine = importlib.import_module('greenlight')
        except ImportError as exc:
            raise ImportError('GreenLight is not importable; ' + INSTALL_HINT) from exc
        root = _engine_root(engine)
        verify_definitions(root, files)
    if _engine_root(engine) != root:
        raise RuntimeError('imported GreenLight engine and model definitions come from different trees')
    return root
