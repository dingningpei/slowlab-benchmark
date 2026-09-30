import hashlib
import sys
import types
from pathlib import Path

import pytest

from slowlab.greenlight_source import DEFINITIONS, pinned_files, resolve_greenlight_source, verify_definitions

ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = Path('/private/tmp/greenlight-v2.0.5')


def fake_tree(tmp_path, tamper=None):
    root = tmp_path / 'gl'
    files = {}
    for name in ('main.json', 'sub/part.json'):
        path = root / DEFINITIONS / name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = f'{{"name": "{name}"}}'
        path.write_text(text)
        files[name] = hashlib.sha256(text.encode()).hexdigest()
    (root / 'greenlight').mkdir(exist_ok=True)
    (root / 'greenlight' / '__init__.py').write_text('')
    if tamper:
        (root / DEFINITIONS / tamper).write_text('changed')
    return root, {'model': {'files': files}}


@pytest.fixture
def clean_modules(monkeypatch):
    monkeypatch.delitem(sys.modules, 'greenlight', raising=False)
    monkeypatch.setattr(sys, 'path', list(sys.path))


def test_pinned_files_come_from_the_model_family_by_default():
    assert pinned_files() == pinned_files({'model': {'files': pinned_files()}})
    assert len(pinned_files()) == 4


def test_verify_definitions_detects_missing_and_changed_files(tmp_path):
    root, contract = fake_tree(tmp_path)
    verify_definitions(root, contract['model']['files'])
    with pytest.raises(FileNotFoundError, match='pip install'):
        verify_definitions(tmp_path / 'nowhere', contract['model']['files'])
    root, contract2 = fake_tree(tmp_path / 'b', tamper='sub/part.json')
    with pytest.raises(ValueError, match='does not match the pinned commit'):
        verify_definitions(root, contract2['model']['files'])


def test_source_checkout_is_verified_and_put_on_the_path(tmp_path, clean_modules):
    root, contract = fake_tree(tmp_path)
    assert resolve_greenlight_source(root, contract) == root.resolve()
    assert sys.path[0] == str(root.resolve())
    assert Path(sys.modules['greenlight'].__file__).resolve().parent.parent == root.resolve()


def test_refuses_to_mix_an_already_imported_engine_with_another_source(tmp_path, clean_modules, monkeypatch):
    root, contract = fake_tree(tmp_path)
    other = types.ModuleType('greenlight')
    other.__file__ = str(tmp_path / 'elsewhere' / 'greenlight' / '__init__.py')
    monkeypatch.setitem(sys.modules, 'greenlight', other)
    with pytest.raises(RuntimeError, match='already imported'):
        resolve_greenlight_source(root, contract)


def test_missing_install_gives_an_actionable_error(clean_modules, monkeypatch):
    monkeypatch.setattr(sys, 'path', [p for p in sys.path if 'greenlight' not in p.lower()])
    monkeypatch.setitem(sys.modules, 'greenlight', None)   # forces ImportError
    with pytest.raises(ImportError, match=r"pip install -e '\.\[greenlight\]'"):
        resolve_greenlight_source(None)


@pytest.mark.skipif(not CHECKOUT.is_dir(), reason='local pinned GreenLight checkout not present')
def test_local_pinned_checkout_matches_the_contract():
    import json
    contract = json.loads((ROOT / 'configs/task_contract_v4.json').read_text())
    verify_definitions(CHECKOUT, contract['model']['files'])
