"""Scan public run outputs for private information before they leave the executor.

Public material is everything a method, a provider or a reader of released
artifacts may see: agent transcripts, outbound audit logs, public summaries and
their file names. The scan looks for

* the blinding policy's forbidden terms (benchmark, simulator and seed words);
* data-source and path identity (weather archive names, private directories);
* the hidden campaign weather year;
* hidden site values: per-compartment model parameters, deep-soil temperature,
  side-wall exchange coefficient and the sensor-noise seed, in the textual forms
  a program would print them.

Public prices are not secret and are not scanned. A finding is an ``error``
when the match is exact (full repr, six significant digits, or an integer
seed) and a ``warning`` for five-significant-digit matches, which can occur by
chance and need a human look.
"""
from __future__ import annotations

import gzip
import json
import re
from datetime import datetime
from pathlib import Path

SOURCE_PATTERNS = (r'\bcabauw\b', r'\bcesar\b', r'\bknmi\b', r'\blc1\b', r'\bkatzin\b', r'\bvanthoor\b')
TEXT_SUFFIXES = {'.json', '.jsonl', '.txt', '.log', '.md', '.csv'}


def _number_forms(value: float) -> list[tuple[str, str]]:
    forms = {repr(float(value)): 'error', f'{value:.6g}': 'error', f'{value:.5g}': 'warning'}
    out = []
    for text, level in forms.items():
        digits = re.sub(r'[^0-9]', '', text.split('e')[0]).lstrip('0')
        if len(digits) >= 4:  # shorter forms match by chance too often
            out.append((text, level))
    return out


def private_needles(spec: dict, policy: dict | None = None) -> list[dict]:
    """Patterns derived from one private site spec (the executor-server spec)."""
    needles = []
    for pattern in (policy or {}).get('forbidden_patterns', ()):
        needles.append({'kind': 'blinding_term', 'regex': pattern, 'level': 'error'})
    for pattern in SOURCE_PATTERNS:
        needles.append({'kind': 'data_source', 'regex': pattern, 'level': 'error'})
    for key in ('trace', 'settlement_out', 'failure_out'):
        if spec.get(key):
            needles.append({'kind': 'private_path', 'regex': re.escape(str(Path(spec[key]).parent)), 'level': 'error'})
    weather = spec.get('weather') or {}
    for key in ('cache', 'audit', 'dev_cache', 'plan'):
        if weather.get(key):
            needles.append({'kind': 'private_path', 'regex': re.escape(Path(weather[key]).name), 'level': 'error'})
    if spec.get('origin_utc'):
        year = datetime.fromisoformat(spec['origin_utc']).year + 1
        needles.append({'kind': 'weather_year', 'regex': rf'(?<![\d.]){year}(?![\d.])', 'level': 'warning'})
    values = []
    site = spec.get('site') or {}
    for unit, params in (site.get('unit_parameters') or {}).items():
        values += [(f'unit {unit} {name}', v) for name, v in params.items()]
    if site.get('boundary_ueff_w_m2_k') is not None:
        values.append(('boundary_ueff_w_m2_k', site['boundary_ueff_w_m2_k']))
    if spec.get('soil_boundary_c') is not None:
        values.append(('soil_boundary_c', spec['soil_boundary_c']))
    for label, value in values:
        for text, level in _number_forms(float(value)):
            needles.append({'kind': 'site_value', 'what': label, 'regex': rf'(?<![\d.]){re.escape(text)}(?!\d)',
                            'level': level})
    seed = (spec.get('sensor_noise') or {}).get('seed')
    if seed is not None:
        needles.append({'kind': 'noise_seed', 'regex': rf'(?<!\d){int(seed)}(?!\d)', 'level': 'error'})
    return needles


def _read_text(path: Path) -> str | None:
    if path.suffix == '.gz':
        inner = Path(path.stem)
        if inner.suffix not in TEXT_SUFFIXES:
            return None
        with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
            return f.read()
    if path.suffix in TEXT_SUFFIXES:
        return path.read_text(encoding='utf-8', errors='replace')
    return None


def scan(paths: list[Path], needles: list[dict], private_dirs: list[Path] = ()) -> dict:
    compiled = [(n, re.compile(n['regex'], re.IGNORECASE)) for n in needles]
    findings, scanned, skipped = [], [], []
    files = []
    for root in map(Path, paths):
        files += sorted(p for p in root.rglob('*') if p.is_file()) if root.is_dir() else [root]
    for private in map(Path, private_dirs):
        for root in map(Path, paths):
            if private.resolve() == root.resolve() or root.resolve() in private.resolve().parents:
                findings.append({'level': 'error', 'kind': 'layout', 'file': str(private),
                                 'match': 'private directory lies inside public output'})
    for path in files:
        for needle, rx in compiled:
            if rx.search(path.name):
                findings.append({'level': needle['level'], 'kind': needle['kind'], 'file': str(path),
                                 'where': 'file name', 'match': path.name, **({'what': needle['what']} if 'what' in needle else {})})
        text = _read_text(path)
        if text is None:
            skipped.append(str(path))
            findings.append({'level': 'warning', 'kind': 'unscanned_file', 'file': str(path),
                             'where': 'whole file', 'match': 'not a known text format'})
            continue
        scanned.append(str(path))
        for needle, rx in compiled:
            for m in rx.finditer(text):
                findings.append({'level': needle['level'], 'kind': needle['kind'], 'file': str(path),
                                 'where': f'offset {m.start()}', 'match': text[max(0, m.start() - 30):m.end() + 30],
                                 **({'what': needle['what']} if 'what' in needle else {})})
                break
    errors = sum(f['level'] == 'error' for f in findings)
    return {'status': 'fail' if errors else ('review' if findings else 'pass'), 'errors': errors,
            'warnings': len(findings) - errors, 'findings': findings, 'scanned': scanned,
            'skipped_binary_or_unknown': skipped}
