"""Resolve pre-carve-up repository paths recorded inside frozen JSON artifacts.

Frozen contracts, locks and audit results are hash-checked and must stay
byte-identical, but several of them name other files by the flat layout used
before the 2026-09 repository carve-up (``configs/v22_*.json``,
``docs/v22_*.md``, ``scripts/*_v22_*.py``). Verifiers call ``frozen_path`` so
those records keep resolving without rewriting the artifacts.
"""
from __future__ import annotations

import re
from pathlib import Path

_AGC_CONFIG_EXTRA = {'greenlight_original_validation_preflight_v0.json', 'root_zone_evidence_v0.json',
                     'v9_data_roles_v0.json'}
_AGC_DOCS = {'agc2019_greenlight_feasibility.md', 'agc2023_development_audit.md',
             'agc2024_input_feasibility.md', 'greenlight_original_validation_preflight.md'}
_V22_FLAT = ('reality_constraints_v2_2', 'simulation_blinding_v2_2')
_ARCHIVE_MODULES = {'agc_lighting', 'agc_temperature_residual', 'agc_greenlight_weather', 'greenhouse_data',
                    'greenlight_sequence', 'root_zone', 'root_zone_response', 'v22_private_weather',
                    'v22_private_weather_v3', 'v22_private_weather_v3a', 'v22_private_weather_v3c',
                    'v22_weather_similarity'}
_V22_MODULES = {'v22_cabauw_weather', 'v22_cached_solver', 'v22_campaign_executor', 'v22_controller',
                'v22_feedback_view', 'v22_greenlight_reuse', 'v22_greenlight_smoke', 'v22_native_rhs',
                'v22_public_sensors', 'v22_resources', 'online_observations', 'task_contract_v22',
                'greenlight_adapter', 'prompt_firewall', 'reality_constraints'}


def _strip22(name: str) -> str:
    return re.sub(r'(^v22_|_v22(?=[._])|_v22$)', '', name)


def candidates(legacy: str) -> list[str]:
    """Current-layout paths that a legacy relative path may now live at, most likely first."""
    parts = legacy.split('/')
    if len(parts) != 2:
        return [legacy]
    top, name = parts
    if top == 'configs':
        if name.startswith('agc') or name in _AGC_CONFIG_EXTRA:
            return [f'configs/agc/{name}']
        if name.startswith(_V22_FLAT):
            return [f'configs/v22/{name}']
        if name.startswith('v22_'):
            stripped = _strip22(name)
            return [f'configs/v22/{stripped}', f'results/v22/{stripped}']
    elif top == 'docs':
        if name in _AGC_DOCS:
            return [f'docs/agc/{name}']
        if name.startswith('v22_') or name.startswith(_V22_FLAT):
            return [f'docs/v22/{_strip22(name)}']
    elif top == 'scripts' and name.endswith('.py'):
        stem = name[:-3]
        if 'agc' in stem or stem == 'run_greenlight_weather_smoke':
            return [f'scripts/agc/{name}']
        if 'v22' in stem or stem in {'audit_reality_support', 'verify_phase0_reality_contract'}:
            return [f'scripts/v22/{_strip22(name)}']
    elif top == 'slowlab' and name.endswith('.py'):
        stem = name[:-3]
        if stem in _ARCHIVE_MODULES:
            return [f'slowlab/archive/{_strip22(stem)}.py']
        if stem in _V22_MODULES:
            new = 'task_contract' if stem == 'task_contract_v22' else _strip22(stem)
            return [f'slowlab/v22/{new}.py']
    return [legacy]


def frozen_path(root: Path, legacy: str) -> Path:
    """Return the existing current path for ``legacy``; fall back to the literal path."""
    if (root / legacy).exists():
        return root / legacy
    for option in candidates(legacy):
        if (root / option).exists():
            return root / option
    return root / legacy
