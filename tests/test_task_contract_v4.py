import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slowlab.campaign_executor import CampaignExecutor, resolve_soil_boundary

ROOT = Path(__file__).resolve().parents[1]
V3_PATH = ROOT / 'configs/task_contract_v3.json'
V4_PATH = ROOT / 'configs/task_contract_v4.json'
V3 = json.loads(V3_PATH.read_text())
V4 = json.loads(V4_PATH.read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v4_supersedes_the_exact_v3_file():
    assert V4['supersedes'] == {**V4['supersedes'], 'contract_id': V3['contract_id'], 'sha256': sha(V3_PATH)}


def test_v4_changes_only_soil_noise_and_identity():
    old, new = copy.deepcopy(V3), copy.deepcopy(V4)
    for contract in (old, new):
        for key in ('contract_id', 'date', 'supersedes'):
            contract.pop(key)
        contract['facility'].pop('deep_soil_boundary', None)
        contract['observations'].pop('noise')
    assert old == new


def test_v4_soil_boundary_declares_native_default_and_site_range():
    soil = V4['facility']['deep_soil_boundary']
    assert soil['greenlight_input'] == 'tSoOut'
    assert soil['development_default_c'] == 20.0
    assert soil['site_parameter']['range_c'] == [8.0, 20.0]
    assert soil['site_parameter']['from_phase'] == 3 and soil['site_parameter']['private'] is True
    assert (ROOT / soil['evidence']['annual_paired_audit']).is_file()
    assert 'deep_soil' not in json.dumps(V4['observations'])  # never a public channel


def test_v4_pins_the_sensor_noise_model_by_hash():
    noise = V4['observations']['noise']
    assert noise['sha256'] == sha(ROOT / noise['model'])
    assert noise['setting'] == 'main'


def test_resolve_soil_boundary_defaults_and_range():
    assert resolve_soil_boundary(V3) == 20.0
    assert resolve_soil_boundary(V4) == 20.0
    assert resolve_soil_boundary(V4, 8) == 8.0
    assert resolve_soil_boundary(V4, 14.5) == 14.5
    for bad in (7.9, 20.1, float('nan'), True, '12'):
        with pytest.raises(ValueError):
            resolve_soil_boundary(V4, bad)
    with pytest.raises(ValueError, match='no deep-soil site parameter'):
        resolve_soil_boundary(V3, 12.0)


class CapturingLifecycle:
    seen = []

    def __init__(self, contract, source, start, **kwargs):
        CapturingLifecycle.seen.append(kwargs['soil_boundary_c'])
        self.clock = float(start)
        self.mode = 'empty'
        self.ready_at = float(start)
        self.model = SimpleNamespace(full_sol={}, input_data=[{}])
        self.state = {'tAir': 20.0}

    @property
    def engine(self):
        return self


def build(contract, **kwargs):
    CapturingLifecycle.seen = []
    return CampaignExecutor(contract, Path('/unused'), None, feedback_mode='full',
                            fallback_policy=POLICIES['policy_a'], lifecycle_factory=CapturingLifecycle,
                            sample_endpoint=lambda *args: None, **kwargs)


def test_executor_passes_one_soil_boundary_to_every_compartment():
    assert build(V3).soil_boundary_c == 20.0 and CapturingLifecycle.seen == [20.0] * 4
    assert build(V4).soil_boundary_c == 20.0 and CapturingLifecycle.seen == [20.0] * 4
    assert build(V4, soil_boundary_c=11.0).soil_boundary_c == 11.0
    assert CapturingLifecycle.seen == [11.0] * 4
    with pytest.raises(ValueError, match='outside the declared site range'):
        build(V4, soil_boundary_c=25.0)
