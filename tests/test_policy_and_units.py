import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slowlab.campaign_executor import CampaignExecutor
from slowlab.policy import Policy

ROOT = Path(__file__).resolve().parents[1]
V4 = json.loads((ROOT / 'configs/task_contract_v4.json').read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
TICK_DAY = 300 / 86400


# ── Policy ────────────────────────────────────────────────────────────────

def test_policy_validates_once_and_keeps_submitted_values():
    payload = POLICIES['policy_a']
    policy = Policy.from_payload(V4, payload)
    assert policy.as_dict() == payload and list(policy) == list(V4['policy']['fields'])
    assert all(type(policy[k]) is type(payload[k]) for k in payload)   # no int/float coercion
    assert Policy.from_payload(V4, policy) is policy
    with pytest.raises(ValueError):
        Policy.from_payload(V4, dict(payload, night_temperature_c=payload['day_temperature_c'] + 1))
    with pytest.raises(ValueError):
        Policy.from_payload(V4, {k: v for k, v in payload.items() if k != 'co2_target_ppm'})


def test_policy_is_immutable_and_copies_out():
    policy = Policy.from_payload(V4, POLICIES['policy_a'])
    with pytest.raises(AttributeError):
        policy._items = ()
    with pytest.raises(TypeError):
        policy['day_temperature_c'] = 30
    out = policy.as_dict()
    out['day_temperature_c'] = 30
    assert policy['day_temperature_c'] == POLICIES['policy_a']['day_temperature_c']


def test_policy_digest_is_canonical_and_contract_bound():
    payload = POLICIES['policy_a']
    shuffled = dict(reversed(list(payload.items())))
    assert Policy.from_payload(V4, payload).digest() == Policy.from_payload(V4, shuffled).digest()
    assert Policy.from_payload(V4, payload).digest() != Policy.from_payload(V4, POLICIES['policy_b']).digest()
    other = dict(V4, contract_id='another-contract')
    with pytest.raises(ValueError, match='different contract'):
        Policy.from_payload(other, Policy.from_payload(V4, payload))


# ── executor: any declared number of compartments ─────────────────────────

class FakeLifecycle:
    def __init__(self, contract, source, start, **kwargs):
        self.clock = float(start)
        self.mode = 'empty'
        self.ready_at = float(start)
        self.model = SimpleNamespace(full_sol={}, input_data=[{}])
        self.state = {'tAir': 20.0}
        self.cleanup = contract['budget']['cleanup_days'] * 86400

    @property
    def engine(self):
        return self

    def replant(self):
        self.mode = 'active'

    def stop(self):
        self.mode = 'empty'
        self.ready_at = self.clock + self.cleanup

    def step(self, commands, end):
        start = self.clock
        self.clock = float(end)
        self.model.full_sol = {'Time': [start, end], 'hBoilPipe': [0., 0.], 'qLampIn': [0., 0.],
                               'mcExtAir': [0., 0.], 'mcFruitHar': [0., 0.], 'mvCanAir': [0., 0.]}
        return dict(self.state)


def sample(engine, weather, origin, controller, public, unit, ledger):
    now = engine.clock
    controller.advance_to(now)
    public.advance_to(now)
    per_m2 = ledger.summary()['per_m2']
    for name, value in {'air_temperature_c': 20., 'relative_humidity_pct': 75., 'co2_ppm': 500.,
                        'outdoor_temperature_c': 10., 'solar_radiation_w_m2': 100.}.items():
        controller.record(unit, name, measurement_time=now, available_at=now, value=value)
    for name, value in {'air_temperature_c': 20., 'relative_humidity_pct': 75., 'co2_ppm': 500.,
                        'canopy_lai_proxy': 1.0, 'cumulative_harvest_fresh_equivalent': per_m2['harvest_kg_m2'],
                        'heating_energy': per_m2['heat_kwh_m2'], 'lighting_energy': per_m2['light_kwh_m2'],
                        'co2_dosed': per_m2['co2_kg_m2']}.items():
        public.record(unit, name, measurement_time=now, available_at=now, value=value)


def contract_with(compartments):
    c = copy.deepcopy(V4)
    c['facility']['compartments'] = compartments
    c['budget'].update(campaign_days=4 * TICK_DAY, crop_days=2 * TICK_DAY,
                       cleanup_days=TICK_DAY, latest_start_day=2 * TICK_DAY)
    return c


def build(compartments):
    return CampaignExecutor(contract_with(compartments), Path('/unused'), None, feedback_mode='full',
                            fallback_policy=POLICIES['policy_a'], lifecycle_factory=FakeLifecycle,
                            sample_endpoint=sample)


@pytest.mark.parametrize('n', [1, 2, 6])
def test_executor_runs_any_declared_number_of_compartments(n):
    x = build(n)
    assert x.units == tuple(str(i) for i in range(n))
    for unit in range(n):
        x.dispatch({'action': 'start', 'unit': unit, 'policy': POLICIES['policy_b']})
    with pytest.raises(ValueError, match='unknown compartment'):
        x.dispatch({'action': 'start', 'unit': n, 'policy': POLICIES['policy_a']})
    x.dispatch({'action': 'advance', 'day': 4 * TICK_DAY})
    x.dispatch({'action': 'recommend', 'policy': POLICIES['policy_b']})
    settled = x.settlement()
    assert set(settled['ledger_by_unit']) == set(x.units) and settled['starts'] == n
    assert settled['recommendation'] == POLICIES['policy_b']


@pytest.mark.parametrize('bad', [0, -1, 2.0, True])
def test_executor_rejects_invalid_compartment_counts(bad):
    with pytest.raises(ValueError, match='positive integer'):
        build(bad)


def test_mutating_a_submitted_payload_cannot_change_the_running_policy():
    x = build(2)
    payload = dict(POLICIES['policy_a'])
    x.dispatch({'action': 'start', 'unit': 0, 'policy': payload})
    payload['day_temperature_c'] = 26
    assert x._policies['0']['day_temperature_c'] == POLICIES['policy_a']['day_temperature_c']
    logged = next(e for e in x.event_log if e['event'] == 'start')['policy']
    assert logged == POLICIES['policy_a']
    logged['day_temperature_c'] = 26
    assert x._policies['0']['day_temperature_c'] == POLICIES['policy_a']['day_temperature_c']
