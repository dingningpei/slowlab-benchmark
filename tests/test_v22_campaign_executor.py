import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slowlab.v22_campaign_executor import CampaignExecutor
from slowlab.v22_greenlight_reuse import CropLifecycle

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / 'configs/v22_task_contract_v3.json').read_text())
POLICIES = json.loads((ROOT / 'configs/v22_campaign_example_v0.json').read_text())
TICK_DAY = 300 / 86400


class FakeLifecycle:
    def __init__(self, contract, source, start, **kwargs):
        assert kwargs['initially_empty'] is True
        self.clock = float(start)
        self.mode = 'empty'
        self.ready_at = float(start)
        self.model = SimpleNamespace(full_sol={}, input_data=[{}])
        self.state = {'tAir': 20.0}
        self.events = []
        self.cleanup = contract['budget']['cleanup_days'] * 86400

    @property
    def engine(self):
        return self

    def replant(self):
        assert self.mode == 'empty' and self.clock >= self.ready_at
        self.mode = 'active'
        self.events.append('plant' if not self.events else 'replant')

    def stop(self):
        assert self.mode == 'active'
        self.mode = 'empty'
        self.ready_at = self.clock + self.cleanup
        self.events.append('stop')

    def step(self, commands, end):
        start = self.clock
        assert end == start + 300
        self.clock = float(end)
        self.model.full_sol = {'Time': [start, end], 'hBoilPipe': [0., 0.],
                               'qLampIn': [0., 0.], 'mcExtAir': [0., 0.],
                               'mcFruitHar': [0., 0.], 'mvCanAir': [0., 0.]}
        return dict(self.state)


def sample(engine, weather, origin, controller, public, unit, ledger):
    now = engine.clock
    controller.advance_to(now)
    public.advance_to(now)
    control_values = {'air_temperature_c': engine.state['tAir'], 'relative_humidity_pct': 75.,
                      'co2_ppm': 500., 'outdoor_temperature_c': 10.,
                      'solar_radiation_w_m2': 100.}
    public_values = {'air_temperature_c': engine.state['tAir'], 'relative_humidity_pct': 75.,
                     'co2_ppm': 500., 'canopy_lai_proxy': 1. if engine.mode == 'active' else 0.,
                     'cumulative_harvest_fresh_equivalent': ledger.summary()['per_m2']['harvest_kg_m2'],
                     'heating_energy': ledger.summary()['per_m2']['heat_kwh_m2'],
                     'lighting_energy': ledger.summary()['per_m2']['light_kwh_m2'],
                     'co2_dosed': ledger.summary()['per_m2']['co2_kg_m2']}
    for name, value in control_values.items():
        controller.record(unit, name, measurement_time=now, available_at=now, value=value)
    for name, value in public_values.items():
        public.record(unit, name, measurement_time=now, available_at=now, value=value)


def executor(mode='full', trace=None):
    contract = copy.deepcopy(BASE)
    contract['budget'].update(campaign_days=4 * TICK_DAY,
                              crop_days=2 * TICK_DAY,
                              cleanup_days=TICK_DAY,
                              latest_start_day=2 * TICK_DAY)
    return CampaignExecutor(contract, Path('/unused'), None, feedback_mode=mode,
                            fallback_policy=POLICIES['policy_a'],
                            lifecycle_factory=FakeLifecycle,
                            sample_endpoint=sample, trace_sink=trace)


def test_stagger_stop_cleanup_replant_and_deadline_settlement():
    trace = io.StringIO()
    x = executor(trace=trace)
    a, b = POLICIES['policy_a'], POLICIES['policy_b']
    x.dispatch({'action': 'start', 'unit': 0, 'policy': a})
    x.dispatch({'action': 'start', 'unit': 1, 'policy': b})
    x.dispatch({'action': 'advance', 'day': TICK_DAY})
    seen = x.dispatch({'action': 'observe', 'unit': 0, 'variable': 'air_temperature_c'})
    assert len(seen['records']) == 2
    with pytest.raises(ValueError):
        x.dispatch({'action': 'observe', 'unit': 0, 'variable': 'air_temperature_c',
                    'end_day': 2 * TICK_DAY})
    x.dispatch({'action': 'stop', 'unit': 0})
    x.dispatch({'action': 'advance', 'day': 2 * TICK_DAY})
    x.dispatch({'action': 'start', 'unit': 0, 'policy': b})
    x.dispatch({'action': 'advance', 'day': 4 * TICK_DAY})
    assert x.dispatch({'action': 'recommend', 'policy': a})['fallback'] is False
    settled = x.settlement()
    assert settled['starts'] == 3 and settled['trace_ticks'] == 4
    assert settled['trace_complete'] is True
    assert len(trace.getvalue().splitlines()) == 4
    assert any(e['event'] == 'normal_completion' and e['unit'] == '1' for e in x.event_log)
    assert [e['event'] for e in x.event_log if e.get('unit') == '0' and e['event'] != 'observe'] == [
        'start', 'stop', 'cleanup_complete', 'start', 'normal_completion']
    assert settled['ledger_by_unit']['0']['per_m2']['days'] == pytest.approx(4 * TICK_DAY)


def test_endpoint_cannot_read_science_but_sees_closed_crop():
    x = executor('endpoint')
    x.dispatch({'action': 'start', 'unit': 0, 'policy': POLICIES['policy_a']})
    assert x.dispatch({'action': 'observe', 'unit': 0})['final_aggregate'] is None
    with pytest.raises(PermissionError):
        x.dispatch({'action': 'observe', 'unit': 0, 'variable': 'air_temperature_c'})
    x.dispatch({'action': 'advance', 'day': TICK_DAY})
    x.dispatch({'action': 'stop', 'unit': 0})
    closed = x.dispatch({'action': 'observe', 'unit': 0})['final_aggregate']
    assert closed['reason'] == 'stop' and closed['closed_at'] == 300
    assert 'trajectory' not in json.dumps(closed)


def test_start_limits_and_immutable_running_policy():
    x = executor()
    a = POLICIES['policy_a']
    x.dispatch({'action': 'start', 'unit': 0, 'policy': a})
    with pytest.raises(ValueError):
        x.dispatch({'action': 'start', 'unit': 0, 'policy': a})
    with pytest.raises(ValueError):
        x.dispatch({'action': 'modify_running_policy', 'unit': 0, 'policy': a})
    with pytest.raises(ValueError):
        x.dispatch({'action': 'advance', 'day': 1.5 * TICK_DAY})


def test_initially_empty_real_lifecycle_transition_uses_first_plant_label(monkeypatch):
    import slowlab.v22_greenlight_reuse as module
    monkeypatch.setattr(module, 'ReusableGreenLight',
                        lambda *args, **kwargs: SimpleNamespace(
                            state={'cBuf': 1., 'cFruit': 1., 'cLeaf': 1., 'cStem': 1.,
                                   'tCanSum': 1., 'tCan24': 20., 'tCan': 20., 'tAir': 20.},
                            clock=0.))
    life = CropLifecycle(BASE, Path('/unused'), start=0, initially_empty=True)
    assert life.mode == 'empty' and life.ready_at == 0
    life.replant()
    assert life.mode == 'active' and life.events[-1]['event'] == 'plant'


def test_sustained_unsafe_temperature_stops_crop_and_charges_cleanup():
    x = executor()
    x.contract['budget']['campaign_days'] = 20 * TICK_DAY
    x.deadline_seconds = 20 * 300
    x.contract['budget']['crop_days'] = 20 * TICK_DAY
    x.dispatch({'action': 'start', 'unit': 0, 'policy': POLICIES['policy_a']})
    x._lifecycles['0'].state['tAir'] = 45.
    x.dispatch({'action': 'advance', 'day': 12 * TICK_DAY})
    assert any(e['event'] == 'safety_stop' and e['unit'] == '0' for e in x.event_log)
    assert x._view().operational_status('0')['phase'] == 'cleanup'
    assert x._ledgers['0'].summary()['event_cost_eur_m2'] == pytest.approx(2.5)


def test_numerical_failure_pauses_campaign_and_preserves_identity():
    x = executor()
    def fail(*args):
        raise FloatingPointError('injected model fault')
    x._lifecycles['2'].step = fail
    with pytest.raises(FloatingPointError):
        x.dispatch({'action': 'advance', 'day': TICK_DAY})
    event = x.event_log[-1]
    assert event == {'event': 'infrastructure_failure', 'unit': '2', 'clock': 0,
                     'error': 'FloatingPointError: injected model fault'}
    with pytest.raises(RuntimeError, match='paused'):
        x.dispatch({'action': 'advance', 'day': 2 * TICK_DAY})


def test_identical_public_history_masks_different_private_state():
    left, right = executor(), executor()
    left._lifecycles['0'].state['private_yield_parameter'] = 1.0
    right._lifecycles['0'].state['private_yield_parameter'] = 999.0
    actions = [
        {'action': 'start', 'unit': 0, 'policy': POLICIES['policy_a']},
        {'action': 'advance', 'day': TICK_DAY},
        {'action': 'observe', 'unit': 0, 'variable': 'air_temperature_c'},
    ]
    assert [left.dispatch(action) for action in actions] == [right.dispatch(action) for action in actions]


def test_private_progress_hook_runs_only_after_completed_day():
    x = executor()
    x.contract['budget']['campaign_days'] = 1
    x.deadline_seconds = 86400
    x.contract['budget']['crop_days'] = 2
    seen = []
    x._progress_hook = lambda campaign: seen.append((campaign.clock, campaign._trace_ticks))
    x.dispatch({'action': 'advance', 'day': 1})
    assert seen == [(86400, 288)]
