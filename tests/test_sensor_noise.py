import copy
import io
import json
import math
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from slowlab.campaign_executor import CampaignExecutor
from slowlab.online_observations import PackedOnlineObservations
from slowlab.sensor_bridge import record_at_endpoint
from slowlab.sensor_noise import SensorNoise, verify_trace_row

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/sensor_noise_v0.json').read_text())
CONTRACT = json.loads((ROOT / 'configs/task_contract_v3.json').read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
TICK = 300
ORIGIN = datetime(2016, 12, 31, 23, tzinfo=timezone.utc)


def noise(seed=7, setting='main', config=None):
    return SensorNoise(config or CONFIG, seed, setting)


# ── the model itself ──────────────────────────────────────────────────────

def test_config_covers_every_public_channel_and_names_settings():
    model = noise()
    public = set(CONTRACT['observations']['public_channels'])
    assert public == model.channels | model.exact_channels
    assert not model.channels & model.exact_channels
    assert CONFIG['status'] == 'development_assumption_not_calibrated'
    bias = noise(setting='sensitivity_sensor_bias')._spec['channels']
    assert bias['air_temperature_c']['sensor_bias_sd_abs'] == 0.3
    assert model._spec['channels']['air_temperature_c']['sensor_bias_sd_abs'] == 0.0


def test_same_reading_identity_always_gives_the_same_value():
    a, b = noise(7), noise(7)
    values = [a.observed_value('co2_ppm', '1', t, 650.0) for t in range(300, 30000, 300)]
    # query in reverse order on an independent instance: order cannot matter
    again = [b.observed_value('co2_ppm', '1', t, 650.0) for t in reversed(range(300, 30000, 300))]
    assert values == list(reversed(again))
    assert values != [noise(8).observed_value('co2_ppm', '1', t, 650.0) for t in range(300, 30000, 300)]
    assert (a.observed_value('co2_ppm', '1', 600, 650.0)
            != a.observed_value('co2_ppm', '2', 600, 650.0))


def test_random_draw_does_not_depend_on_the_true_value():
    model = noise(3)
    for t in range(300, 60000, 300):
        low = model.observed_value('air_temperature_c', '0', t, 20.0) - 20.0
        high = model.observed_value('air_temperature_c', '0', t, 25.0) - 25.0
        assert low == pytest.approx(high, abs=1e-9)


def test_additive_error_matches_declared_scale():
    model = noise(11)
    errors = [model.observed_value('air_temperature_c', '0', t, 20.0) - 20.0
              for t in range(300, 300 * 20001, 300)]
    assert abs(statistics.fmean(errors)) < 0.01
    rounding = 0.1 / math.sqrt(12)
    assert statistics.pstdev(errors) == pytest.approx(math.hypot(0.2, rounding), rel=0.05)
    co2 = [model.observed_value('co2_ppm', '0', t, 1000.0) - 1000.0
           for t in range(300, 300 * 20001, 300)]
    assert statistics.pstdev(co2) == pytest.approx(math.hypot(10.0, 20.0), rel=0.05)


def test_climate_box_drops_all_three_channels_together_at_declared_rate():
    model = noise(5)
    truth = {'air_temperature_c': 20., 'relative_humidity_pct': 70., 'co2_ppm': 600.,
             'canopy_lai_proxy': 2.0, 'heating_energy': 1.0}
    box_drops = camera_drops = 0
    n = 100000
    for k in range(1, n + 1):
        observed, dropped = model.observe('2', k * TICK, truth)
        box = {'air_temperature_c', 'relative_humidity_pct', 'co2_ppm'} & set(dropped)
        assert box in (set(), {'air_temperature_c', 'relative_humidity_pct', 'co2_ppm'})
        box_drops += bool(box)
        camera_drops += 'canopy_lai_proxy' in dropped
        assert observed['heating_energy'] == 1.0
        assert not set(dropped) & set(observed)
    assert 400 <= box_drops <= 600          # p = 0.005
    assert 1750 <= camera_drops <= 2250     # p = 0.02


def test_no_dropout_at_campaign_start_even_when_dropout_is_likely():
    config = copy.deepcopy(CONFIG)
    for group in config['settings']['main']['sensor_groups'].values():
        group['dropout_probability'] = 0.999
    model = noise(1, config=config)
    assert model.dropped_groups('0', 0) == frozenset()
    assert model.dropped_groups('0', TICK) == frozenset({'climate_box', 'canopy_camera'})


def test_lai_error_is_shared_within_a_day_and_independent_across_days():
    model = noise(13)
    within, daily = [], []
    for day in range(400):
        logs = [math.log(model.observed_value('canopy_lai_proxy', '0', day * 86400 + k * TICK, 3.0) / 3.0)
                for k in range(1, 289)]
        daily.append(statistics.fmean(logs))
        within.append(statistics.pstdev(logs))
    assert statistics.fmean(within) == pytest.approx(0.03, rel=0.15)
    assert statistics.pstdev(daily) == pytest.approx(0.10, rel=0.12)
    assert model.observed_value('canopy_lai_proxy', '0', 600, 0.0) == 0.0


def test_clipping_and_resolution():
    model = noise(17)
    for t in range(300, 300 * 3000, 300):
        rh = model.observed_value('relative_humidity_pct', '0', t, 99.9)
        assert 0.0 <= rh <= 100.0 and rh == round(rh, 1)
        co2 = model.observed_value('co2_ppm', '0', t, 1.0)
        assert co2 >= 0.0 and co2 == round(co2)


def test_rejects_undeclared_channels_bad_seeds_and_fractional_times():
    model = noise()
    with pytest.raises(ValueError, match='no declared measurement model'):
        model.observe('0', TICK, {'secret_crop_pool': 1.0})
    for seed in (-1, 2 ** 64, True, 1.5):
        with pytest.raises(ValueError):
            noise(seed)
    with pytest.raises(ValueError):
        model.observed_value('co2_ppm', '0', 300.5, 600.0)


# ── sensor bridge: one physical sensor, two stores ────────────────────────

class FakeLedger:
    def __init__(self):
        self.clock = 0.0

    def summary(self):
        return {'per_m2': {'harvest_kg_m2': 0.5, 'heat_kwh_m2': 3.0,
                           'light_kwh_m2': 1.0, 'co2_kg_m2': 0.2}}


class FakeWeather:
    def at_utc(self, when):
        return SimpleNamespace(interval_end_utc=when + timedelta(microseconds=1),
                               t_out_c=5.0, i_glob_w_m2=150.0)


def fake_engine(clock, t_air=21.0):
    return SimpleNamespace(clock=float(clock),
                           state={'tAir': t_air, 'cLeaf': 1.0e5},
                           model=SimpleNamespace(full_sol={'Time': [float(clock)], 'rhIn': [80.0],
                                                           'co2InPpm': [700.0]},
                                                 consts={'sla': 2.66e-5}))


def stores():
    obs = CONTRACT['observations']
    return (PackedOnlineObservations(obs['controller_channels']),
            PackedOnlineObservations(obs['public_channels']))


def test_bridge_without_noise_records_the_deterministic_values():
    controller, public = stores()
    ledger = FakeLedger()
    inside, _ = record_at_endpoint(fake_engine(0), FakeWeather(), ORIGIN, controller, public, '0', ledger)
    assert inside == {'air_temperature_c': 21.0, 'relative_humidity_pct': 80.0, 'co2_ppm': 700.0}
    assert public.latest('0', 'co2_ppm')['value'] == 700.0
    assert controller.latest('0', 'air_temperature_c')['value'] == 21.0


def test_bridge_with_noise_writes_one_reading_to_both_stores_and_drops_in_both():
    config = copy.deepcopy(CONFIG)
    config['settings']['main']['sensor_groups']['climate_box']['dropout_probability'] = 0.3
    model = noise(21, config=config)
    controller, public = stores()
    ledger = FakeLedger()
    drops = 0
    for k in range(0, 200):
        ledger.clock = k * TICK
        *_, detail = record_at_endpoint(fake_engine(k * TICK), FakeWeather(), ORIGIN, controller, public,
                                        '0', ledger, sensor_noise=model, return_detail=True)
        assert set(detail['sensor_truth']) == model.channels
        for name in ('air_temperature_c', 'relative_humidity_pct', 'co2_ppm'):
            c = controller.latest('0', name)
            p = public.latest('0', name)
            assert c['value'] == p['value'] and c['measurement_time'] == p['measurement_time']
            if name in detail['sensor_dropped']:
                assert p['measurement_time'] < k * TICK
            else:
                assert p['measurement_time'] == k * TICK
                assert p['value'] == model.observed_value(name, '0', k * TICK, detail['sensor_truth'][name])
        drops += bool(detail['sensor_dropped'])
        assert public.latest('0', 'heating_energy')['value'] == 3.0
    assert drops > 20
    # repeated queries return the stored reading, never a fresh draw
    first = public.history('0', 'air_temperature_c')
    assert first == public.history('0', 'air_temperature_c')
    assert len(first) == 200 - sum(1 for k in range(1, 200) if 'climate_box' in model.dropped_groups('0', k * TICK))


# ── executor end to end with a fake crop model ────────────────────────────

class FakeLifecycle:
    def __init__(self, contract, source, start, **kwargs):
        self.clock = float(start)
        self.mode = 'empty'
        self.ready_at = float(start)
        self.state = {'tAir': 20.0, 'cLeaf': 0.0}
        self.model = SimpleNamespace(full_sol={'Time': [self.clock], 'rhIn': [75.0], 'co2InPpm': [500.0]},
                                     input_data=[{}], consts={'sla': 2.66e-5})
        self.cleanup = contract['budget']['cleanup_days'] * 86400

    @property
    def engine(self):
        return self

    def replant(self):
        self.mode = 'active'
        self.state['cLeaf'] = 5.0e4

    def stop(self):
        self.mode = 'empty'
        self.state['cLeaf'] = 0.0
        self.ready_at = self.clock + self.cleanup

    def step(self, commands, end):
        start = self.clock
        self.clock = float(end)
        self.state['tAir'] = 20.0 + (end / 300) % 3
        self.model.full_sol = {'Time': [start, end], 'hBoilPipe': [0., 0.], 'qLampIn': [0., 0.],
                               'mcExtAir': [0., 0.], 'mcFruitHar': [0., 0.], 'mvCanAir': [0., 0.],
                               'rhIn': [75.0, 76.0], 'co2InPpm': [500.0, 510.0]}
        return dict(self.state)


def small_contract(ticks=12):
    contract = copy.deepcopy(CONTRACT)
    contract['budget'].update(campaign_days=ticks * TICK / 86400, crop_days=6 * TICK / 86400,
                              cleanup_days=2 * TICK / 86400, latest_start_day=4 * TICK / 86400)
    return contract


def run_campaign(model, mode='full', extra_observes=0, ticks=12):
    trace = io.StringIO()
    campaign = CampaignExecutor(small_contract(ticks), Path('/unused'), FakeWeather(), feedback_mode=mode,
                                fallback_policy=POLICIES['policy_a'], lifecycle_factory=FakeLifecycle,
                                trace_sink=trace, sensor_noise=model)
    campaign.dispatch({'action': 'start', 'unit': 0, 'policy': POLICIES['policy_a']})
    campaign.dispatch({'action': 'advance', 'day': 5 * TICK / 86400})
    if mode == 'full':
        for _ in range(extra_observes):
            campaign.dispatch({'action': 'observe', 'unit': 0, 'variable': 'air_temperature_c'})
    campaign.dispatch({'action': 'advance', 'day': ticks * TICK / 86400})
    campaign.dispatch({'action': 'recommend', 'policy': POLICIES['policy_b']})
    frames = [json.loads(line) for line in trace.getvalue().splitlines()]
    return campaign, frames


def test_executor_trace_rows_reproduce_from_truth_and_model():
    config = copy.deepcopy(CONFIG)
    config['settings']['main']['sensor_groups']['climate_box']['dropout_probability'] = 0.25
    model = noise(99, config=config)
    campaign, frames = run_campaign(model)
    assert campaign.settlement()['sensor_noise'] == {'noise_id': 'slowlab-sensor-noise-v0',
                                                     'setting': 'main'}
    previous = {}
    dropped_any = False
    for frame in frames:
        for unit, row in frame['units'].items():
            verify_trace_row(model, unit, frame['end'], row, previous.get(unit))
            dropped_any |= bool(row['sensor_dropped'])
            previous[unit] = row
    assert dropped_any
    # a tampered reading is caught
    row = copy.deepcopy(frames[3]['units']['0'])
    kept = next(c for c in ('air_temperature_c', 'co2_ppm') if c not in row['sensor_dropped'])
    row['public_endpoint'][kept]['value'] += 0.1
    with pytest.raises(AssertionError):
        verify_trace_row(model, '0', frames[3]['end'], row, frames[2]['units']['0'])


def test_repeated_observation_returns_identical_records():
    campaign, _ = run_campaign(noise(5))
    one = campaign.dispatch({'action': 'observe', 'unit': 0, 'variable': 'co2_ppm', 'start_day': 0})
    two = campaign.dispatch({'action': 'observe', 'unit': 0, 'variable': 'co2_ppm', 'start_day': 0})
    assert one['records'] == two['records'] and one['records']


def test_queries_and_feedback_condition_do_not_change_readings():
    _, quiet = run_campaign(noise(42), mode='full', extra_observes=0)
    _, busy = run_campaign(noise(42), mode='full', extra_observes=5)
    _, endpoint = run_campaign(noise(42), mode='endpoint')
    for a, b, c in zip(quiet, busy, endpoint):
        for unit in a['units']:
            assert a['units'][unit]['public_endpoint'] == b['units'][unit]['public_endpoint'] \
                == c['units'][unit]['public_endpoint']


def test_noise_requires_the_built_in_bridge():
    with pytest.raises(ValueError, match='built-in sensor bridge'):
        CampaignExecutor(small_contract(), Path('/unused'), FakeWeather(), feedback_mode='full',
                         fallback_policy=POLICIES['policy_a'], lifecycle_factory=FakeLifecycle,
                         sample_endpoint=lambda *args: None, sensor_noise=noise())
