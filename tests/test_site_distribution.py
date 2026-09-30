import copy
import json
from pathlib import Path

import pytest

from slowlab.agent_client import CampaignProcess
from slowlab.site_distribution import check_site, draw, sample_site
from slowlab.site_parameters import MODEL_PARAMETERS

ROOT = Path(__file__).resolve().parents[1]
DIST = json.loads((ROOT / 'configs/site_distribution_v0.json').read_text())
CONTRACT = json.loads((ROOT / 'configs/task_contract_v5.json').read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
TICK_DAY = 300 / 86400


def test_sampling_is_deterministic_and_per_site():
    a, b = sample_site(DIST, 7, 3), sample_site(DIST, 7, 3)
    assert a == b
    assert sample_site(DIST, 7, 4)['draws'] != a['draws']
    assert sample_site(DIST, 8, 3)['draws'] != a['draws']


def test_components_draw_from_independent_streams():
    base = sample_site(DIST, 7, 3)
    fewer = copy.deepcopy(DIST)
    del fewer['site_level']['max_lai']
    del fewer['compartment_level']['leakage_multiplier']
    other = sample_site(fewer, 7, 3)
    for name, value in other['draws'].items():
        assert base['draws'][name] == value
    assert other['sensor_noise_seed'] == base['sensor_noise_seed']


def test_every_draw_is_within_its_declared_range_and_passes_validation():
    specs = {**DIST['site_level'], **{k: v for k, v in DIST['prices_public'].items() if 'distribution' in v}}
    for i in range(300):
        s = sample_site(DIST, 11, i)
        check_site(CONTRACT, s)
        for name, spec in specs.items():
            d = spec['distribution']
            assert d['low'] <= s['draws'][name] <= d['high'], name
        for unit, values in s['site']['unit_parameters'].items():
            assert set(values) <= set(MODEL_PARAMETERS)
            for name, spec in DIST['compartment_level'].items():
                d = spec['distribution']
                assert d['low'] <= s['draws'][f'unit{unit}/{name}'] <= d['high']
        assert 8.0 <= s['soil_boundary_c'] <= 20.0
        assert 0 <= s['sensor_noise_seed'] < 2 ** 63


def test_implied_values_follow_the_rules():
    s = sample_site(DIST, 5, 0)
    d, units, prices = s['draws'], s['site']['unit_parameters'], s['site']['prices']
    shift = d['temperature_window_shift_c']
    for unit in '0123':
        v = units[unit]
        assert v['tCan24Min'] == pytest.approx(15 + shift) and v['tCan24Max'] == pytest.approx(24.5 + shift)
        assert v['tauRfPar'] == v['tauRfNir'] == d['cover_par_nir_transmission']
        assert v['rgFruit'] == pytest.approx(0.328 * d['fruit_growth_potential_factor'] * d[f'unit{unit}/fruit_growth_multiplier'])
        assert v['cLeakage'] == pytest.approx(d['leakage_coefficient'] * d[f'unit{unit}/leakage_multiplier'])
    assert len({units[u]['rgFruit'] for u in '0123'}) == 4
    f = d['energy_price_factor']
    assert prices['electricity_eur_per_kwh'] == pytest.approx(0.15 * f)
    assert prices['delivered_heat_eur_per_kwh'] == pytest.approx(0.05 * f)
    assert 0.10 - 1e-9 <= prices['electricity_eur_per_kwh'] <= 0.30 + 1e-9
    assert prices['co2_eur_per_kg'] == 0.2
    assert s['site']['boundary_ueff_w_m2_k'] == d['side_wall_exchange_w_m2_k']


def test_loguniform_is_uniform_in_log_space():
    import math
    import numpy as np
    rng = np.random.default_rng(0)
    xs = [math.log10(draw({'type': 'loguniform', 'low': 1e-5, 'high': 1e-3}, rng)) for _ in range(4000)]
    assert abs(sum(x < -4 for x in xs) / len(xs) - 0.5) < 0.03
    with pytest.raises(ValueError):
        draw({'type': 'normal'}, rng)


def test_every_parameter_is_labelled_with_evidence():
    for section in ('site_level', 'prices_public', 'compartment_level'):
        for name, spec in DIST[section].items():
            assert spec['evidence'] in ('literature_constrained', 'assumed'), name
            if spec['evidence'] == 'literature_constrained':
                assert spec['sources'], name


def test_sampled_site_runs_through_the_server_and_stays_private(tmp_path):
    s = sample_site(DIST, 3, 1)
    contract = copy.deepcopy(CONTRACT)
    contract['budget'].update(campaign_days=3 * TICK_DAY, crop_days=2 * TICK_DAY, cleanup_days=TICK_DAY,
                              latest_start_day=TICK_DAY)
    (tmp_path / 'c.json').write_text(json.dumps(contract))
    spec = {'backend': 'fake', 'contract': str(tmp_path / 'c.json'), 'feedback_mode': 'full',
            'fallback_policy': POLICIES['policy_a'], 'origin_utc': '2016-12-31T23:00:00+00:00',
            'soil_boundary_c': s['soil_boundary_c'], 'trace': None, 'settlement_out': str(tmp_path / 's.json'),
            'failure_out': str(tmp_path / 'f.json'), 'site': s['site']}
    (tmp_path / 'site.json').write_text(json.dumps(spec))
    with CampaignProcess(tmp_path / 'site.json', private_log=tmp_path / 'log') as campaign:
        task = json.dumps(campaign.session.task)
        assert campaign.session.task['economics']['electricity_eur_per_kwh'] == pytest.approx(
            s['site']['prices']['electricity_eur_per_kwh'])
        for name in MODEL_PARAMETERS:
            assert name not in task
        campaign.session.dispatch({'action': 'advance', 'day': 3 * TICK_DAY})
        campaign.session.dispatch({'action': 'recommend', 'policy': POLICIES['policy_a']})
        campaign.close()
    settlement = json.loads((tmp_path / 's.json').read_text())
    assert settlement['unit_parameters'] == s['site']['unit_parameters']


def test_evaluation_draws_keep_site_values_and_redraw_compartments():
    from slowlab.site_distribution import evaluation_draws
    s = sample_site(DIST, 5, 2)
    a, b = evaluation_draws(DIST, 5, 2, 2003), evaluation_draws(DIST, 5, 2, 2004)
    assert a == evaluation_draws(DIST, 5, 2, 2003) and a['unit_parameters'] != b['unit_parameters']
    for unit, values in a['unit_parameters'].items():
        site = s['site_model_parameters']
        assert values['tauRfPar'] == site['tauRfPar'] and values['j25LeafMax'] == site['j25LeafMax']
        assert 0.95 * site['rgFruit'] <= values['rgFruit'] <= 1.05 * site['rgFruit']
        assert values['rgFruit'] != s['site']['unit_parameters'][unit]['rgFruit']
    assert a['sensor_noise_seed'] not in (b['sensor_noise_seed'], s['sensor_noise_seed'])


def test_weather_years_are_disjoint_deterministic_and_site_specific():
    from slowlab.site_distribution import assign_weather_years
    pool = list(range(2002, 2012)) + [2015]
    a = assign_weather_years(DIST, 9, 0, pool, 3)
    assert a == assign_weather_years(DIST, 9, 0, reversed(pool), 3)
    assert a['campaign_year'] not in a['evaluation_years'] and len(set(a['evaluation_years'])) == 3
    assert set(a['evaluation_years']) | {a['campaign_year']} <= set(pool)
    campaign_years = {assign_weather_years(DIST, 9, i, pool, 3)['campaign_year'] for i in range(40)}
    assert len(campaign_years) >= 8
    with pytest.raises(ValueError):
        assign_weather_years(DIST, 9, 0, [2003, 2004], 2)
