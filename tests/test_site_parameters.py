import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from slowlab.agent_client import CampaignProcess
from slowlab.agent_protocol import public_task_view
from slowlab.campaign_executor import CampaignExecutor
from slowlab.site_parameters import MODEL_PARAMETERS, site_contract, validate_model_parameters

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'configs/task_contract_v5.json').read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
TICK_DAY = 300 / 86400


def test_model_parameters_are_validated():
    assert validate_model_parameters({'j25LeafMax': 180, 'tauRfPar': 0.8}) == {'j25LeafMax': 180.0, 'tauRfPar': 0.8}
    for bad in ({'sla': 3e-5}, {'tauRfPar': 1.2}, {'rgFruit': float('nan')}, {'cLeakage': True},
                {'tCan24Min': 20, 'tCan24Max': 19}):
        with pytest.raises(ValueError):
            validate_model_parameters(bad)
    assert validate_model_parameters(None) == {}


def test_site_contract_sets_public_prices_and_side_wall_exchange():
    site = {'prices': {'electricity_eur_per_kwh': 0.27, 'price_eur_per_kg_fresh_equivalent': 1.8},
            'boundary_ueff_w_m2_k': 1.5}
    c = site_contract(CONTRACT, site)
    assert c['economics']['electricity_eur_per_kwh'] == 0.27 and c['facility']['boundary']['ueff_w_m2_k'] == 1.5
    assert CONTRACT['economics']['electricity_eur_per_kwh'] == 0.15
    view = public_task_view(c, 'full')
    assert view['economics']['electricity_eur_per_kwh'] == 0.27
    assert view['economics']['fruit_price_eur_per_kg_fresh'] == 1.8
    assert 'ueff' not in json.dumps(view).lower()
    for bad in ({'prices': {'rent': 1}}, {'weather_year': 2015}, {'boundary_ueff_w_m2_k': -1}):
        with pytest.raises(ValueError):
            site_contract(CONTRACT, bad)


class Capture:
    seen = {}

    def __init__(self, contract, source, start, **kwargs):
        Capture.seen[len(Capture.seen)] = kwargs.get('parameter_overrides')
        self.clock, self.mode, self.ready_at = 0.0, 'empty', 0.0
        self.model = SimpleNamespace(full_sol={}, input_data=[{}])

    @property
    def engine(self):
        return self


def test_executor_gives_each_compartment_its_own_parameters():
    Capture.seen = {}
    units = {'0': {'rgFruit': 0.30}, '2': {'rgFruit': 0.36, 'cLeakage': 2e-4}}
    x = CampaignExecutor(CONTRACT, Path('/unused'), None, feedback_mode='full', fallback_policy=POLICIES['policy_a'],
                         lifecycle_factory=Capture, sample_endpoint=lambda *a: None, unit_parameters=units)
    assert Capture.seen == {0: {'rgFruit': 0.30}, 1: None, 2: {'rgFruit': 0.36, 'cLeakage': 2e-4}, 3: None}
    assert x.unit_parameters['1'] == {}
    with pytest.raises(ValueError, match='unknown compartment'):
        CampaignExecutor(CONTRACT, Path('/unused'), None, feedback_mode='full', fallback_policy=POLICIES['policy_a'],
                         lifecycle_factory=Capture, sample_endpoint=lambda *a: None, unit_parameters={'7': {}})


def test_server_applies_site_prices_publicly_and_records_parameters_privately(tmp_path):
    contract = copy.deepcopy(CONTRACT)
    contract['budget'].update(campaign_days=3 * TICK_DAY, crop_days=2 * TICK_DAY, cleanup_days=TICK_DAY,
                              latest_start_day=TICK_DAY)
    (tmp_path / 'c.json').write_text(json.dumps(contract))
    site = {'prices': {'electricity_eur_per_kwh': 0.3}, 'unit_parameters': {'1': {'laiMax': 3.4}}}
    spec = {'backend': 'fake', 'contract': str(tmp_path / 'c.json'), 'feedback_mode': 'full',
            'fallback_policy': POLICIES['policy_a'], 'origin_utc': '2016-12-31T23:00:00+00:00', 'soil_boundary_c': None,
            'trace': None, 'settlement_out': str(tmp_path / 's.json'), 'failure_out': str(tmp_path / 'f.json'),
            'site': site}
    (tmp_path / 'site.json').write_text(json.dumps(spec))
    with CampaignProcess(tmp_path / 'site.json', private_log=tmp_path / 'log') as campaign:
        assert campaign.session.task['economics']['electricity_eur_per_kwh'] == 0.3
        assert 'laiMax' not in json.dumps(campaign.session.task)
        campaign.session.dispatch({'action': 'advance', 'day': 3 * TICK_DAY})
        campaign.session.dispatch({'action': 'recommend', 'policy': POLICIES['policy_a']})
        assert 'laiMax' not in json.dumps(campaign.session.transcript)
        campaign.close()
    settlement = json.loads((tmp_path / 's.json').read_text())
    assert settlement['unit_parameters']['1'] == {'laiMax': 3.4} and settlement['unit_parameters']['0'] == {}


def test_allow_list_is_what_the_site_distribution_may_touch():
    assert set(MODEL_PARAMETERS) == {'j25LeafMax', 'rgFruit', 'tEndSum', 'laiMax', 'tCan24Min', 'tCan24Max',
                                     'tauRfPar', 'tauRfNir', 'cLeakage', 'cDgh', 'cWgh', 'etaLampPar'}
