import copy
import json
from pathlib import Path
import pytest
from slowlab.v22.task_contract import dry_run, validate_contract, validate_policy

ROOT = Path(__file__).resolve().parents[2]

@pytest.fixture
def inputs():
    return (json.loads((ROOT/'configs/v22/task_contract_v0.json').read_text()),
            json.loads((ROOT/'configs/v22/campaign_example_v0.json').read_text()))


def test_example_charges_stopped_crops_and_cleanup(inputs):
    c, example = inputs
    result = dry_run(c, example)
    assert (result['starts'], result['early_stops'], result['normal_completions']) == (4,1,3)
    assert result['active_compartment_days'] == 554
    assert result['cleanup_compartment_days'] == 8
    assert result['occupied_m2_days'] == 53952
    assert result['known_fixed_cost_eur'] == pytest.approx(1491.84)
    assert result['harvest_energy_and_margin'] == 'not evaluated'


@pytest.mark.parametrize('day', [14, 15.99])
def test_cannot_replant_during_cleaning(inputs, day):
    c,e=inputs
    e['actions'][5]['day']=day
    with pytest.raises(ValueError, match='cleaning'):
        dry_run(c,e)


def test_future_observation_and_clock_rewind_rejected(inputs):
    c,e=inputs
    e['actions'][3]['as_of_day']=15
    with pytest.raises(ValueError, match='future'):
        dry_run(c,e)
    del e['actions'][3]['as_of_day']
    e['actions'][5]['day']=13
    with pytest.raises(ValueError, match='clock'):
        dry_run(c,e)


@pytest.mark.parametrize('bad', [float('nan'), 1800, True])
def test_invalid_policy_value_rejected(inputs,bad):
    c,e=inputs
    e['policy_a']['co2_target_ppm']=bad
    with pytest.raises(ValueError):
        dry_run(c,e)


def test_hidden_extra_field_and_running_edit_rejected(inputs):
    c,e=inputs
    e['policy_a']['truth_seed']=42
    with pytest.raises(ValueError, match='exactly'):
        validate_policy(c,e['policy_a'])
    del e['policy_a']['truth_seed']
    e['actions'][3]['action']='modify_running_policy'
    with pytest.raises(ValueError,match='unsupported'):
        dry_run(c,e)


def test_geometry_and_capacity_inconsistency_rejected(inputs):
    c,_=inputs
    broken=copy.deepcopy(c);broken['facility']['capacity']['co2_mg_s']=7.5
    with pytest.raises(ValueError,match='CO2'):
        validate_contract(broken)
    c['facility']['geometry']['reference_cover_area_m2']=96
    with pytest.raises(ValueError,match='cover'):
        validate_contract(c)


def test_start_deadline_and_initial_budget_enforced(inputs):
    c,e=inputs
    e['actions'][9]['day']=184
    with pytest.raises(ValueError,match='deadline'):
        dry_run(c,e)
    e['actions'][9]['day']=182
    c['budget']['max_starts']=3
    with pytest.raises(ValueError,match='budget'):
        dry_run(c,e)


def test_v3_central_adequacy_is_validated():
    c=json.loads((ROOT/'configs/v22/task_contract_v3.json').read_text())
    validate_contract(c)
    for key in ('heating_w_min','co2_mg_s_min','lamp_electric_w_min'):
        broken=copy.deepcopy(c);broken['facility']['central_supply'][key]=0
        with pytest.raises(ValueError,match='central supply'):
            validate_contract(broken)
