import json
from pathlib import Path
from slowlab.v22.greenlight_smoke import held_commands


def policy():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/v22/campaign_example_v0.json').read_text())['policy_a']


def test_control_is_capacity_bounded_and_lamps_follow_causal_window():
    p=policy(); observed={'temperature':20,'rh':80,'co2':500,'solar':100}
    day=held_commands(p,observed,7)
    night=held_commands(p,{**observed,'solar':0},23)
    assert all(0 <= value <= 1 for value in day.values())
    assert day['uLamp']==1 and night['uLamp']==0 and night['uExtCo2']==0
    assert held_commands(p,{**observed,'solar':500},7)['uLamp']==0
    assert held_commands(p,{**observed,'temperature':36},7)['uLamp']==0


def test_heat_and_ventilation_respond_in_expected_directions():
    p=policy(); obs={'temperature':16,'rh':80,'co2':500,'solar':100}
    cold=held_commands(p,obs,7); hot=held_commands(p,{**obs,'temperature':30},7)
    assert cold['uBoil'] > hot['uBoil']
    assert cold['uRoof'] < hot['uRoof']
    assert held_commands(p,{**obs,'co2':1200},7)['uExtCo2'] < cold['uExtCo2']


def test_unused_future_weather_cannot_change_commands():
    p=policy(); prefix={'temperature':20,'rh':80,'co2':500,'solar':100}
    assert held_commands(p,{**prefix,'future_daily_solar':1},7)==held_commands(p,{**prefix,'future_daily_solar':1000},7)
