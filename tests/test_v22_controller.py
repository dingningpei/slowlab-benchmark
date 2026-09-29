import json
from pathlib import Path
import pytest
from slowlab.online_observations import OnlineObservations
from slowlab.v22_controller import commands_from_observations,screen_command
R=Path(__file__).resolve().parents[1]
C=json.loads((R/'configs/v22_task_contract_v2.json').read_text())
P=json.loads((R/'configs/v22_campaign_example_v0.json').read_text())['policy_a']
VALUES={'air_temperature_c':20.,'relative_humidity_pct':80.,'outdoor_temperature_c':0.,'co2_ppm':600.,'solar_radiation_w_m2':100.}

def store(time=21600):
    s=OnlineObservations(C['observations']['controller_channels']);s.advance_to(time)
    for k,v in VALUES.items():s.record('a',k,measurement_time=time,available_at=time,value=v)
    return s


def test_delayed_record_cannot_change_command_until_arrival():
    s=store();s.advance_to(21900)
    before=commands_from_observations(C,s,'a',phase='active',policy=P)[0]
    s.record('a','air_temperature_c',measurement_time=21900,available_at=22200,value=35.)
    assert commands_from_observations(C,s,'a',phase='active',policy=P)[0]==before
    s.advance_to(22200)
    assert commands_from_observations(C,s,'a',phase='active',policy=P)[0]!=before


def test_same_available_history_same_control_without_private_state_argument():
    a,b=store(),store()
    assert commands_from_observations(C,a,'a',phase='active',policy=P)==commands_from_observations(C,b,'a',phase='active',policy=P)
    assert a.clock==21600


def test_missing_sensor_rejected_not_hidden_state_fallback():
    s=OnlineObservations(C['observations']['controller_channels']);s.advance_to(21600)
    with pytest.raises(ValueError,match='missing available'):commands_from_observations(C,s,'a',phase='idle')


def test_standby_ignores_previous_crop_policy_and_disables_dosing_and_lighting():
    a=commands_from_observations(C,store(),'a',phase='cleanup',policy=P)[0]
    b=commands_from_observations(C,store(),'a',phase='cleanup',policy={'invalid':'ignored'})[0]
    assert a==b and a['uLamp']==a['uExtCo2']==0
    assert set(a)=={'uBoil','uRoof','uExtCo2','uLamp','uThScr'}


def test_native_screen_limiting_behaviour():
    cold=screen_command(C,18.,70.,0.,23.,20.,87.)
    hot=screen_command(C,35.,70.,0.,23.,20.,87.)
    humid=screen_command(C,23.,100.,0.,23.,20.,87.)
    assert cold>.99 and hot<.01 and humid<cold
    assert screen_command(C,18.,70.,8.,12.,20.,87.) < screen_command(C,18.,70.,8.,23.,20.,87.)


def test_no_future_query_or_unknown_phase():
    with pytest.raises(ValueError):commands_from_observations(C,store(),'a',phase='unknown')
    with pytest.raises(ValueError):commands_from_observations(C,store(),'a',phase='active',policy={})


def test_local_clock_offset_changes_daylight_policy_without_changing_sensor_cutoff():
    import json
    from pathlib import Path
    from slowlab.online_observations import OnlineObservations
    from slowlab.v22_controller import commands_from_observations
    r=Path(__file__).resolve().parents[1]
    c=json.loads((r/'configs/v22_task_contract_v3.json').read_text())
    policy=json.loads((r/'configs/v22_campaign_example_v0.json').read_text())['policy_a']
    obs=OnlineObservations(c['observations']['controller_channels'])
    vals={'air_temperature_c':20.,'relative_humidity_pct':70.,'co2_ppm':400.,
          'outdoor_temperature_c':0.,'solar_radiation_w_m2':0.}
    for name,value in vals.items():obs.record('0',name,measurement_time=0,available_at=0,value=value)
    midnight,_=commands_from_observations(c,obs,'0',phase='active',policy=policy,local_clock_offset_seconds=0)
    morning,decision=commands_from_observations(c,obs,'0',phase='active',policy=policy,local_clock_offset_seconds=6*3600)
    assert midnight['uLamp']==0 and morning['uLamp']==1
    assert morning['uExtCo2']>0 and midnight['uExtCo2']==0
    assert all(v['measurement_time']==0 for v in decision['sensor_records'].values())
