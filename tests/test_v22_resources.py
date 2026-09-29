import json
from pathlib import Path
import pytest
from slowlab.v22_resources import ResourceLedger,standby_commands
C=json.loads((Path(__file__).resolve().parents[1]/'configs/v22_task_contract_v1.json').read_text())

@pytest.mark.parametrize('t',[-10,0,10,20,30,40,60])
def test_standby_disables_crop_inputs_and_bounds_commands(t):
    u=standby_commands(C,t)
    assert u['uLamp']==u['uExtCo2']==0
    assert all(0<=v<=1 for v in u.values())
    if t<=0:assert u['uBoil']>.99 and u['uRoof']<.01
    if t>=40:assert u['uRoof']>.99 and u['uBoil']<.01

@pytest.mark.parametrize('t',[None,True,float('nan'),float('inf')])
def test_missing_temperature_cannot_fall_back_to_hidden_state(t):
    with pytest.raises(ValueError):standby_commands(C,t)

def segment(start=0,end=300,harvest=0):
    return {'Time':[start,end],'hBoilPipe':[120,120],'qLampIn':[100,100],'mcExtAir':[2,2],'mcFruitHar':[harvest,harvest],'mvCanAir':[0,0]}

def test_units_events_and_standby_costs():
    l=ResourceLedger(C,0);l.record_event('plant',0)
    l.add_segment(segment(harvest=1),0,300,'active');l.record_event('stop',300)
    l.add_segment(segment(300,600),300,600,'cleanup')
    s=l.summary()
    assert s['per_m2']['heat_kwh_m2']==pytest.approx(.02)
    assert s['per_m2']['co2_kg_m2']==pytest.approx(.0012)
    assert s['per_m2']['harvest_kg_m2']==pytest.approx(.005)
    assert s['event_cost_eur_m2']==2.5
    assert s['by_phase_per_m2']['cleanup']['heat_kwh_m2']==pytest.approx(.01)
    assert s['occupied_m2_days']==pytest.approx(96*600/86400)
    with pytest.raises(ValueError):l.add_segment(segment(300,600),300,600,'cleanup')
    with pytest.raises(ValueError):l.record_event('stop',600)

def test_failed_segment_does_not_mutate_ledger():
    l=ResourceLedger(C,0);before=l.summary()
    with pytest.raises(ValueError):l.add_segment(segment(harvest=1),0,300,'idle')
    assert l.summary()==before and l.clock==0
    data=segment();data['hBoilPipe']=[float('nan'),0]
    with pytest.raises(ValueError):l.add_segment(data,0,300,'active')
    assert l.summary()==before

def test_invalid_time_samples_rejected():
    l=ResourceLedger(C,0);data=segment();data['Time']=[0,0]
    with pytest.raises(ValueError):l.add_segment(data,0,300,'cleanup')


def test_four_unit_independent_realisation_and_validation():
    import copy,json
    from pathlib import Path
    from slowlab.v22_resources import realise_independent_commands
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/v22_task_contract_v3.json').read_text())
    commands={str(i):{'uBoil':1.,'uRoof':0.,'uExtCo2':1.,'uLamp':1.,'uThScr':0.} for i in range(4)}
    realised=realise_independent_commands(c,commands)
    assert realised==commands and realised is not commands
    commands['0']['uBoil']=0.
    assert realised['0']['uBoil']==1.
    with pytest.raises(ValueError,match='every'):
        realise_independent_commands(c,{k:v for k,v in commands.items() if k!='3'})
    bad=copy.deepcopy(commands);bad['2']['uLamp']=float('nan')
    with pytest.raises(ValueError,match='outside'):
        realise_independent_commands(c,bad)
