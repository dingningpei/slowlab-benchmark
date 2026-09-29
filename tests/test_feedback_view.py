import json
from pathlib import Path
import pytest
from slowlab.online_observations import OnlineObservations
from slowlab.resources import ResourceLedger
from slowlab.feedback_view import FeedbackView

C=json.loads((Path(__file__).resolve().parents[1]/'configs/task_contract_v3.json').read_text())

def test_full_sees_only_arrived_records_and_endpoint_never_sees_running_science():
    store=OnlineObservations(C['observations']['public_channels'])
    store.record('0','air_temperature_c',measurement_time=0,available_at=0,value=20)
    full=FeedbackView('full',store,['0']);endpoint=FeedbackView('endpoint',store,['0'])
    for view in (full,endpoint):view.executor_set_status('0','active','start')
    assert full.history('0','air_temperature_c')[0]['value']==20
    with pytest.raises(PermissionError):endpoint.history('0','air_temperature_c')
    assert endpoint.final_aggregate('0') is None
    assert endpoint.operational_status('0')=={'unit':'0','phase':'active','clock':0.,'event':'start','run_index':1}
    store.advance_to(300)
    store.record('0','air_temperature_c',measurement_time=300,available_at=400,value=30)
    assert len(full.history('0','air_temperature_c'))==1
    with pytest.raises(ValueError):full.history('0','air_temperature_c',as_of=400)


def test_early_stop_releases_only_accrued_aggregate_not_hidden_trajectory():
    store=OnlineObservations(C['observations']['public_channels'])
    endpoint=FeedbackView('endpoint',store,['0']);ledger=ResourceLedger(C,0)
    ledger.record_event('plant',0)
    endpoint.executor_set_status('0','active','start')
    assert endpoint.final_aggregate('0') is None
    with pytest.raises(ValueError):endpoint.executor_release_final('0','stop',ledger)
    data={'Time':[0,300],'hBoilPipe':[120,120],'qLampIn':[0,0],
          'mcExtAir':[0,0],'mcFruitHar':[0,0],'mvCanAir':[0,0]}
    ledger.add_segment(data,0,300,'active');ledger.record_event('stop',300)
    store.advance_to(300)
    endpoint.executor_set_status('0','cleanup','stop')
    endpoint.executor_release_final('0','stop',ledger)
    payload=endpoint.final_aggregate('0')
    assert payload['reason']=='stop' and payload['accrued']['heat_kwh_m2']==pytest.approx(.01)
    assert payload['event_cost_eur_m2']==pytest.approx(2.5)
    assert 'trajectory' not in payload and 'cFruit' not in str(payload)
    payload['accrued']['heat_kwh_m2']=100
    assert endpoint.final_aggregate('0')['accrued']['heat_kwh_m2']==pytest.approx(.01)
    with pytest.raises(ValueError):endpoint.executor_release_final('0','stop',ledger)
    with pytest.raises(PermissionError):endpoint.history('0','air_temperature_c')


def test_replant_creates_separate_release_identity():
    store=OnlineObservations(C['observations']['public_channels'])
    view=FeedbackView('endpoint',store,['0']);ledger=ResourceLedger(C,0)
    ledger.record_event('plant',0)
    view.executor_set_status('0','active','start')
    data={'Time':[0,300],'hBoilPipe':[0,0],'qLampIn':[0,0],
          'mcExtAir':[0,0],'mcFruitHar':[0,0],'mvCanAir':[0,0]}
    ledger.add_segment(data,0,300,'active');ledger.record_event('stop',300);store.advance_to(300)
    view.executor_set_status('0','cleanup','stop');view.executor_release_final('0','stop',ledger)
    first=view.final_aggregate('0',1)
    assert first['run_index']==1
    view.executor_set_status('0','active','start',ledger=ledger)
    assert view.operational_status('0')['run_index']==2
    data['Time']=[300,600];ledger.record_event('plant',300)
    ledger.add_segment(data,300,600,'active');ledger.record_event('stop',600);store.advance_to(600)
    view.executor_set_status('0','cleanup','stop');view.executor_release_final('0','stop',ledger)
    assert view.final_aggregate('0',1)==first
    assert view.final_aggregate('0',2)['run_index']==2
    assert view.final_aggregate('0',2)['accrued']['days']==pytest.approx(300/86400)
    assert view.final_aggregate('0')['closed_at']==600
