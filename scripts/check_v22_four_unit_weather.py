#!/usr/bin/env python3
"""Bounded four-unit same-clock weather/command/observation/ledger gate."""
from __future__ import annotations
import argparse,json,math,resource,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from slowlab.online_observations import OnlineObservations
from slowlab.v22_feedback_view import FeedbackView
from slowlab.task_contract_v22 import validate_contract
from slowlab.v22_cabauw_weather import CabauwLc1Weather
from slowlab.v22_controller import commands_from_observations
from slowlab.v22_greenlight_reuse import ReusableGreenLight
from slowlab.v22_resources import ResourceLedger,realise_independent_commands
from check_v22_online_weather import record_at_endpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('/private/tmp/slowlab-v22-cabauw-gapfilled'))
    p.add_argument('--origin-utc',default='2016-12-31T23:00:00+00:00')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.source))
    c=json.loads((ROOT/'configs/v22_task_contract_v3.json').read_text());validate_contract(c)
    policies=json.loads((ROOT/'configs/v22_campaign_example_v0.json').read_text())
    origin=datetime.fromisoformat(a.origin_utc)
    if origin.tzinfo is None or origin.utcoffset()!=timezone.utc.utcoffset(origin):
        raise ValueError('origin must be timezone-aware UTC')
    weather=CabauwLc1Weather(a.cache,ROOT/'configs/v22_weather_gapfilled_plan.json')
    local_time=origin+timedelta(hours=1)  # fixed CET (UTC+1), no DST
    local_offset=local_time.hour*3600+local_time.minute*60+local_time.second
    controller=OnlineObservations(c['observations']['controller_channels'])
    public=OnlineObservations(c['observations']['public_channels'])
    started=time.monotonic();ids=tuple(str(i) for i in range(c['facility']['compartments']))
    full=FeedbackView('full',public,ids);endpoint=FeedbackView('endpoint',public,ids)
    engines={unit:ReusableGreenLight(c,a.source,start=0,cached_solver=True,weather=weather,
                                      weather_origin_utc=origin,soil_boundary_c=20.,local_clock_offset_seconds=local_offset) for unit in ids}
    ledgers={unit:ResourceLedger(c,0) for unit in ids}
    for unit in ids:
        ledgers[unit].record_event('plant',0)
        full.executor_set_status(unit,'active','start',ledger=ledgers[unit])
        endpoint.executor_set_status(unit,'active','start',ledger=ledgers[unit])
        record_at_endpoint(engines[unit],weather,origin,controller,public,unit,ledgers[unit])
    trace=[];max_flux={'heat_w_m2':0.,'co2_mg_m2_s':0.,'lamp_w_m2':0.}
    for tick in range(6):
        now=tick*300
        if any(e.clock!=now for e in engines.values()):raise AssertionError('unsynchronized physical clock')
        requested={};decisions={}
        for unit in ids:
            policy=policies['policy_a' if int(unit)%2==0 else 'policy_b']
            requested[unit],decisions[unit]=commands_from_observations(c,controller,unit,phase='active',policy=policy,local_clock_offset_seconds=local_offset)
        realised=realise_independent_commands(c,requested)
        assert realised==requested
        events=[]
        for unit in ids:
            e=engines[unit];state=e.step(realised[unit],now+300);df=e.model.full_sol
            if not all(math.isfinite(x) for x in state.values()):raise AssertionError('nonfinite state')
            ledgers[unit].add_segment(df,now,now+300,'active')
            rates={'heat_w_m2':float(df['hBoilPipe'].max()),
                   'co2_mg_m2_s':float(df['mcExtAir'].max()),
                   'lamp_w_m2':float(df['qLampIn'].max())}
            for name,v in rates.items():max_flux[name]=max(max_flux[name],v)
            assert rates['heat_w_m2']<=c['facility']['capacity']['heating_w_m2']+1e-8
            assert rates['co2_mg_m2_s']<=c['facility']['capacity']['co2_mg_s']/c['facility']['floor_area_m2']+1e-8
            assert rates['lamp_w_m2']<=c['facility']['capacity']['lamp_electric_w_m2']+1e-8
            assert len(e.model.input_data)==1
            events.append({'unit':unit,'requested':requested[unit],'realised':realised[unit],
                           'forcing_time':now,'sensor_max_time':max(record['available_at'] for record in decisions[unit]['sensor_records'].values()),
                           'temperature_c':state['tAir'],'rates':rates})
        # No compartment may publish its new endpoint before all reach it.
        for unit in ids:
            record_at_endpoint(engines[unit],weather,origin,controller,public,unit,ledgers[unit])
        assert controller.clock==public.clock==now+300
        trace.append({'start':now,'end':now+300,'compartments':events})
    permissions={}
    for unit in ids:
        assert len(full.history(unit,'air_temperature_c'))==7
        try:endpoint.history(unit,'air_temperature_c')
        except PermissionError:pass
        else:raise AssertionError('endpoint read running scientific records')
        assert endpoint.final_aggregate(unit) is None
        permissions[unit]={'full_records':len(full.history(unit,'air_temperature_c')),'endpoint_running_science':'denied','endpoint_final_aggregate':'unreleased'}
    result={'status':'passed_bounded_four_unit_bridge','contract_id':c['contract_id'],
            'scope':'four active compartments, same clock, six 300s steps; not full campaign or annual validation',
            'origin_utc':origin.isoformat(),'local_clock_offset_seconds':local_offset,'central_supply_model':c['facility']['central_supply']['model'],
            'steps':trace,'max_realised_flux':max_flux,
            'ledgers':{unit:ledger.summary() for unit,ledger in ledgers.items()},
            'public_indoor_record_counts':{unit:len(public.history(unit,'air_temperature_c')) for unit in ids},
            'feedback_permissions':permissions,'model_load_count':sum(e.load_count for e in engines.values()),
            'elapsed_seconds':time.monotonic()-started,'peak_rss_bytes_platform':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    assert all(n==7 for n in result['public_indoor_record_counts'].values())
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','contract_id','max_realised_flux','model_load_count','elapsed_seconds','peak_rss_bytes_platform')}))
if __name__=='__main__':main()
