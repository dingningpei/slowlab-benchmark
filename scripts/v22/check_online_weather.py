#!/usr/bin/env python3
"""Bounded V2.2 weather→controller→executor→sensor check, not a campaign."""
from __future__ import annotations
import argparse,json,math,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from slowlab.v22.online_observations import OnlineObservations
from slowlab.v22.cabauw_weather import CabauwLc1Weather
from slowlab.v22.controller import commands_from_observations
from slowlab.v22.greenlight_reuse import ReusableGreenLight
from slowlab.v22.resources import ResourceLedger
from slowlab.v22.public_sensors import public_endpoint_measurements

from slowlab.v22.sensor_bridge import indoor, record_at_endpoint  # noqa: E402,F401

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--soil-boundary-c',type=float,default=None,help='Explicit deep-soil scenario (°C); default uses GreenLight native 20°C')
    parser.add_argument('--native-rhs',action='store_true')
    parser.add_argument('--array-output',action='store_true')
    parser.add_argument('--cache',type=Path,default=Path('/private/tmp/slowlab-v22-cabauw-gapfilled'))
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    sys.path.insert(0,str(args.source))
    contract=json.loads((ROOT/'configs/v22/task_contract_v2.json').read_text())
    policy=json.loads((ROOT/'configs/v22/campaign_example_v0.json').read_text())['policy_a']
    origin=datetime(2016,12,31,23,tzinfo=timezone.utc) # 2017-01-01 00:00 fixed CET, no DST
    weather=CabauwLc1Weather(args.cache,ROOT/'configs/v22/weather_gapfilled_plan.json')
    controller=OnlineObservations(contract['observations']['controller_channels'])
    public=OnlineObservations(contract['observations']['public_channels'])
    started=time.monotonic()
    engine=ReusableGreenLight(contract,args.source,start=0,cached_solver=True,
                               native_rhs=args.native_rhs,array_output=args.array_output,
                               weather=weather,weather_origin_utc=origin,soil_boundary_c=args.soil_boundary_c)
    ledger=ResourceLedger(contract,0);ledger.record_event('plant',0)
    initial,_=record_at_endpoint(engine,weather,origin,controller,public,'a',ledger)
    ticks=[]
    for tick in range(6):
        start=engine.clock
        requested,decision=commands_from_observations(contract,controller,'a',phase='active',policy=policy)
        used={name:decision['sensor_records'][name]['measurement_time'] for name in decision['sensor_records']}
        assert all(t<=start for t in used.values())
        if tick % 2:
            assert used['outdoor_temperature_c']==start-300
            assert used['solar_radiation_w_m2']==start-300
        else:
            assert used['outdoor_temperature_c']==start
            assert used['solar_radiation_w_m2']==start
        state=engine.step(requested,start+300)
        df=engine.model.full_sol
        if not all(math.isfinite(v) for v in state.values()):raise ValueError('nonfinite model state')
        ledger.add_segment(df,start,engine.clock,'active')
        inside,outside=record_at_endpoint(engine,weather,origin,controller,public,'a',ledger)
        for name in inside:
            assert public.latest('a',name)['measurement_time']==engine.clock
        if outside is None:
            assert controller.latest('a','solar_radiation_w_m2')['measurement_time']==start
        else:
            assert controller.latest('a','solar_radiation_w_m2')['measurement_time']==engine.clock
        ticks.append({'start':start,'end':engine.clock,'requested':requested,
                      'realised':dict(requested),'controller_sensor_times':used,
                      'endpoint_indoor':inside,'new_outdoor_sample':outside,
                      'model_input_rows':len(engine.model.input_data)})
        assert len(engine.model.input_data)==1
    try:public.history('a','solar_radiation_w_m2')
    except ValueError:pass
    else:raise AssertionError('controller-only solar channel exposed through public store')
    assert len(public.history('a','air_temperature_c'))==7
    assert len(controller.history('a','solar_radiation_w_m2'))==4
    result={'status':'passed_bounded_online_weather_bridge','scope':'one active compartment; six 300s steps, deterministic sensors, no Full/Endpoint agent API or annual campaign',
            'soil_boundary_c':20.0 if args.soil_boundary_c is None else args.soil_boundary_c,
            'origin_utc':origin.isoformat(),'local_standard_timezone':'CET UTC+1 fixed, no DST',
            'first_measurement_time':0,'steps':ticks,'ledger':ledger.summary(),
            'public_air_records':len(public.history('a','air_temperature_c')),
            'controller_solar_records':len(controller.history('a','solar_radiation_w_m2')),
            'model_load_count':engine.load_count,'final_state':dict(engine.state),
            'native_rhs':args.native_rhs,'array_output':args.array_output,
            'elapsed_seconds':time.monotonic()-started}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','first_measurement_time','public_air_records','controller_solar_records','model_load_count','elapsed_seconds')}))
if __name__=='__main__':main()
