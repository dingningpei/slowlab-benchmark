#!/usr/bin/env python3
"""Bounded Phase 1 check: lc1 parsing plus six dynamic-weather GreenLight steps."""
import argparse,json,math,sys
from datetime import datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from slowlab.v22.cabauw_weather import CabauwLc1Weather
from slowlab.v22.greenlight_reuse import ReusableGreenLight

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=Path('/private/tmp/slowlab-v22-cabauw-gapfilled'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.source))
    weather=CabauwLc1Weather(args.cache,ROOT/'configs/v22/weather_gapfilled_plan.json')
    checked=0
    for year in range(2017,2021):
        for month in range(1,13):
            first=datetime(year,month,1,tzinfo=timezone.utc)
            a=weather.at_utc(first)
            b=weather.at_utc(first+timedelta(minutes=5))
            assert a==b and a.interval_start_utc==first
            checked+=1
    weather.at_utc(datetime(2016,12,31,23,55,tzinfo=timezone.utc));checked+=1
    try:weather.at_utc(datetime(2021,1,1,tzinfo=timezone.utc))
    except ValueError:pass
    else:raise AssertionError('outside-weather-scope query accepted')
    contract=json.loads((ROOT/'configs/v22/task_contract_v2.json').read_text())
    origin=datetime(2017,1,1,tzinfo=timezone.utc)
    engine=ReusableGreenLight(contract,args.source,start=28800,cached_solver=True,
                               weather=weather,weather_origin_utc=origin)
    wanted=set(weather.at_utc(origin+timedelta(seconds=28800)).greenlight_inputs())
    assert wanted.issubset(engine.model.inputs)
    steps=[]
    for _ in range(6):
        start=engine.clock
        state=engine.step({key:0.0 for key in engine.COMMANDS},start+300)
        expected=weather.at_utc(origin+timedelta(seconds=start)).greenlight_inputs()
        row=engine.model.input_data.iloc[0]
        assert len(engine.model.input_data)==1 and float(row['Time'])==start
        assert all(math.isclose(float(row[key]),value,abs_tol=1e-9) for key,value in expected.items())
        assert all(math.isfinite(value) for value in state.values())
        steps.append({'start_seconds':start,'iGlob':float(row['iGlob']),
                      'tOut':float(row['tOut']),'tAir_endpoint':state['tAir'],
                      'input_rows':len(engine.model.input_data)})
    assert len({step['iGlob'] for step in steps})>1
    result={'status':'passed_bounded_dynamic_weather_smoke','scope':'49 month boundary queries; six 300s GreenLight steps with lc1 weather; not full campaign or agent-visible causality validation',
            'months_checked':checked,'registered_inputs':sorted(engine.model.inputs),
            'steps':steps,'model_load_count':engine.load_count}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'months_checked':checked,'steps':len(steps),
                      'iGlob_range':[min(s['iGlob'] for s in steps),max(s['iGlob'] for s in steps)]}))
if __name__=='__main__':main()
