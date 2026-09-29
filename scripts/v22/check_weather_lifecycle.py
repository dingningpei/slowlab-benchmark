#!/usr/bin/env python3
"""One-unit two-day cleanup/replant with archived weather, causal sensors, ledger."""
from __future__ import annotations
import argparse,json,math,resource,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from slowlab.v22.online_observations import OnlineObservations
from slowlab.v22.feedback_view import FeedbackView
from slowlab.v22.cabauw_weather import CabauwLc1Weather
from slowlab.v22.controller import commands_from_observations
from slowlab.v22.greenlight_reuse import CropLifecycle,CROP_STATES
from slowlab.v22.resources import ResourceLedger
from check_online_weather import record_at_endpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('/private/tmp/slowlab-v22-cabauw-gapfilled'))
    p.add_argument('--native-rhs',action='store_true')
    p.add_argument('--array-output',action='store_true')
    p.add_argument('--profile-steps',type=Path,default=None)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.source))
    c=json.loads((ROOT/'configs/v22/task_contract_v3.json').read_text())
    policy=json.loads((ROOT/'configs/v22/campaign_example_v0.json').read_text())['policy_a']
    origin=datetime(2016,12,31,23,tzinfo=timezone.utc)
    weather=CabauwLc1Weather(a.cache,ROOT/'configs/v22/weather_gapfilled_plan.json')
    started=time.monotonic();result={'status':'running','scope':'one unit; two-day weather-driven cleanup/replant; 90s compute bound','native_rhs':a.native_rhs,'array_output':a.array_output}
    ledger=None
    try:
        life=CropLifecycle(c,a.source,start=0,cached_solver=True,native_rhs=a.native_rhs,weather=weather,weather_origin_utc=origin,
                           soil_boundary_c=20.,local_clock_offset_seconds=0,array_output=a.array_output)
        model_load_elapsed=time.monotonic()-started
        ledger=ResourceLedger(c,0);ledger.record_event('plant',0)
        ctl=OnlineObservations(c['observations']['controller_channels'])
        pub=OnlineObservations(c['observations']['public_channels'])
        full=FeedbackView('full',pub,['0']);endpoint=FeedbackView('endpoint',pub,['0'])
        for view in (full,endpoint):view.executor_set_status('0','active','start',ledger=ledger)
        record_at_endpoint(life.engine,weather,origin,ctl,pub,'0',ledger)
        profiler=None
        if a.profile_steps is not None:
            import cProfile
            profiler=cProfile.Profile();profiler.enable()
        def step(phase):
            if time.monotonic()-started>90:raise TimeoutError('90-second bounded compute gate')
            now=life.engine.clock
            cmd,decision=commands_from_observations(c,ctl,'0',phase=phase,policy=policy if phase=='active' else None)
            assert all(rec['available_at']<=now for rec in decision['sensor_records'].values())
            state=life.step(cmd,now+300)
            if not all(math.isfinite(v) for v in state.values()):raise AssertionError('nonfinite physical state')
            ledger.add_segment(life.engine.model.full_sol,now,now+300,phase)
            record_at_endpoint(life.engine,weather,origin,ctl,pub,'0',ledger)
            assert len(life.engine.model.input_data)==1
        step('active');before=dict(life.engine.state);closed_at=life.engine.clock
        life.stop();ledger.record_event('stop',closed_at)
        assert all(life.engine.state[k]==v for k,v in before.items() if k not in CROP_STATES)
        for view in (full,endpoint):view.executor_set_status('0','cleanup','stop')
        endpoint.executor_release_final('0','stop',ledger)
        assert endpoint.final_aggregate('0')['closed_at']==closed_at
        assert endpoint.final_aggregate('0')['accrued']['days']==300/86400
        try:endpoint.history('0','air_temperature_c')
        except PermissionError:pass
        else:raise AssertionError('endpoint timeline leak')
        try:life.replant()
        except ValueError:pass
        else:raise AssertionError('premature replant accepted')
        cleanup_started=time.monotonic();cleanup_steps=0
        while life.engine.clock<life.ready_at:
            step('cleanup');cleanup_steps+=1
        cleanup_elapsed=time.monotonic()-cleanup_started
        assert cleanup_steps==576
        assert abs(ledger.totals['cleanup']['days']-2)<1e-10
        after_cleanup=dict(life.engine.state);replant_at=life.engine.clock
        life.replant();ledger.record_event('plant',replant_at)
        assert all(life.engine.state[k]==v for k,v in after_cleanup.items() if k not in CROP_STATES)
        for view in (full,endpoint):view.executor_set_status('0','active','start',ledger=ledger)
        step('active')
        assert len(full.history('0','air_temperature_c'))==579
        if profiler is not None:
            profiler.disable();profiler.dump_stats(str(a.profile_steps))
        result.update(status='passed_weather_lifecycle',active_steps=2,cleanup_steps=cleanup_steps,
                      stop_at=closed_at,replant_at=replant_at,facility_state_continuity=True,
                      endpoint_early_stop_release='accrued_only',public_air_records=len(full.history('0','air_temperature_c')),
                      model_load_count=sum(e.load_count for e in life.engines.values()),
                      model_load_elapsed_seconds=model_load_elapsed,cleanup_elapsed_seconds=cleanup_elapsed,
                      final_state=dict(life.engine.state))
    except Exception as exc:
        traceback.print_exc();result.update(status='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        result['elapsed_seconds']=time.monotonic()-started
        result['peak_rss_bytes_platform']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if ledger:result['ledger']=ledger.summary()
        a.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ('status','elapsed_seconds','peak_rss_bytes_platform',*(['error'] if 'error' in result else []))}),flush=True)
    if result['status']!='passed_weather_lifecycle':sys.exit(1)
if __name__=='__main__':main()
