#!/usr/bin/env python3
"""Bounded early active-crop weather window for Phase-1 runtime evidence."""
from __future__ import annotations
import argparse,json,math,resource,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from slowlab.v22.online_observations import OnlineObservations
from slowlab.v22.cabauw_weather import CabauwLc1Weather
from slowlab.v22.controller import commands_from_observations
from slowlab.v22.greenlight_reuse import ReusableGreenLight
from slowlab.v22.resources import ResourceLedger
from check_online_weather import record_at_endpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('/private/tmp/slowlab-v22-cabauw-gapfilled'))
    p.add_argument('--array-output',action='store_true');p.add_argument('--days',type=int,default=7)
    p.add_argument('--action-trace-out',type=Path,default=None,
                   help='Optional per-step requested command trace for numerical comparison')
    p.add_argument('--replay-actions',type=Path,default=None,
                   help='Diagnostic only: replace controller output with a fixed command trace')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not 1<=a.days<=7:raise ValueError('bounded check allows 1–7 days')
    replay=json.loads(a.replay_actions.read_text()) if a.replay_actions is not None else None
    if replay is not None and len(replay)!=a.days*288:
        raise ValueError('replay command count must equal bounded step count')
    sys.path.insert(0,str(a.source))
    c=json.loads((ROOT/'configs/v22/task_contract_v3.json').read_text())
    policy=json.loads((ROOT/'configs/v22/campaign_example_v0.json').read_text())['policy_a']
    origin=datetime(2016,12,31,23,tzinfo=timezone.utc)
    weather=CabauwLc1Weather(a.cache,ROOT/'configs/v22/weather_gapfilled_plan.json')
    started=time.monotonic();out={'status':'running','scope':'one early active crop; archived Jan 2017 weather; bounded 1–7 days','array_output':a.array_output,'days':a.days}
    ledger=None
    action_trace=[] if a.action_trace_out is not None else None
    try:
        engine=ReusableGreenLight(c,a.source,start=0,cached_solver=True,native_rhs=True,
                                   weather=weather,weather_origin_utc=origin,soil_boundary_c=20.,
                                   array_output=a.array_output)
        load_seconds=time.monotonic()-started
        ledger=ResourceLedger(c,0);ledger.record_event('plant',0)
        ctl=OnlineObservations(c['observations']['controller_channels'])
        pub=OnlineObservations(c['observations']['public_channels'])
        record_at_endpoint(engine,weather,origin,ctl,pub,'0',ledger)
        stepping_started=time.monotonic()
        for tick in range(a.days*288):
            if time.monotonic()-stepping_started>60:raise TimeoutError('60s stepping bound')
            now=engine.clock
            command,event=commands_from_observations(c,ctl,'0',phase='active',policy=policy)
            if replay is not None:
                if set(replay[tick])!=set(command):raise ValueError('replay command channels differ')
                command={k:float(v) for k,v in replay[tick].items()}
            if action_trace is not None:action_trace.append(dict(command))
            if any(rec['available_at']>now for rec in event['sensor_records'].values()):
                raise AssertionError('controller used unarrived sensor')
            state=engine.step(command,now+300)
            if any(not math.isfinite(v) for v in state.values()):raise AssertionError('nonfinite state')
            ledger.add_segment(engine.model.full_sol,now,now+300,'active')
            record_at_endpoint(engine,weather,origin,ctl,pub,'0',ledger)
        out.update(status='passed_active_window',model_load_seconds=load_seconds,
                   stepping_seconds=time.monotonic()-stepping_started,steps=a.days*288,
                   final_state=dict(engine.state),ledger=ledger.summary(),
                   public_air_records=len(pub.history('0','air_temperature_c')))
    except Exception as exc:
        traceback.print_exc();out.update(status='failed',error=type(exc).__name__+': '+str(exc))
    finally:
        if action_trace is not None:
            a.action_trace_out.write_text(json.dumps(action_trace)+'\n')
        out['elapsed_seconds']=time.monotonic()-started
        out['peak_rss_bytes_platform']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        a.out.write_text(json.dumps(out,indent=2)+'\n')
        print(json.dumps({k:out[k] for k in ('status','array_output','elapsed_seconds','peak_rss_bytes_platform',*(['error'] if 'error' in out else []))}),flush=True)
    if out['status']!='passed_active_window':sys.exit(1)
if __name__=='__main__':main()
