"""Common sampled controller; accepts available sensor records, not model state."""
import math
from .task_contract import validate_policy
from .policy import Policy
from .greenlight_smoke import held_commands
from .resources import standby_commands


def proportional(x,target,band):
    z=2/band*math.log(100)*(x-target-band/2)
    return 1/(1+math.exp(-max(-700,min(700,z))))


def screen_command(contract, temperature, rh, outdoor_temperature, local_hour, heat_target, rh_target):
    s=contract['controller']['thermal_screen']
    day=6<=local_hour<22
    cold=proportional(outdoor_temperature,s['outdoor_day_target_c'] if day else s['outdoor_night_target_c'],s['cold_band_c'])
    heat=1-proportional(temperature,heat_target+s['heat_dead_zone_c'],-s['cold_band_c'])
    vent_cold=1-proportional(temperature,heat_target-s['vent_cold_offset_c'],s['vent_cold_band_c'])
    humidity=max(1-proportional(rh,rh_target+s['rh_offset_pct'],s['rh_band_pct']),1-vent_cold)
    return min(cold,heat,humidity)


def commands_from_observations(contract, observations, compartment, *, phase, policy=None, local_clock_offset_seconds=0):
    """No time advancement or hidden-state fallback. Event is executor-private.

    Zero-delay and delayed measurements share the same latest-arrived semantics.
    Active policy remains the previously accepted design; standby ignores it.
    """
    if phase not in ('active','cleanup','idle'):raise ValueError('invalid controller phase')
    if isinstance(local_clock_offset_seconds,bool) or not isinstance(local_clock_offset_seconds,(int,float)) or not math.isfinite(local_clock_offset_seconds) or not 0<=local_clock_offset_seconds<86400:
        raise ValueError('invalid local clock offset')
    if phase=='active':
        if isinstance(policy,Policy):policy.require_contract(contract)
        else:validate_policy(contract,policy)
    names=['air_temperature_c','relative_humidity_pct','outdoor_temperature_c']
    if phase=='active':names+=['co2_ppm','solar_radiation_w_m2']
    now=observations.clock;records={};values={}
    for name in names:
        record=observations.latest(compartment,name,as_of=now)
        if record is None:raise ValueError('missing available controller sensor: '+name)
        if not (0<=record['measurement_time']<=record['available_at']<=now):raise ValueError('unavailable sensor')
        if record['unit']!=contract['observations']['controller_channels'][name]:raise ValueError('controller sensor unit mismatch')
        v=record['value']
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError('invalid controller sensor')
        values[name]=v;records[name]=dict(record)
    hour=((now+local_clock_offset_seconds)/3600)%24;day=6<=hour<22
    if phase=='active':
        target=policy['day_temperature_c'] if day else policy['night_temperature_c'];rh_target=policy['vent_rh_threshold_pct']
        u=held_commands(policy,{'temperature':values['air_temperature_c'],'rh':values['relative_humidity_pct'],'co2':values['co2_ppm'],'solar':values['solar_radiation_w_m2']},hour)
    else:
        target=contract['controller']['standby']['heating_target_c'];rh_target=contract['controller']['thermal_screen']['standby_rh_target_pct']
        u=standby_commands(contract,values['air_temperature_c'])
    u['uThScr']=screen_command(contract,values['air_temperature_c'],values['relative_humidity_pct'],values['outdoor_temperature_c'],hour,target,rh_target)
    return u,{'time':now,'phase':phase,'sensor_records':records,'requested':dict(u),'heat_target_c':target,'screen_rh_target_pct':rh_target}
