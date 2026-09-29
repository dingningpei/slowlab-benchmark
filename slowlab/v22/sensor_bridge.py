"""Bridge from a stepped GreenLight engine to the causal observation stores.

``record_at_endpoint`` is the only path by which physical state reaches the
controller store (private) and the public store (agent-visible). Indoor sensors
are read from the raw solver endpoint; archived exterior 10-minute interval
means are released only at their right endpoint, never ahead of time.
"""
from __future__ import annotations

import math
from datetime import timedelta

from .public_sensors import public_endpoint_measurements


def indoor(engine):
    """Deterministic virtual indoor sensors read from the engine endpoint."""
    state = engine.state
    t = float(state['tAir'])
    if not math.isfinite(t):
        raise ValueError('invalid indoor air temperature')
    df = engine.model.full_sol
    has_endpoint = (len(df['Time']) > 0 if isinstance(df, dict) else not df.empty)
    last_time = ((float(df['Time'][-1]) if isinstance(df, dict) else float(df['Time'].iloc[-1]))
                 if has_endpoint else None)
    if has_endpoint and last_time == engine.clock:
        rh = float(df['rhIn'][-1] if isinstance(df, dict) else df['rhIn'].iloc[-1])
        co2 = float(df['co2InPpm'][-1] if isinstance(df, dict) else df['co2InPpm'].iloc[-1])
    else:
        rh = 100 * state['vpAir'] / (610.78 * math.exp(17.2694 * t / (t + 238.3)))
        co2 = 8.3144598 * (t + 273.15) * state['co2Air'] / (101325 * 44.01e-3)
    if not all(math.isfinite(v) for v in (rh, co2)):
        raise ValueError('invalid derived indoor sensor')
    return {'air_temperature_c': t, 'relative_humidity_pct': rh, 'co2_ppm': co2}


def record_at_endpoint(engine, weather, origin, controller, public, compartment, ledger):
    """Record indoor and (at 10-minute boundaries) outdoor measurements at ``engine.clock``."""
    now = engine.clock
    controller.advance_to(now)
    public.advance_to(now)
    inside = indoor(engine)
    for name, value in inside.items():
        controller.record(compartment, name, measurement_time=now, available_at=now, value=value)
    for name, value in public_endpoint_measurements(engine, ledger, inside).items():
        public.record(compartment, name, measurement_time=now, available_at=now, value=value)
    outdoor = None
    if now % 600 == 0:
        # Archived interval means are delivered only at the right endpoint.
        observed = weather.at_utc(origin + timedelta(seconds=now, microseconds=-1))
        if observed.interval_end_utc != origin + timedelta(seconds=now):
            raise AssertionError('weather observation interval ends after delivery')
        outdoor = {'outdoor_temperature_c': observed.t_out_c,
                   'solar_radiation_w_m2': observed.i_glob_w_m2}
        for name, value in outdoor.items():
            controller.record(compartment, name, measurement_time=now, available_at=now, value=value)
    return inside, outdoor
