"""Deterministic stand-in physics for protocol and isolation tests.

It never reads weather, calendar year, deep-soil boundary or any other private
site input, so two servers that differ only in private inputs must produce
byte-identical public transcripts. Not a crop model; never used for results.
"""
from __future__ import annotations

from types import SimpleNamespace


class FakeLifecycle:
    def __init__(self, contract, source, start, **kwargs):
        self.clock = float(start)
        self.mode = 'empty'
        self.ready_at = float(start)
        self.state = {'tAir': 20.0}
        self.model = SimpleNamespace(full_sol={}, input_data=[{}])
        self.cleanup = contract['budget']['cleanup_days'] * 86400

    @property
    def engine(self):
        return self

    def replant(self):
        self.mode = 'active'

    def stop(self):
        self.mode = 'empty'
        self.ready_at = self.clock + self.cleanup

    def step(self, commands, end):
        start = self.clock
        self.clock = float(end)
        self.state['tAir'] = 20.0 + commands['uBoil'] - commands['uRoof']
        zero = [0.0, 0.0]
        self.model.full_sol = {'Time': [start, end], 'hBoilPipe': [10.0, 10.0], 'qLampIn': list(zero),
                               'mcExtAir': list(zero), 'mcFruitHar': ([1.0, 1.0] if self.mode == 'active' else zero),
                               'mvCanAir': ([1e-6, 1e-6] if self.mode == 'active' else zero)}
        return dict(self.state)


def sample_endpoint(engine, weather, origin, controller, public, unit, ledger):
    now = engine.clock
    controller.advance_to(now)
    public.advance_to(now)
    per_m2 = ledger.summary()['per_m2']
    t = engine.state['tAir']
    for name, value in {'air_temperature_c': t, 'relative_humidity_pct': 75.0, 'co2_ppm': 500.0,
                        'outdoor_temperature_c': 10.0, 'solar_radiation_w_m2': 100.0}.items():
        controller.record(unit, name, measurement_time=now, available_at=now, value=value)
    for name, value in {'air_temperature_c': t, 'relative_humidity_pct': 75.0, 'co2_ppm': 500.0,
                        'canopy_lai_proxy': 1.0 if engine.mode == 'active' else 0.0,
                        'cumulative_harvest_fresh_equivalent': per_m2['harvest_kg_m2'],
                        'heating_energy': per_m2['heat_kwh_m2'], 'lighting_energy': per_m2['light_kwh_m2'],
                        'co2_dosed': per_m2['co2_kg_m2']}.items():
        public.record(unit, name, measurement_time=now, available_at=now, value=value)


class FailingLifecycle(FakeLifecycle):
    """Raises a numerical fault once time reaches 600 s (infrastructure-failure tests)."""

    def step(self, commands, end):
        if end >= 600:
            raise FloatingPointError('injected numerical fault')
        return super().step(commands, end)
