import importlib.util
from pathlib import Path

P=Path(__file__).resolve().parents[1]/'scripts'/'audit_agc2024_inputs.py'
spec=importlib.util.spec_from_file_location('audit_agc2024_inputs',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_indoor_outcomes_are_forbidden_but_control_targets_are_allowed():
    assert m.forbidden('compartment/air_temperature')
    assert m.forbidden('compartment/relative_humidity.microclimate')
    assert m.forbidden('compartment/co2_concentration')
    assert m.forbidden('compartment/humidity_deficit')
    assert not m.forbidden('compartment/co2_concentration_setpoint')
    assert not m.forbidden('compartment/humidity_deficit_vip')
    assert not m.forbidden('weather/air_temperature.outside')

def test_derived_resource_values_are_excluded():
    assert m.derived_only('energy/energy_use.heating')
    assert m.derived_only('economics/heating_costs.per_m2')
    assert not m.derived_only('compartment/heating_lower_circuit/pipe_temperature')
