import gzip
import json
from pathlib import Path

import pytest

from scripts.audit_v22_tick_trace import audit

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'configs/v22_task_contract_v3.json'


def fixture(tmp_path):
    contract = json.loads(CONTRACT.read_text())
    controller = contract['observations']['controller_channels']
    public = contract['observations']['public_channels']
    requested = {name: 0.0 for name in ('uBoil', 'uRoof', 'uExtCo2', 'uLamp', 'uThScr')}
    increment = {name: 0.0 for name in ('heat_kwh_m2', 'light_kwh_m2', 'co2_kg_m2',
                                         'harvest_kg_m2', 'transpiration_kg_m2')}
    increment['days'] = 300 / 86400
    units = {}
    for unit in map(str, range(4)):
        units[unit] = {
            'phase': 'active', 'model_input_rows': 1,
            'controller_records': {
                name: {'compartment': unit, 'variable': name,
                       'measurement_time': 0, 'available_at': 0, 'value': 1.0}
                for name in controller},
            'requested': dict(requested), 'realised': dict(requested),
            'ledger_increment': dict(increment),
            'public_endpoint': {
                name: {'compartment': unit, 'variable': name,
                       'measurement_time': 300, 'available_at': 300, 'value': 0.0}
                for name in public},
        }
    frame = {'tick': 0, 'start': 0, 'end': 300, 'units': units}
    result = {'status': 'passed_four_unit_runtime_window', 'completed_steps': 1,
              'ledgers': {unit: {'per_m2': dict(increment)} for unit in units}}
    result_path = tmp_path / 'result.json'
    result_path.write_text(json.dumps(result))
    trace_path = tmp_path / 'trace.jsonl.gz'
    return frame, result_path, trace_path


def write_trace(path, frame):
    with gzip.open(path, 'wt') as stream:
        stream.write(json.dumps(frame) + '\n')


def test_tick_trace_accepts_causal_complete_frame(tmp_path):
    frame, result_path, trace_path = fixture(tmp_path)
    write_trace(trace_path, frame)
    assert audit(trace_path, result_path, CONTRACT)['unit_steps'] == 4


@pytest.mark.parametrize('corruption', ['future_sensor', 'changed_realisation', 'missing_public'])
def test_tick_trace_rejects_broken_event_chain(tmp_path, corruption):
    frame, result_path, trace_path = fixture(tmp_path)
    row = frame['units']['0']
    if corruption == 'future_sensor':
        row['controller_records']['air_temperature_c']['available_at'] = 300
    elif corruption == 'changed_realisation':
        row['realised']['uBoil'] = 0.5
    else:
        del row['public_endpoint']['co2_ppm']
    write_trace(trace_path, frame)
    with pytest.raises(AssertionError):
        audit(trace_path, result_path, CONTRACT)
