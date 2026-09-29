#!/usr/bin/env python3
"""Check the Phase-1 component assumption register against frozen contracts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from slowlab.frozen_paths import frozen_path  # noqa: E402


def verify():
    d = json.loads((ROOT / 'configs/component_evidence_v0.json').read_text())
    contract = json.loads((ROOT / 'configs/task_contract_v3.json').read_text())
    assert d['status'] == 'phase1_assumption_register_not_empirical_validation'
    assert d['evidence_classes'] == ['observed', 'literature_constrained', 'assumed', 'unsupported']
    ids = set()
    for row in d['components']:
        assert row['id'] not in ids
        ids.add(row['id'])
        assert row['class'] in d['evidence_classes']
        assert frozen_path(ROOT, row['source']).is_file()
        assert row['scope']
        assert not (row['class'] == 'unsupported' and row['core_use'])
        if row['class'] == 'assumed' and row['core_use']:
            assert row.get('sensitivity')
    assert d['annual_soil_sensitivity']['soil_boundary_c'] == [8, 20]
    assert d['annual_soil_sensitivity']['status'] == 'completed_paired_scenario_not_calibration'
    soil_audit = json.loads(frozen_path(ROOT, d['annual_soil_sensitivity']['result']).read_text())
    assert soil_audit['status'] == 'passed_paired_structural_audit'
    assert soil_audit['days'] == 365 and soil_audit['weather_year'] == 2017
    assert contract['facility']['floor_area_m2'] == 96
    assert contract['facility']['root_zone'].startswith('non-limiting')
    assert contract['observations']['delivery_delay_seconds'] == 0
    assert contract['economics']['fruit_dry_matter_fraction'] == 0.06
    return {'status': 'passed_phase1_component_evidence_register',
            'components': len(ids), 'unsupported_excluded':
                sum(row['class'] == 'unsupported' for row in d['components']),
            'annual_soil_sensitivity_run': True,
            'annual_soil_sensitivity_calibrated': False}


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
