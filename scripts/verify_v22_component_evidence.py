#!/usr/bin/env python3
"""Check the Phase-1 component assumption register against frozen contracts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify():
    d = json.loads((ROOT / 'configs/v22_component_evidence_v0.json').read_text())
    contract = json.loads((ROOT / 'configs/v22_task_contract_v3.json').read_text())
    assert d['status'] == 'phase1_assumption_register_not_empirical_validation'
    assert d['evidence_classes'] == ['observed', 'literature_constrained', 'assumed', 'unsupported']
    ids = set()
    for row in d['components']:
        assert row['id'] not in ids
        ids.add(row['id'])
        assert row['class'] in d['evidence_classes']
        assert (ROOT / row['source']).is_file()
        assert row['scope']
        assert not (row['class'] == 'unsupported' and row['core_use'])
        if row['class'] == 'assumed' and row['core_use']:
            assert row.get('sensitivity')
    assert d['annual_soil_sensitivity']['soil_boundary_c'] == [8, 20]
    assert contract['facility']['floor_area_m2'] == 96
    assert contract['facility']['root_zone'].startswith('non-limiting')
    assert contract['observations']['delivery_delay_seconds'] == 0
    assert contract['economics']['fruit_dry_matter_fraction'] == 0.06
    return {'status': 'passed_phase1_component_evidence_register',
            'components': len(ids), 'unsupported_excluded':
                sum(row['class'] == 'unsupported' for row in d['components']),
            'annual_soil_sensitivity_run': False}


if __name__ == '__main__':
    print(json.dumps(verify(), sort_keys=True))
