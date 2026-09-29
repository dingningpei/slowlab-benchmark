import copy
import json
from pathlib import Path

import pytest

from scripts.download_weather_v3_development import selected_rows

ROOT = Path(__file__).resolve().parents[1]
PLAN = json.loads((ROOT/'configs/weather_v3_expanded_acquisition_proposal.json').read_text())


def test_development_scope_excludes_sealed_holdouts_and_caps_bytes():
    rows = selected_rows(PLAN,'remaining')
    sample = selected_rows(PLAN,'sample')
    assert len(rows) == 96 and len(sample) == 8
    assert sum(row['size'] for row in rows) == 13_313_151
    assert PLAN['stages']['development']['net_new_bytes_cap'] == 13_027_903
    assert all(row['filename'].rsplit('_',1)[-1][:4] in
               {'2012','2013','2014','2016'} for row in rows)
    assert not any('2015' in row['filename'] or '2025' in row['filename'] for row in rows)


def test_changed_year_or_byte_cap_fails_closed():
    tampered = copy.deepcopy(PLAN)
    tampered['stages']['development']['files'][0]['filename'] = (
        tampered['stages']['development']['files'][0]['filename'].replace('2012','2015'))
    with pytest.raises(RuntimeError):
        selected_rows(tampered,'remaining')
    tampered = copy.deepcopy(PLAN)
    tampered['stages']['development']['net_new_bytes_cap'] += 1
    with pytest.raises(RuntimeError):
        selected_rows(tampered,'sample')
