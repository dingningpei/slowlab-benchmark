import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def w(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def test_core_assembly_replaces_only_the_repaired_parts(tmp_path):
    F, R, A = tmp_path / 'F', tmp_path / 'R', tmp_path / 'A'
    jobs = [{'id': 'site00_m_r0', 'method': 'llm', 'model_name': 'm'}, {'id': 'site00_m_r1', 'method': 'llm', 'model_name': 'm'},
            {'id': 'site00_pbo_s0_full', 'method': 'bo', 'feedback': 'full'}]
    w(F / 'campaigns.json', {'out_dir': 'x', 'jobs': jobs})
    for j in jobs[:2]:
        w(F / 'campaigns' / f"{j['id']}.json", {'status': 'completed', 'initial_recommendation': {'a': 1},
                                                 'branches': {'full': {'tag': 'old-full'}, 'endpoint': {'tag': 'old-end'}}})
    w(F / 'campaigns' / 'site00_pbo_s0_full.json', {'status': 'completed', 'tag': 'old-bo'})
    for name in ('site00_m_r0_full_y2002', 'site00_m_r1_full_y2002', 'site00_pbo_s0_full_y2002', 'site00_fixed_reference_y2002'):
        w(F / 'evaluations' / f'{name}.json', {'status': 'completed', 'tag': 'old'})
    w(F / 'phase6' / 'picks.json', {'boundary_sensitivity_sites': [0]})
    w(F / 'phase6' / 'reference' / 'site00_seed0.json', {'best': None})
    w(F / 'phase6' / 'sensitivity' / 'site00_pbo_ueff0_y2002.json', {'tag': 'old'})
    (F / 'ledger.jsonl').write_text('{"usd": 1}\n')
    w(R / 'llm_jobs.json', ['site00_m_r0'])
    w(R / 'llm' / 'site00_m_r0' / 'record.json', {'status': 'completed', 'initial_recommendation': {'a': 1},
                                                   'branches': {'full': {'tag': 'new-full'}}, 'repair': {'live_calls': 3},
                                                   'outbound_audit': {}, 'provider_call_records': []})
    w(R / 'baseline' / 'campaigns' / 'site00_pbo_s0_full.json', {'status': 'completed', 'tag': 'new-bo'})
    for name in ('site00_m_r0_full_y2002', 'site00_pbo_s0_full_y2002'):
        w(R / 'evaluations' / f'{name}.json', {'status': 'completed', 'tag': 'new'})
    w(R / 'sensitivity' / 'site00_pbo_ueff0_y2002.json', {'tag': 'new'})
    (R / 'ledger.jsonl').write_text('{"usd": 0.5}\n')
    run = lambda *a: subprocess.run([sys.executable, str(ROOT / 'scripts/assemble_repaired_run.py'), *a], capture_output=True, text=True)
    p = run('core', '--formal', str(F), '--repair', str(R), '--out', str(A))
    assert p.returncode == 0, p.stderr
    r0 = json.loads((A / 'campaigns' / 'site00_m_r0.json').read_text())
    assert r0['branches']['full']['tag'] == 'new-full' and r0['branches']['endpoint']['tag'] == 'old-end' and r0['repair']['live_calls'] == 3
    assert json.loads((A / 'campaigns' / 'site00_m_r1.json').read_text())['branches']['full']['tag'] == 'old-full'
    assert json.loads((A / 'campaigns' / 'site00_pbo_s0_full.json').read_text())['tag'] == 'new-bo'
    tags = {f.stem: json.loads(f.read_text())['tag'] for f in (A / 'evaluations').glob('*.json')}
    assert tags == {'site00_m_r0_full_y2002': 'new', 'site00_m_r1_full_y2002': 'old', 'site00_pbo_s0_full_y2002': 'new',
                    'site00_fixed_reference_y2002': 'old'}
    assert json.loads((A / 'phase6' / 'sensitivity' / 'site00_pbo_ueff0_y2002.json').read_text())['tag'] == 'new'
    assert (A / 'ledger.jsonl').read_text().count('usd') == 2
    p = run('manifest', '--formal', str(F), '--out', str(A))
    assert p.returncode == 0 and 'campaigns/site00_m_r0.json' in json.loads((A / 'MANIFEST.json').read_text())['files']
