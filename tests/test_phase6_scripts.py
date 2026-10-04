import json
import subprocess
import sys
from pathlib import Path

from slowlab.commitment import make_commitment

ROOT = Path(__file__).resolve().parents[1]
SITE = {'distribution': 'configs/site_distribution_v1.json', 'master_seed': 20260930, 'site_index': 3}


def test_sensitivity_evaluation_overrides_only_the_boundary(tmp_path):
    reference = json.loads((ROOT / 'configs/fixed_reference_v1.json').read_text())['policy']
    outs = {}
    for name, extra in (('plain', {}), ('ueff4', {'site_overrides': {'boundary_ueff_w_m2_k': 4.0}})):
        spec = {'backend': 'fake', 'contract': 'configs/task_contract_v8.json', 'policy': reference, 'site': SITE,
                'year': 2014, 'out': str(tmp_path / f'{name}.json'), **extra}
        (tmp_path / f'{name}.spec.json').write_text(json.dumps(spec))
        script = 'scripts/evaluate_sensitivity.py' if extra else 'scripts/evaluate_recommendation.py'
        proc = subprocess.run([sys.executable, str(ROOT / script), str(tmp_path / f'{name}.spec.json')],
                              capture_output=True, text=True, cwd=ROOT)
        assert proc.returncode == 0, proc.stdout + proc.stderr[-2000:]
        outs[name] = json.loads((tmp_path / f'{name}.json').read_text())
    s = outs['ueff4']
    assert s['site_overrides'] == {'boundary_ueff_w_m2_k': 4.0} and 1.0 <= s['site_value_replaced']['boundary_ueff_w_m2_k'] <= 3.0
    assert s['identity'] == outs['plain']['identity'] and s['unit_parameters'] == outs['plain']['unit_parameters']
    bad = {'backend': 'fake', 'contract': 'configs/task_contract_v8.json', 'policy': reference, 'site': SITE,
           'year': 2014, 'out': str(tmp_path / 'bad.json'), 'site_overrides': {'price_eur_per_kg_fresh_equivalent': 9}}
    (tmp_path / 'bad.spec.json').write_text(json.dumps(bad))
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_sensitivity.py'), str(tmp_path / 'bad.spec.json')],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 3


def test_phase6_batches_pick_committed_sites(tmp_path):
    jobs = []
    for s in range(48):
        site = {**SITE, 'site_index': s}
        for m in ('a', 'b'):
            jobs.append({'id': f'site{s:02d}_{m}_r0', 'method': 'llm', 'model_name': m, 'site': site,
                         'evaluation_years': [2014, 2018, 2020]})
    (tmp_path / 'campaigns.json').write_text(json.dumps({'out_dir': str(tmp_path / 'campaigns'), 'jobs': jobs}))
    public, opening = make_commitment({'seed': 7}, label='e3-packet-sample-v1')
    (tmp_path / 'c.json').write_text(json.dumps(public)); (tmp_path / 'o.json').write_text(json.dumps(opening))
    cmd = [sys.executable, str(ROOT / 'scripts/phase6_batches.py'), '--campaigns', str(tmp_path / 'campaigns.json'),
           '--seed-opening', str(tmp_path / 'o.json'), '--commitment', str(tmp_path / 'c.json'), '--out', str(tmp_path / 'p6'),
           '--dev-cache', '/dev', '--formal-cache', '/formal']
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    picks = json.loads((tmp_path / 'p6' / 'picks.json').read_text())
    assert len(set(picks['reference_search_sites'])) == 8 and len(set(picks['boundary_sensitivity_sites'])) == 16
    batch = json.loads((tmp_path / 'p6' / 'sensitivity.json').read_text())
    assert len(batch['jobs']) == 16 * (2 + 1 + 1) * 2 * 3
    assert (tmp_path / 'p6' / 'reference_search.sh').read_text().count('search_reference_policy.py') == 10
