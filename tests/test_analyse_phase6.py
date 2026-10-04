import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
YEARS = (2014, 2018, 2020)


def ev(d, name, score, year):
    d.mkdir(parents=True, exist_ok=True)
    rec = {'status': 'completed', 'score_eur_m2': score,
           'plantings': [{'crops': [{'margin_eur_m2': score, 'accrued': {'harvest_kg_m2': 30.0}}]},
                         {'crops': [{'margin_eur_m2': score, 'accrued': {'harvest_kg_m2': 30.0}}]}]}
    (d / f'{name}_y{year}.json').write_text(json.dumps(rec))


def test_phase6_analysis_on_synthetic_data(tmp_path):
    rng = np.random.default_rng(0)
    formal, e3, p6 = tmp_path / 'formal', tmp_path / 'e3', tmp_path / 'p6'
    models = ['m1', 'm2']
    jobs = []
    for s in range(6):
        site = {'distribution': 'configs/site_distribution_v1.json', 'master_seed': 20260930, 'site_index': s}
        base = rng.normal(30, 5)
        for m in models:
            jobs.append({'id': f'site{s:02d}_{m}_r0', 'method': 'llm', 'model_name': m, 'site': site, 'year': 2017})
            for r in (0, 1):
                for y in YEARS:
                    ev(formal / 'evaluations', f'site{s:02d}_{m}_r{r}_full', base + 1, y)
                    ev(formal / 'evaluations', f'site{s:02d}_{m}_r{r}_endpoint', base, y)
        for r in (0, 1):
            for cond in ('full', 'endpoint'):
                for y in YEARS:
                    ev(formal / 'evaluations', f'site{s:02d}_pbo_s{r}_{cond}', base, y)
        for y in YEARS:
            ev(formal / 'evaluations', f'site{s:02d}_fixed_reference', base - 1, y)
            for m in models + ['pbo', 'fixed_reference']:
                for u in ('0', '4'):
                    ev(p6 / 'sensitivity', f'site{s:02d}_{m}_ueff{u}', base + (2 if m == 'm1' else 0) - float(u), y)
    (formal / 'campaigns.json').write_text(json.dumps({'jobs': jobs}))
    packets = []
    for i, (m, s) in enumerate([(m, s) for m in models + ['pbo'] for s in range(4)]):
        pid = f'p{i:03d}'
        packets.append({'packet_id': pid, 'source_method': m,
                        'source_job': f'site{s:02d}_{m}_r0' if m != 'pbo' else f'site{s:02d}_pbo_s0_full'})
        for rd, add in (('gp-reader', 0.5), ('m1', 2.0)):
            for y in YEARS:
                ev(e3 / 'evaluations', f'{pid}_{rd}', 30 + add, y)
    (e3 / 'sample.json').write_text(json.dumps({'packets': packets}))
    (p6 / 'picks.json').write_text(json.dumps({'boundary_sensitivity_sites': list(range(6))}))
    for s, seeds in ((0, (0, 1)), (1, (0,))):
        (p6 / 'reference').mkdir(exist_ok=True)
        for sd in seeds:
            (p6 / 'reference' / f'site{s:02d}_seed{sd}.json').write_text(json.dumps({'best': {'final_mean': 40.0 + sd}}))
    report = {'primary': {f'E1_{m}_full_minus_baseline_full': {} for m in models} |
              {f'E2_{m}_full_minus_endpoint': {} for m in models + ['pbo']},
              'secondary': {'x': {'p_sign_flip': 0.01}, 'y': {'p_sign_flip': 0.2}}}
    (tmp_path / 'formal.json').write_text(json.dumps(report))
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/analyse_phase6.py'), '--formal', str(formal), '--phase6', str(p6),
                           '--e3', str(e3), '--formal-report', str(tmp_path / 'formal.json'), '--report', str(tmp_path / 'out.json')],
                          capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads((tmp_path / 'out.json').read_text())
    pooled = out['e3']['reader_swap_gain_pooled']
    assert set(pooled) == {'gp-reader', 'm1'} and pooled['m1']['n'] == 4
    assert set(out['e3']['secondary_bh_with_e3']) == {'x', 'y', 'E3_reader_swap_gp-reader', 'E3_reader_swap_m1'}
    b = out['boundary_sensitivity']['settings']
    assert b['0']['ranking'][0] == 'm1' and abs(b['4']['differences']['m1_minus_baseline']['mean'] - 2) < 1e-9
    e1 = out['dry_matter_sensitivity']['0.05']['E1_m1_full_minus_baseline_full']
    assert abs(e1['mean'] - 1.0) < 1e-9  # equal harvests: the dry-matter change cancels in the difference
    rs = out['reference_search']
    assert rs['search_error_abs_difference'] == [1.0] and rs['sites']['0']['best_known'] == 41.0
