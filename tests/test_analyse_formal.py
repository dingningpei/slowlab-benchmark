import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_eval(d, name, score, year):
    rec = {'status': 'completed', 'score_eur_m2': score,
           'plantings': [{'crops': [{'accrued': {'heat_kwh_m2': 200.0, 'light_kwh_m2': 80.0, 'co2_kg_m2': 5.0}}]}]}
    (d / f'{name}_y{year}.json').write_text(json.dumps(rec))


def test_formal_analysis_on_synthetic_scores(tmp_path):
    rng = np.random.default_rng(0)
    models = ['m1', 'm2', 'm3']
    jobs, ev = [], tmp_path / 'ev'
    ev.mkdir()
    years = [2002, 2003, 2004]
    for s in range(12):
        site = {'site_index': s}
        base = rng.normal(30, 10)
        for r in range(2):
            for m in models:
                jobs.append({'id': f'site{s:02d}_{m}_r{r}', 'method': 'llm', 'model_name': m, 'site': site})
                for y in years:
                    write_eval(ev, f'site{s:02d}_{m}_r{r}_full', base + 3 * (m == 'm1') + rng.normal(0, 1), y)
                    write_eval(ev, f'site{s:02d}_{m}_r{r}_endpoint', base + rng.normal(0, 1), y)
                    if r == 0:
                        write_eval(ev, f'site{s:02d}_{m}_r0_initial', base - 1, y)
        for seed in range(2):
            for cond in ('full', 'endpoint'):
                jobs.append({'id': f'site{s:02d}_pbo_s{seed}_{cond}', 'method': 'bo', 'feedback': cond, 'site': site})
                for y in years:
                    write_eval(ev, f'site{s:02d}_pbo_s{seed}_{cond}', base + rng.normal(0, 1), y)
        for y in years:
            write_eval(ev, f'site{s:02d}_fixed_reference', base - 0.5, y)
    (ev / f'site00_m2_r1_full_y{years[0]}.json').unlink()  # one missing evaluation
    (tmp_path / 'campaigns.json').write_text(json.dumps({'jobs': jobs}))
    out = tmp_path / 'report.json'
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/analyse_formal.py'), '--campaigns', str(tmp_path / 'campaigns.json'),
                           '--evaluations', str(ev), '--report', str(out)], capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, proc.stderr[-2000:]
    rep = json.loads(out.read_text())
    p = rep['primary']
    assert len(p) == 7 and all('p_holm' in v for v in p.values())
    assert p['E1_m1_full_minus_baseline_full']['p_holm'] < 0.01 and 2 < p['E1_m1_full_minus_baseline_full']['mean'] < 4
    assert p['E1_m2_full_minus_baseline_full']['sites'] == 11  # site 0 incomplete for m2 Full
    assert rep['missing_data_sensitivity']['E1_m2_full_minus_baseline_full']['sites'] == 12
    assert rep['missing_site_method'] == {'m2_full': [0]}
    assert len(rep['secondary']) == 3 + 4 + 6 and all('p_bh' in v for v in rep['secondary'].values())
    assert set(p['E2_pbo_full_minus_endpoint']['by_evaluation_year']) == {'2002', '2003', '2004'}
