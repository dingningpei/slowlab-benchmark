import copy
import json
from pathlib import Path

import numpy as np
import pytest

from slowlab.agent_client import CampaignProcess
from slowlab.prior_bo import FrozenGP, PriorBOConfig, PriorLocalBOAgent, fit_shared_hyperparameters, hadamard8

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'configs/task_contract_v8.json').read_text())
POLICIES = json.loads((ROOT / 'configs/campaign_example_v0.json').read_text())
ANCHOR = json.loads((ROOT / 'configs/fixed_reference_v1.json').read_text())['policy']
TICK_DAY = 300 / 86400
KERNEL = {'form': 'season_interaction', 'policy_lengthscales': [0.6, 0.6, 0.8, 0.7, 1.0, 1.2],
          'season_lengthscale': 0.5, 'gamma': 0.3, 'tau': 2.0}
PRODUCT = {'form': 'product', 'lengthscales': [0.6, 0.6, 0.8, 0.7, 1.0, 1.2, 0.5, 0.5]}


def config(**kw):
    return PriorBOConfig(anchor=ANCHOR, kernel=KERNEL, noise_ratio=0.1, **kw)


def run(tmp_path, name, *, mode='endpoint', seed=3, cfg=None, real_calendar=False):
    folder = tmp_path / name
    folder.mkdir()
    contract = copy.deepcopy(CONTRACT)
    if not real_calendar:
        contract['budget'].update(campaign_days=26 * TICK_DAY, crop_days=12 * TICK_DAY,
                                  cleanup_days=1 * TICK_DAY, latest_start_day=13 * TICK_DAY)
    (folder / 'contract.json').write_text(json.dumps(contract))
    spec = {'backend': 'fake', 'contract': str(folder / 'contract.json'), 'feedback_mode': mode,
            'fallback_policy': POLICIES['policy_a'], 'origin_utc': '2016-12-31T23:00:00+00:00',
            'soil_boundary_c': None, 'trace': None, 'settlement_out': str(folder / 'settlement.json'),
            'failure_out': str(folder / 'failure.json')}
    (folder / 'site.json').write_text(json.dumps(spec))
    with CampaignProcess(folder / 'site.json', private_log=folder / 'server.log') as campaign:
        summary = PriorLocalBOAgent(campaign.session, seed, cfg or config()).run()
        transcript = campaign.session.transcript
        assert campaign.close() == 0
    return summary, transcript, json.loads((folder / 'settlement.json').read_text())


def scaled(policy):
    f = CONTRACT['policy']['fields']
    return np.array([(policy[k] - s['min']) / (s['max'] - s['min']) for k, s in f.items()])


def test_frozen_gp_ranking_does_not_depend_on_margin_scale_or_offset():
    rng = np.random.default_rng(0)
    x, y, xs = rng.random((5, 8)), rng.normal(size=5), rng.random((40, 8))
    for kernel in (KERNEL, PRODUCT):
        a = FrozenGP(x, y, kernel, 0.1)
        b = FrozenGP(x, 7.0 * y + 30.0, kernel, 0.1)
        assert np.argsort(a.predict(xs)[0]).tolist() == np.argsort(b.predict(xs)[0]).tolist()
        assert np.allclose(b.predict(xs)[1], 7.0 * a.predict(xs)[1])


def test_season_interaction_separates_a_season_offset_from_policy_effects():
    # margins: a large season offset plus the same policy effect in both seasons
    rng = np.random.default_rng(2)
    groups = []
    for _ in range(30):
        pol = rng.random((4, 6))
        rows = np.vstack([np.hstack([pol, np.tile([0.5, 1.0], (4, 1))]), np.hstack([pol, np.tile([0.5, 0.0], (4, 1))])])
        effect = 8 * pol[:, 0]
        groups.append((rows, np.concatenate([effect + 25, effect]) + rng.normal(0, 0.2, 8) + rng.normal(0, 3)))
    hyper = fit_shared_hyperparameters(groups, rng, restarts=4, form='season_interaction')
    assert hyper['kernel']['gamma'] < 0.3 and hyper['kernel']['tau'] > 1
    # trained on day-0 crops only, it still ranks second-season outcomes by the policy effect
    x, y = groups[0]
    gp = FrozenGP(x[:4], y[:4], hyper['kernel'], hyper['noise_ratio'])
    pred = gp.predict(x[4:])[0]
    assert np.argsort(pred).tolist() == np.argsort(y[4:]).tolist()


def test_shared_fit_finds_the_relevant_input():
    rng = np.random.default_rng(1)
    groups = []
    for _ in range(30):
        x = rng.random((6, 8))
        groups.append((x, 10 * np.sin(3 * x[:, 0]) + rng.normal(0, 0.3, 6) + rng.normal(0, 5)))
    hyper = fit_shared_hyperparameters(groups, rng, restarts=4)
    ls = hyper['kernel']['lengthscales']
    assert ls[0] < min(ls[1:]) and hyper['noise_ratio'] < 0.2


def test_starts_at_the_anchor_and_recommends_only_tried_policies(tmp_path):
    summary, transcript, settlement = run(tmp_path, 'a', cfg=config(radius=0.25))
    starts = [e for e in summary['log'] if e['action'] == 'start']
    assert starts[0]['basis'] == 'anchor' and starts[0]['policy'] == {k: float(v) for k, v in ANCHOR.items()}
    assert [e['basis'] for e in starts[1:4]] == ['anchor_design'] * 3
    for e in starts[1:4]:
        step = np.abs(scaled(e['policy']) - scaled(ANCHOR))
        # every field moves by the radius unless clipped at a bound (or night capped at day)
        assert np.all((np.abs(step - 0.25) < 0.01) | (step < 0.25))
        assert np.sum(np.abs(step - 0.25) < 0.01) >= 3
    assert [e['basis'] for e in starts[4:]] == ['expected_improvement'] * 2
    assert summary['completed_crops'] == 6 and settlement['starts'] == 6
    tried = [e['policy'] for e in starts]
    assert summary['recommendation'] in tried
    assert summary['log'][-1]['basis'] == 'max_posterior_scored_mean_tried'
    assert settlement['recommendation'] == summary['recommendation']
    assert not any('variable' in r['request'] for r in transcript if r['request']['action'] == 'observe')


def test_expected_improvement_stays_inside_the_trust_region(tmp_path):
    summary, _, _ = run(tmp_path, 'a', cfg=config(radius=0.15))
    starts = [e for e in summary['log'] if e['action'] == 'start']
    centres = [scaled(e['policy']) for e in starts[:4]]
    for e in starts[4:]:
        half = e['trust_region_side'] / 2
        assert any(np.all(np.abs(scaled(e['policy']) - c) <= half + 0.01) for c in centres)


def test_same_seed_repeats_and_other_seed_differs(tmp_path):
    a = run(tmp_path, 'a', seed=5)[0]
    b = run(tmp_path, 'b', seed=5)[0]
    c = run(tmp_path, 'c', seed=6)[0]
    assert a == b
    assert a['log'][0]['policy'] == c['log'][0]['policy']  # both start at the anchor
    assert a['log'][1]['policy'] != c['log'][1]['policy']


def test_full_feedback_uses_the_process_predictor_for_late_compartments(tmp_path):
    summary, transcript, settlement = run(tmp_path, 'full', mode='full', real_calendar=True)
    starts = [e for e in summary['log'] if e['action'] == 'start']
    assert [e['basis'] for e in starts[:2]] == ['anchor', 'anchor_design']
    assert all(e['basis'] == 'expected_improvement' for e in starts[2:4])
    assert any(e['action'] == 'predicted_final_margins' for e in summary['log'])
    assert any('variable' in r['request'] for r in transcript if r['request']['action'] == 'observe')
    assert settlement['starts'] == 6 and summary['recommendation'] in [e['policy'] for e in starts]
    assert settlement['decision_calls'] <= CONTRACT['budget']['max_decision_calls']


def test_design_is_orthogonal_and_configuration_is_checked():
    d = hadamard8()[:, 1:]
    assert (d.T @ d == 8 * np.eye(7)).all()
    for bad in (dict(radius=0.0), dict(radius=0.9), dict(stagger_fraction=1.0), dict(initial_units=0)):
        with pytest.raises(ValueError):
            config(**bad)
    with pytest.raises(ValueError):
        PriorBOConfig(anchor=ANCHOR, kernel={'form': 'other'}, noise_ratio=0.1)
