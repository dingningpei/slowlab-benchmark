import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_phase4 import paired_family
from analyze_phase4_secondary import adjust_bh, paired_condition_contrast
from run_phase4 import commands
from phase4_noise_sensitivity import perturbed_task, AGENT_FOR_CONDITION
from phase4_price_sensitivity import sweep as price_sweep, _perturbed_env
from phase4_factor_range_sensitivity import widened
from phase4_site_distribution_sensitivity import widened_spread, run_episode as run_distribution_episode
from phase4_fixed_policy_sensitivity import (
    _final_recommendation, _site_latent, add_holm_adjustment, canonical_scenario,
    factor_width, paired_summary, physiological_params,
)
from run_llm import noise_scaled_task
from run_phase4_noise_robustness import commands as noise_commands
from analyze_phase4_noise_robustness import analyze as analyze_noise
from slowlab import SlowLabEnv, TASKS
from slowlab.world import MANAGEMENT_FACTORS
from slowlab.economics import SITE_ECON_SPREAD


def config():
    return json.loads((ROOT / "configs" / "phase4_preregistered.json").read_text())


def test_frozen_runner_expands_every_primary_generation_seed_and_mode(tmp_path):
    expanded = list(commands(config(), "primary", tmp_path))
    assert len(expanded) == 8
    texts = [" ".join(command) for _, command in expanded]
    assert all("--seeds 56 --seed-start 6000" in text for text in texts)
    assert {text.split("--generation-seed ")[1].split()[0] for text in texts} == {
        "1701", "1702"}
    assert {text.split("--tool-mode ")[1].split()[0] for text in texts} == {
        "bare", "design", "inference", "both"}
    assert all("--redo-incomplete" not in text for text in texts)


def test_cost_amendment_uses_sol_for_strong_model(tmp_path):
    expanded = list(commands(config(), "secondary:strong_model_replication", tmp_path))
    assert len(expanded) == 4
    assert all("--model openai/gpt-5.6-sol" in " ".join(command)
               for _, command in expanded)


def test_primary_analysis_averages_generation_seeds_within_site():
    rows = []
    for site in [10, 11]:
        for generation_seed in [1, 2]:
            rows.append({"task": "T3", "model": "m", "tool_mode": "bare",
                         "seed": site, "generation_seed": generation_seed,
                         "regret": float(site)})
            rows.append({"task": "T3", "model": "m", "tool_mode": "inference",
                         "seed": site, "generation_seed": generation_seed,
                         "regret": float(site + (1 if site == 10 else 3))})
    result = paired_family(rows, "T3", "m", ["bare", "inference"],
                           [10, 11], [1, 2])
    contrast = result["contrasts"]["inference_minus_bare"]
    assert contrast["n_sites"] == 2
    assert np.isclose(contrast["mean"], 2.0)
    assert result["missing_cells"] == []


def test_secondary_analysis_uses_benjamini_hochberg_with_monotone_adjustment():
    adjusted = adjust_bh([("a", 0.01), ("b", 0.04), ("c", 0.03)])
    assert np.isclose(adjusted["a"], 0.03)
    assert np.isclose(adjusted["b"], 0.04)
    assert np.isclose(adjusted["c"], 0.04)


def test_feedback_analysis_pairs_generation_seeds_then_clusters_by_site():
    rows = []
    for site, effect in [(10, -1.0), (11, 3.0)]:
        for generation_seed in [1, 2]:
            for condition, regret in [("terminal_only", 5.0),
                                      ("within_cycle", 5.0 + effect)]:
                rows.append({"task": "T3", "model": "m", "seed": site,
                             "generation_seed": generation_seed,
                             "analysis_condition": condition, "regret": regret})
    result = paired_condition_contrast(
        rows, "T3", "m", "terminal_only", "within_cycle", [10, 11], [1, 2])
    assert result["contrast"]["n_sites"] == 2
    assert np.isclose(result["contrast"]["mean"], 1.0)
    assert result["missing_cells"] == []


def test_perturbed_task_scales_only_the_noise_layer():
    base = TASKS["Optimise"]
    doubled = perturbed_task(2.0)
    assert np.isclose(doubled.plant_cv, base.plant_cv * 2)
    assert np.isclose(doubled.tau_chamber, base.tau_chamber * 2)
    assert np.isclose(doubled.tau_loop, base.tau_loop * 2)
    assert np.isclose(doubled.tau_batch, base.tau_batch * 2)
    # everything that is not a noise-layer parameter must be untouched, so a
    # multiplier can never silently change the budget, factors, or facility size
    assert doubled.factor_names == base.factor_names
    assert doubled.n_rounds == base.n_rounds
    assert doubled.units_per_round == base.units_per_round
    assert set(AGENT_FOR_CONDITION) == {"bare", "design", "inference", "both"}


def test_price_sensitivity_sweep_perturbs_only_the_named_econ_fields_and_restores_nothing_global():
    task = TASKS["Optimise"]
    # price_per_kg_fw is drawn per site (sample_site_econ), not fixed at the
    # EconomicModel default, so a mutation that leaked into module/class state
    # would show up as this seed's *un-swept* price changing between the two reads.
    price_before = SlowLabEnv(task, seed=0).truth.econ.price_per_kg_fw
    result = price_sweep("smoke", (1.0, 2.0), ("price_per_kg_fw",), [0, 1], task)
    assert result["fields_scaled"] == ["price_per_kg_fw"]
    assert set(result["contrasts_by_multiplier"]["1.0"]) >= {
        "mean_regret", "design_minus_bare", "inference_minus_bare", "both_minus_bare"}
    price_after = SlowLabEnv(task, seed=0).truth.econ.price_per_kg_fw
    assert np.isclose(price_before, price_after)


def test_perturbed_env_disables_oracle_caching_so_regret_reflects_the_econ_mutation():
    # World.oracle() caches its "best" value on disk keyed by (seed, factor bounds,
    # cycle_days, energy_shock) -- none of which cover the econ fields this script
    # perturbs. Without `_oracle_cacheable = False`, a second call for the same seed
    # silently returns the first call's oracle value regardless of the econ mutation,
    # which showed up as regret going negative and "optimum" frozen across
    # multipliers. Guard both symptoms directly.
    task = TASKS["Optimise"]
    env = _perturbed_env(task, seed=0, econ_overrides={})
    assert env.truth._oracle_cacheable is False


def test_widened_factor_range_restores_bounds_after_the_context_manager_exits():
    spec = next(f for f in MANAGEMENT_FACTORS if f.name == "density")
    old_low, old_high = spec.low, spec.high
    with widened("density", 2.0):
        assert np.isclose(spec.high - spec.low, (old_high - old_low) * 2)
        assert not np.isclose(spec.low, old_low) or not np.isclose(spec.high, old_high)
    assert np.isclose(spec.low, old_low)
    assert np.isclose(spec.high, old_high)


def test_widened_spread_keeps_the_center_fixed_and_restores_after_exit():
    old_lo, old_hi = SITE_ECON_SPREAD["price_per_kg_fw"]
    old_center = (old_lo + old_hi) / 2
    with widened_spread("price_per_kg_fw", 2.0):
        lo, hi = SITE_ECON_SPREAD["price_per_kg_fw"]
        assert np.isclose(hi - lo, (old_hi - old_lo) * 2)
        assert np.isclose((lo + hi) / 2, old_center)
    assert SITE_ECON_SPREAD["price_per_kg_fw"] == (old_lo, old_hi)


def test_site_distribution_sweep_does_not_reuse_an_oracle_from_another_width(monkeypatch):
    seen = []

    class Agent:
        probes = None

        def run(self, env):
            seen.append(env.truth._oracle_cacheable)
            return np.full(env.task.d, 0.5)

    monkeypatch.setattr("phase4_site_distribution_sensitivity.make_agent",
                        lambda _: Agent())
    run_distribution_episode(TASKS["Optimise"], "bare", 987654)
    assert seen == [False]


def test_fixed_policy_recommendation_recovery_handles_final_and_early_stop():
    final = [{"assistant": json.dumps({
        "recommendation": {"day_temp": 24, "density": 2.4}})}]
    stopped = [{"assistant": json.dumps({
        "stop": True, "current_best": {"day_temp": 23, "density": 3.1}})}]
    assert _final_recommendation(final)[0] == "recommendation"
    assert _final_recommendation(stopped)[0] == "current_best"


def test_fixed_policy_pairing_averages_generation_seeds_inside_site():
    rows = []
    for site, effect in [(10, -1.0), (11, 3.0)]:
        for gseed in (1, 2):
            rows.extend([
                {"tool_mode": "bare", "site": site,
                 "generation_seed": gseed, "regret": 5.0},
                {"tool_mode": "design", "site": site,
                 "generation_seed": gseed, "regret": 5.0 + effect},
                {"tool_mode": "inference", "site": site,
                 "generation_seed": gseed, "regret": 5.0},
                {"tool_mode": "both", "site": site,
                 "generation_seed": gseed, "regret": 5.0},
            ])
    result = paired_summary(rows, n_boot=200, seed=1)
    assert result["design_minus_bare"]["n_sites"] == 2
    assert np.isclose(result["design_minus_bare"]["mean"], 1.0)


def test_physiology_scenarios_reproduce_baseline_and_contract_fruit_span():
    baseline = physiological_params(6000, veg_cv=0.06, fruit_span_fraction=1.0)
    frozen = SlowLabEnv(TASKS["Optimise"], seed=6000).truth.params
    assert baseline.values == frozen.values
    midpoint_a = _site_latent(6000, fruit_span_fraction=0.0)
    midpoint_b = _site_latent(6001, fruit_span_fraction=0.0)
    assert np.allclose(midpoint_a[4:], 0.5)
    assert np.allclose(midpoint_b[4:], 0.5)


def test_fixed_policy_factor_width_restores_bounds():
    spec = next(f for f in MANAGEMENT_FACTORS if f.name == "day_temp")
    original = (spec.low, spec.high)
    with factor_width("day_temp", 0.8):
        assert np.isclose(spec.high - spec.low, 0.8 * (original[1] - original[0]))
    assert (spec.low, spec.high) == original


def test_equivalent_fixed_policy_baselines_share_a_canonical_scenario():
    baseline = canonical_scenario({"label": "frozen"})
    assert canonical_scenario({"econ_multipliers": {"elec_price": 1.0}}) == baseline
    assert canonical_scenario({"physiology": {
        "veg_cv": 0.06, "fruit_span_fraction": 1.0}}) == baseline
    assert canonical_scenario({"factor_width": {
        "factor": "density", "multiplier": 1.0}}) == baseline


def test_fixed_policy_holm_family_counts_equivalent_scenarios_once():
    def result(scenario, p):
        return {"scenario": scenario, "paired_contrasts": {
            "design_minus_bare": {"p_raw_normal_approx": p}}}

    results = {
        "a": {"baseline": result({"label": "baseline"}, 0.01)},
        "b": {"same_baseline": result({"econ_multipliers": {"elec_price": 1.0}}, 0.01),
              "changed": result({"econ_multipliers": {"elec_price": 2.0}}, 0.04)},
    }
    audit = add_holm_adjustment(results)
    assert audit["n_tests"] == 2
    assert np.isclose(results["a"]["baseline"]["paired_contrasts"]
                      ["design_minus_bare"]["p_holm_post_hoc_sensitivity_family"], 0.02)


def test_llm_noise_multiplier_changes_only_noise_fields():
    base = TASKS["Optimise"]
    changed = noise_scaled_task(base, 2.0)
    for field in ("plant_cv", "tau_chamber", "tau_loop", "tau_batch"):
        assert np.isclose(getattr(changed, field), 2.0 * getattr(base, field))
    assert changed.factor_names == base.factor_names
    assert changed.n_rounds == base.n_rounds
    assert changed.units_per_round == base.units_per_round
    assert changed.cycle_days == base.cycle_days


def test_frozen_noise_runner_expands_all_cells(tmp_path):
    cfg = json.loads((ROOT / "configs" / "phase4_noise_robustness.json").read_text())
    expanded = list(noise_commands(cfg, tmp_path))
    assert len(expanded) == 24
    texts = [" ".join(item[-1]) for item in expanded]
    assert all("--seeds 16 --seed-start 9400" in text for text in texts)
    assert {text.split("--noise-multiplier ")[1].split()[0] for text in texts} == {
        "0.5", "1.0", "2.0"}


def test_noise_analysis_pairs_generation_seeds_and_computes_interaction(tmp_path):
    cfg = {
        "protocol_id": "test", "analysis_family": "test",
        "noise_multipliers": [0.5, 1.0, 2.0],
        "tool_modes": ["bare", "design", "inference", "both"],
        "site_seeds": {"start": 10, "count": 2}, "generation_seeds": [1, 2],
    }
    rows = []
    for mult in cfg["noise_multipliers"]:
        for site in (10, 11):
            for gseed in (1, 2):
                for mode in cfg["tool_modes"]:
                    effect = 0.0 if mode == "bare" else mult
                    rows.append({"task": "T3", "model": "m", "tool_mode": mode,
                                 "seed": site, "generation_seed": gseed,
                                 "noise_multiplier": mult, "regret": 5.0 + effect})
    folder = tmp_path / "noise" / "cell"
    folder.mkdir(parents=True)
    (folder / "episodes_test.json").write_text(json.dumps(rows))
    result = analyze_noise(cfg, tmp_path)
    assert not result["missing_cells"]
    assert result["n_episode_rows"] == 48
    assert np.isclose(result["primary_high_noise"]["inference_minus_bare"]["mean"], 2.0)
    assert np.isclose(result["secondary"]
                      ["noise_2.0_minus_1.0:inference_minus_bare"]["mean"], 1.0)
