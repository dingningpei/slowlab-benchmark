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
from slowlab import SlowLabEnv, TASKS
from slowlab.world import MANAGEMENT_FACTORS


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
