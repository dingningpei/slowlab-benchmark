import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_phase4 import paired_family
from run_phase4 import commands


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
