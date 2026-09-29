import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase5_model_extension",
    ROOT / "scripts" / "analyze_phase5_model_extension.py",
)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def synthetic_rows(config):
    base = {"luna": 0.030, "deepseek": 0.020, "qwen": 0.010}
    requested = {key: spec["requested_id"] for key, spec in config["models"].items()}
    rows = []
    for matrix in config["secondary_matrices"]:
        model = matrix["model"]
        for mode in matrix["tool_modes"]:
            for site in range(7100, 7124):
                for gseed in matrix["generation_seeds"]:
                    jitter = (site - 7111.5) * 0.00001 + (gseed - 9701.5) * 0.0001
                    regret = base[model] + jitter - (0.005 if mode == "inference" else 0)
                    rows.append({
                        "task": "T3", "model": requested[model], "tool_mode": mode,
                        "seed": site, "generation_seed": gseed, "regret": regret,
                        "cumulative_regret": 100 + 10 * base[model]
                        - (2 if mode == "inference" else 0),
                        "cash": -500 + site - 7100, "rounds_submitted": 3,
                        "format_failures": 0, "infeasible": 0,
                        "tool_use_records": ([{"inference_adopted": True}]
                                             if mode == "inference" else []),
                        "api_calls": [{
                            "actual_model": requested[model], "provider": "test",
                            "attempt": 1,
                            "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                      "cost": 0.001},
                        }],
                    })
    return rows


def test_frozen_analysis_uses_site_level_pairs_and_prespecified_bh_families():
    config = json.loads((ROOT / "configs" / "phase5_model_task_extension.json").read_text())
    report = module.analyze_rows(
        config, synthetic_rows(config), bootstrap_seed=7, bootstrap_resamples=200
    )
    assert report["n_expected_identities"] == 288
    assert report["n_matched_identities"] == 288
    assert not report["missing_identities"]
    assert not report["duplicate_identities"]
    assert not report["unexpected_rows"]
    assert len(report["tool_contrasts_bh_family"]) == 3
    assert len(report["model_contrasts_bh_family"]) == 6
    for contrast in report["tool_contrasts_bh_family"].values():
        assert contrast["n_sites"] == 24
        assert contrast["mean"] == pytest.approx(-0.005)
        assert "p_bh" in contrast
    assert report["arms"]["luna|inference"]["inference_adoption_rate"] == 1.0
    assert report["arms"]["luna|bare"]["n_episodes"] == 48
