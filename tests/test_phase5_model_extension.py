import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase5_model_task_extension.json"
FROZEN_SHA256 = "1de54d9063d5c10ba9147885ced59f60d9fb62023a4afff61293b2c67536249e"
A2_CONFIG = ROOT / "configs" / "phase5_model_task_extension_a2.json"
A2_FROZEN_SHA256 = "af66534b5d975b0eee4e071b6c7ba09c8bf0f70c4541964c723ca10f81e60dd5"


def test_phase5_model_extension_is_frozen_optimise_only_and_has_288_identities():
    raw = CONFIG.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FROZEN_SHA256
    cfg = json.loads(raw)
    assert cfg["status"] == "frozen_before_post_primary_execution"
    assert set(cfg["models"]) == {"luna", "deepseek", "qwen"}
    assert cfg["cost_plan_usd"]["hard_pause_before"] == 4.0

    identities = []
    for matrix in cfg["secondary_matrices"]:
        assert matrix["task"] == "T3"
        assert matrix["tool_modes"] == ["bare", "inference"]
        assert matrix["site_seeds"] == {"start": 7100, "count": 24}
        assert matrix["generation_seeds"] == [9701, 9702]
        assert matrix["prompt_variant"] == "standard"
        for mode in matrix["tool_modes"]:
            for site in range(7100, 7124):
                for generation_seed in matrix["generation_seeds"]:
                    identities.append((matrix["model"], mode, site, generation_seed))

    assert len(identities) == 288
    assert len(set(identities)) == 288
    assert cfg["execution_size"] == {
        "model_anchor_episodes": 288,
        "total_episodes": 288,
    }


def test_phase5_analysis_manifest_locks_script_and_multiplicity():
    manifest = json.loads(
        (ROOT / "configs" / "phase5_model_extension_analysis.json").read_text()
    )
    assert manifest["status"] == "frozen_before_post_primary_execution"
    assert manifest["parent_protocol_sha256"] == FROZEN_SHA256
    script = ROOT / manifest["analysis_script"]
    assert hashlib.sha256(script.read_bytes()).hexdigest() == manifest["analysis_script_sha256"]
    assert manifest["statistical_unit"] == "site"
    assert [family["members"] for family in manifest["multiplicity_families"]] == [3, 6]
    assert all(family["adjustment"] == "Benjamini-Hochberg"
               for family in manifest["multiplicity_families"])


def test_phase5_a2_amendment_uses_direct_deepseek_and_accounts_for_prior_spend():
    raw = A2_CONFIG.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == A2_FROZEN_SHA256
    cfg = json.loads(raw)
    assert cfg["status"] == "frozen_before_post_primary_execution"
    assert cfg["amends"]["protocol_id"] == cfg["protocol_id"].replace("a2", "a1")
    assert cfg["amends"]["unchanged_completed_cells"] == {"model_anchor_luna": 96}
    assert cfg["amends"]["excluded_pilot"]["episodes"] == 22
    assert cfg["models"]["deepseek"] == {
        "requested_id": "deepseek-flash",
        "reported_version": "DeepSeek-V4.1-Flash",
        "provider_route": "DeepSeek direct API",
        "temperature": 0.0,
        "max_tokens": 4096,
        "reasoning_effort": "low",
    }
    assert cfg["cost_plan_usd"]["prior_recorded_spend"] == 0.419938676
    assert cfg["cost_plan_usd"]["hard_pause_before"] == 4.0
    assert cfg["cost_plan_usd"]["direct_model_peak_rates_per_million"] == {
        "deepseek-flash": {
            "input_cache_miss": 0.3,
            "output": 1.2,
            "currency": "USD",
            "policy": "conservative peak-rate upper bound; cache hits intentionally priced as misses",
        }
    }


def test_phase5_runner_prices_direct_usage_conservatively(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from run_phase5_model_extension import rows_and_cost

    episode = {
        "api_calls": [
            {
                "requested_model": "deepseek-flash",
                "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
            },
            {
                "requested_model": "openai/gpt-5.6-luna",
                "usage": {"cost": 0.25},
            },
        ]
    }
    (tmp_path / "episodes_test.json").write_text(json.dumps([episode]))
    cfg = {
        "cost_plan_usd": {
            "direct_model_peak_rates_per_million": {
                "deepseek-flash": {"input_cache_miss": 0.3, "output": 1.2}
            }
        }
    }
    assert rows_and_cost(tmp_path, cfg) == (1, 1.75)
