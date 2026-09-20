import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase5_model_task_extension.json"
FROZEN_SHA256 = "1de54d9063d5c10ba9147885ced59f60d9fb62023a4afff61293b2c67536249e"


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
