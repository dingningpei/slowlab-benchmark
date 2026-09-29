#!/usr/bin/env python3
"""Execute the frozen post-primary model extension without viewing effects."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from run_phase4 import commands

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("/Users/dingningpei/Desktop/slowlab/output/reviews/phase5_model_extension")


def rows_and_cost(root: Path, config: dict | None = None) -> tuple[int, float]:
    rates = ((config or {}).get("cost_plan_usd", {})
             .get("direct_model_peak_rates_per_million", {}))
    rows, cost = 0, 0.0
    for path in root.rglob("episodes_*.json") if root.exists() else []:
        data = json.loads(path.read_text())
        rows += len(data)
        for row in data:
            for call in row.get("api_calls", []):
                usage = call.get("usage") or {}
                explicit = usage.get("cost")
                if explicit is None:
                    explicit = usage.get("cost_usd")
                if explicit is not None:
                    cost += float(explicit)
                    continue
                model = call.get("requested_model") or call.get("actual_model")
                rate = rates.get(model)
                if rate:
                    # Direct DeepSeek does not return dollar cost. Use peak input
                    # cache-miss and output rates for a conservative hard-cap check.
                    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
                    cost += (float(prompt or 0) * float(rate["input_cache_miss"])
                             + float(completion or 0) * float(rate["output"])) / 1_000_000
    return rows, cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs" / "phase5_model_task_extension.json")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--matrix", action="append",
                        help="matrix name; repeat as needed (default: all in frozen order)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw = args.config.read_bytes()
    config = json.loads(raw)
    if config.get("status") != "frozen_before_post_primary_execution":
        raise SystemExit("refusing to execute a protocol that is not frozen")
    digest = hashlib.sha256(raw).hexdigest()
    args.out_root.mkdir(parents=True, exist_ok=True)
    snapshot = args.out_root / "protocol_snapshot.json"
    hash_path = args.out_root / "protocol_sha256.txt"
    if snapshot.exists() and hashlib.sha256(snapshot.read_bytes()).hexdigest() != digest:
        raise SystemExit("output directory contains a different frozen protocol")
    if not snapshot.exists() and not args.dry_run:
        snapshot.write_bytes(raw)
        hash_path.write_text(digest + "\n")

    names = args.matrix or [item["name"] for item in config["secondary_matrices"]]
    cap = float(config["cost_plan_usd"]["hard_pause_before"])
    prior_cost = float(config["cost_plan_usd"].get("prior_recorded_spend", 0.0))
    run_number = 0
    for name in names:
        for _, cmd in commands(config, f"secondary:{name}", args.out_root):
            run_number += 1
            cmd.append("--quiet-effects")
            rows, current_cost = rows_and_cost(args.out_root, config)
            cost = prior_cost + current_cost
            if cost >= cap:
                raise SystemExit(f"hard cost pause: ${cost:.4f} >= ${cap:.2f}")
            print(f"\n[{run_number}] {name}: starting command; completed rows={rows}, "
                  f"total recorded spend=${cost:.4f}", flush=True)
            print(" ".join(map(str, cmd)) if args.dry_run else "effect values remain hidden", flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=ROOT, check=True)
                rows, current_cost = rows_and_cost(args.out_root, config)
                cost = prior_cost + current_cost
                print(f"[{run_number}] complete; rows={rows}, total recorded spend=${cost:.4f}",
                      flush=True)
                if cost >= cap:
                    raise SystemExit(f"hard cost pause after command: ${cost:.4f} >= ${cap:.2f}")

    rows, current_cost = rows_and_cost(args.out_root, config)
    cost = prior_cost + current_cost
    print(f"execution complete: rows={rows}, total recorded spend=${cost:.4f}, protocol={digest}")


if __name__ == "__main__":
    main()
