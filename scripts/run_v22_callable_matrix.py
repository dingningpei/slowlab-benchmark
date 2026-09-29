#!/usr/bin/env python3
"""Run the V2.2 callable-tool pilot or frozen Optimise matrix without effects."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _costs(out: Path, rates: dict) -> tuple[int, dict[str, float]]:
    rows, costs = 0, {"openrouter": 0.0, "deepseek": 0.0}
    for path in out.rglob("episodes_*.json") if out.exists() else []:
        for row in json.loads(path.read_text()):
            rows += 1
            for call in row.get("api_calls", []):
                usage = call.get("usage") or {}
                explicit = usage.get("cost", usage.get("cost_usd"))
                model = call.get("requested_model", "")
                route = "openrouter" if "/" in model else "deepseek"
                if explicit is not None:
                    costs[route] += float(explicit)
                elif model in rates:
                    r = rates[model]
                    inp = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                    outp = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
                    costs[route] += (inp * r["input"] + outp * r["output"]) / 1_000_000
    return rows, costs


def _model_args(spec: dict) -> list[str]:
    args = ["--model", spec["requested_id"], "--temperature", str(spec["temperature"]),
            "--max-tokens", str(spec["max_tokens"])]
    if spec.get("json_mode"):
        args.append("--json-mode")
    if spec.get("reasoning_effort"):
        args += ["--reasoning-effort", spec["reasoning_effort"]]
    for provider in spec.get("provider_only", []):
        args += ["--provider-only", provider]
    if spec.get("expected_provider"):
        args += ["--expected-provider", spec["expected_provider"]]
    if spec.get("expected_actual_model"):
        args += ["--expected-actual-model", spec["expected_actual_model"]]
    if spec.get("rpm"):
        args += ["--rpm", str(spec["rpm"])]
    return args


def identities(config: dict, phase: str) -> list[dict]:
    models = list(config["models"])
    modes = ["bare", "design", "inference", "both"]
    if phase == "pilot":
        cells = [(m, mode) for m in models for mode in modes]
        return [{"model": m, "mode": mode, "site": 8900 + i,
                 "generation_seed": 12001} for i, (m, mode) in enumerate(cells)]
    sites = range(config["optimise_confirmatory"]["site_seeds"]["start"],
                  config["optimise_confirmatory"]["site_seeds"]["start"] +
                  config["optimise_confirmatory"]["site_seeds"]["count"])
    reps = config["optimise_confirmatory"]["requested_generation_replicates"]
    cells = [(m, mode) for m in models for mode in modes]
    blocks = [(s, r) for s in sites for r in reps]
    rng = random.Random(config["optimise_confirmatory"]["schedule"]["seed"])
    rng.shuffle(blocks)
    ordered = []
    for b, (site, rep) in enumerate(blocks):
        offset = b % len(cells)
        order = cells[offset:] + cells[:offset]
        if (b // len(cells)) % 2:
            order = list(reversed(order))
        ordered.extend({"model": m, "mode": mode, "site": site,
                        "generation_seed": rep} for m, mode in order)
    return ordered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--phase", choices=["pilot", "main"], required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    if args.phase == "main" and config.get("status") != "frozen_before_execution":
        raise SystemExit("refusing formal execution: protocol is not frozen_before_execution")
    schedule = identities(config, args.phase)
    args.out_root.mkdir(parents=True, exist_ok=True)
    snapshot = args.out_root / "protocol_snapshot.json"
    schedule_path = args.out_root / "schedule.json"
    if not args.dry_run:
        if snapshot.exists() and snapshot.read_bytes() != config_bytes:
            raise SystemExit("output directory contains a different protocol snapshot")
        snapshot.write_bytes(config_bytes)
        schedule_path.write_text(json.dumps(schedule, indent=2) + "\n")
        (args.out_root / "protocol_sha256.txt").write_text(
            hashlib.sha256(config_bytes).hexdigest() + "\n")

    caps = config["cost_plan_usd"]["hard_pause"]
    rates = config["cost_plan_usd"]["direct_peak_rates_per_million"]
    for i, item in enumerate(schedule, 1):
        rows, costs = _costs(args.out_root, rates)
        if any(costs[k] >= float(caps[k]) for k in costs):
            raise SystemExit(f"hard cost pause before identity {i}: {costs}")
        spec = config["models"][item["model"]]
        dest = args.out_root / item["model"] / item["mode"] / f"gseed{item['generation_seed']}"
        cmd = [sys.executable, "-B", str(ROOT / "scripts/run_llm.py"),
               *_model_args(spec), "--tasks", "T3", "--seeds", "1",
               "--seed-start", str(item["site"]), "--generation-seed",
               str(item["generation_seed"]), "--tool-mode", item["mode"],
               "--tool-delivery", "callable", "--quiet-effects", "--out", str(dest),
               "--blinding-policy",
               str(ROOT / "configs" / "simulation_blinding_v2_2.json")]
        print(f"[{i}/{len(schedule)}] completed={rows} costs={costs} "
              f"identity={item['model']}/{item['mode']}/s{item['site']}/g{item['generation_seed']}",
              flush=True)
        if args.dry_run:
            print(" ".join(cmd)); continue
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    rows, costs = _costs(args.out_root, rates)
    print(f"complete rows={rows} costs={costs}", flush=True)


if __name__ == "__main__":
    main()
