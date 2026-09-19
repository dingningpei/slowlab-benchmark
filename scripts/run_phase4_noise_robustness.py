#!/usr/bin/env python3
"""Execute the frozen Phase 4 adaptive noise-robustness extension."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def commands(config: dict, out_root: pathlib.Path):
    model = config["model"]
    sites = config["site_seeds"]
    for multiplier in config["noise_multipliers"]:
        for generation_seed in config["generation_seeds"]:
            for mode in config["tool_modes"]:
                destination = out_root / f"noise_{multiplier:g}" / mode
                command = [
                    sys.executable, "-B", str(ROOT / "scripts" / "run_llm.py"),
                    "--model", model["requested_id"],
                    "--temperature", str(model["temperature"]),
                    "--max-tokens", str(model["max_tokens"]),
                    "--tasks", config["task"],
                    "--seeds", str(sites["count"]),
                    "--seed-start", str(sites["start"]),
                    "--generation-seed", str(generation_seed),
                    "--tool-mode", mode,
                    "--prompt-variant", config["prompt_variant"],
                    "--noise-multiplier", str(multiplier),
                    "--out", str(destination),
                ]
                if model.get("reasoning_effort"):
                    command += ["--reasoning-effort", model["reasoning_effort"]]
                yield multiplier, mode, generation_seed, command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path,
                        default=ROOT / "configs" / "phase4_noise_robustness.json")
    parser.add_argument("--out-root", type=pathlib.Path,
                        default=pathlib.Path("/Users/dingningpei/Desktop/slowlab/output/reviews"
                                             "/phase4_noise_robustness"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("status") != "frozen_before_sensitivity_execution":
        raise SystemExit("refusing to run an unfrozen noise-sensitivity protocol")
    for multiplier, mode, generation_seed, command in commands(config, args.out_root):
        print(f"noise={multiplier:g} mode={mode} gseed={generation_seed}", flush=True)
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
