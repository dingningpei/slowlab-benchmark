#!/usr/bin/env python3
"""Execute a frozen Phase 4 matrix without reading interim condition effects."""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def model_args(config, key):
    model = config["models"][key]
    args = ["--model", model["requested_id"], "--temperature", str(model["temperature"]),
            "--max-tokens", str(model["max_tokens"])]
    if model.get("reasoning_effort"):
        args += ["--reasoning-effort", model["reasoning_effort"]]
    return args


def commands(config, selection, out_root):
    if selection == "primary":
        matrices = [("primary", config["primary_matrix"])]
    elif selection == "exploratory":
        matrices = [("exploratory", config["exploratory_matrix"])]
    else:
        wanted = selection.removeprefix("secondary:")
        matrices = [(item["name"], item) for item in config["secondary_matrices"]
                    if wanted in {"all", item["name"]}]
        if not matrices:
            raise SystemExit(f"unknown matrix selection {selection!r}")

    for name, matrix in matrices:
        model_key = matrix["model"]
        tasks = matrix.get("tasks", [matrix.get("task")])
        sites = matrix["site_seeds"]
        variants = matrix.get("prompt_variants", [matrix.get("prompt_variant", "standard")])
        conditions = matrix.get("conditions")
        modes = matrix.get("tool_modes", [matrix.get("tool_mode", "bare")])
        for variant in variants:
            for generation_seed in matrix["generation_seeds"]:
                if conditions:
                    cells = [("bare", condition == "within_cycle", condition)
                             for condition in conditions]
                else:
                    cells = [(mode, False, mode) for mode in modes]
                for mode, within, label in cells:
                    destination = out_root / name / variant / label
                    cmd = [sys.executable, "-B", str(ROOT / "scripts" / "run_llm.py")]
                    cmd += model_args(config, model_key)
                    cmd += ["--tasks", *tasks, "--seeds", str(sites["count"]),
                            "--seed-start", str(sites["start"]),
                            "--generation-seed", str(generation_seed),
                            "--tool-mode", mode, "--prompt-variant", variant,
                            "--out", str(destination)]
                    if within:
                        cmd.append("--within-cycle")
                    yield name, cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path,
                        default=ROOT / "configs" / "phase4_preregistered.json")
    parser.add_argument("--matrix", default="primary",
                        help="primary, exploratory, secondary:all, or secondary:<name>")
    parser.add_argument("--out-root", type=pathlib.Path, default=pathlib.Path(
        "/Users/dingningpei/Desktop/slowlab/output/reviews/phase4_confirmatory"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get("status") not in {
            "frozen_before_confirmatory_execution",
            "frozen_amendment_before_confirmatory_execution",
            "frozen_execution_amendment",
            "frozen_before_post_primary_execution"}:
        raise SystemExit("refusing to execute a protocol that is not frozen")
    for name, cmd in commands(config, args.matrix, args.out_root):
        print(name, " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
