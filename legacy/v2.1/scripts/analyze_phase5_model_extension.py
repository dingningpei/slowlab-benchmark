#!/usr/bin/env python3
"""Frozen site-level analysis for the 288-episode post-primary model extension."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 20260920
BOOTSTRAP_RESAMPLES = 10_000


def load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(root.rglob("episodes_*.json")):
        for row in json.loads(path.read_text()):
            item = dict(row)
            item["source"] = str(path)
            rows.append(item)
    return rows


def normal_pvalue(values: list[float] | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or values.std(ddof=1) == 0:
        return 1.0 if not len(values) or values.mean() == 0 else 0.0
    z = abs(values.mean() / (values.std(ddof=1) / math.sqrt(len(values))))
    return float(math.erfc(z / math.sqrt(2)))


def bootstrap_ci(values: list[float] | np.ndarray, *, seed: int,
                 n_boot: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def adjust_bh(items: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(items, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 1.0
    total = len(ordered)
    for rank, (name, p_value) in reversed(list(enumerate(ordered, start=1))):
        running = min(running, total * p_value / rank)
        adjusted[name] = min(1.0, running)
    return adjusted


def expected_identities(config: dict) -> set[tuple[str, str, int, int]]:
    identities = set()
    for matrix in config["secondary_matrices"]:
        start = int(matrix["site_seeds"]["start"])
        count = int(matrix["site_seeds"]["count"])
        for mode in matrix["tool_modes"]:
            for site in range(start, start + count):
                for generation_seed in matrix["generation_seeds"]:
                    identities.add((matrix["model"], mode, site, int(generation_seed)))
    return identities


def _mean_se(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()) if len(x) else None,
        "se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else None,
    }


def _adoption(rows: list[dict], field: str) -> float | None:
    values = [record[field]
              for row in rows for record in row.get("tool_use_records", [])
              if record.get(field) is not None]
    return float(np.mean(values)) if values else None


def _arm_summary(rows: list[dict]) -> dict:
    calls = [call for row in rows for call in row.get("api_calls", [])]
    usage = [call.get("usage") or {} for call in calls]
    return {
        "n_episodes": len(rows),
        "n_sites": len({int(row["seed"]) for row in rows}),
        "final_simple_regret": _mean_se([float(row["regret"]) for row in rows]),
        "cumulative_regret": _mean_se(
            [float(row["cumulative_regret"]) for row in rows]),
        "campaign_cash": _mean_se([float(row["cash"]) for row in rows]),
        "mean_rounds_submitted": _mean_se(
            [float(row["rounds_submitted"]) for row in rows]),
        "format_failures": int(sum(int(row.get("format_failures") or 0) for row in rows)),
        "infeasible_submissions": int(sum(int(row.get("infeasible") or 0) for row in rows)),
        "episodes_below_round_cap": int(sum(int(row.get("rounds_submitted") or 0) < 3
                                               for row in rows)),
        "design_adoption_rate": _adoption(rows, "design_adopted"),
        "inference_adoption_rate": _adoption(rows, "inference_adopted"),
        "api_audit": {
            "successful_calls": len(calls),
            "actual_models": sorted({call.get("actual_model") for call in calls
                                      if call.get("actual_model")}),
            "providers": sorted({call.get("provider") for call in calls
                                 if call.get("provider")}),
            "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in usage),
            "completion_tokens": sum(int(item.get("completion_tokens") or 0)
                                     for item in usage),
            "reported_cost_usd": sum(float(item.get("cost") or 0.0) for item in usage),
            "max_successful_attempt": max(
                (int(call.get("attempt", 1)) for call in calls), default=None),
        },
    }


def _contrast(values: list[float], *, seed: int, n_boot: int) -> dict:
    x = np.asarray(values, dtype=float)
    sd = float(x.std(ddof=1)) if len(x) > 1 else None
    return {
        "n_sites": int(len(x)),
        "mean": float(x.mean()) if len(x) else None,
        "median": float(np.median(x)) if len(x) else None,
        "sd": sd,
        "standardized_effect": float(x.mean() / sd) if sd else None,
        "ci95": bootstrap_ci(x, seed=seed, n_boot=n_boot) if len(x) else None,
        "p_raw_normal_approx": normal_pvalue(x),
        "site_values": [float(value) for value in x],
    }


def analyze_rows(config: dict, rows: list[dict], *, bootstrap_seed: int = BOOTSTRAP_SEED,
                 bootstrap_resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    requested_to_key = {
        spec["requested_id"]: key for key, spec in config["models"].items()
    }
    cell: dict[tuple[str, str, int, int], dict] = {}
    duplicates: list[tuple[str, str, int, int]] = []
    unexpected: list[dict] = []
    expected = expected_identities(config)

    for row in rows:
        model_key = requested_to_key.get(row.get("model"))
        identity = None if model_key is None else (
            model_key,
            row.get("tool_mode"),
            int(row["seed"]),
            int(row["generation_seed"]),
        )
        if row.get("task") != "T3" or identity not in expected:
            unexpected.append({
                "task": row.get("task"), "model": row.get("model"),
                "tool_mode": row.get("tool_mode"), "seed": row.get("seed"),
                "generation_seed": row.get("generation_seed"),
                "source": row.get("source"),
            })
            continue
        if identity in cell:
            duplicates.append(identity)
        cell[identity] = row

    missing = sorted(expected - set(cell))
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for (model, mode, _, _), row in cell.items():
        grouped[(model, mode)].append(row)
    arms = {
        f"{model}|{mode}": _arm_summary(grouped[(model, mode)])
        for model in config["models"] for mode in ("bare", "inference")
    }

    sites = sorted({identity[2] for identity in expected})
    gseeds = sorted({identity[3] for identity in expected})
    tool_contrasts = {}
    for model in config["models"]:
        regret_values, cumulative_values = [], []
        for site in sites:
            rdiff, cdiff = [], []
            for gseed in gseeds:
                treatment = cell.get((model, "inference", site, gseed))
                baseline = cell.get((model, "bare", site, gseed))
                if treatment is not None and baseline is not None:
                    rdiff.append(float(treatment["regret"] - baseline["regret"]))
                    cdiff.append(float(treatment["cumulative_regret"]
                                       - baseline["cumulative_regret"]))
            if rdiff:
                regret_values.append(float(np.mean(rdiff)))
                cumulative_values.append(float(np.mean(cdiff)))
        result = _contrast(regret_values, seed=bootstrap_seed,
                           n_boot=bootstrap_resamples)
        result["cumulative_regret"] = _contrast(
            cumulative_values, seed=bootstrap_seed + 1,
            n_boot=bootstrap_resamples)
        tool_contrasts[f"{model}:inference_minus_bare"] = result

    tool_adjusted = adjust_bh([
        (name, value["p_raw_normal_approx"])
        for name, value in tool_contrasts.items()
    ])
    for name, value in tool_contrasts.items():
        value["p_bh"] = tool_adjusted[name]

    model_contrasts = {}
    model_names = list(config["models"])
    for mode in ("bare", "inference"):
        for left, right in combinations(model_names, 2):
            regret_values, cumulative_values = [], []
            for site in sites:
                rdiff, cdiff = [], []
                for gseed in gseeds:
                    right_row = cell.get((right, mode, site, gseed))
                    left_row = cell.get((left, mode, site, gseed))
                    if right_row is not None and left_row is not None:
                        rdiff.append(float(right_row["regret"] - left_row["regret"]))
                        cdiff.append(float(right_row["cumulative_regret"]
                                           - left_row["cumulative_regret"]))
                if rdiff:
                    regret_values.append(float(np.mean(rdiff)))
                    cumulative_values.append(float(np.mean(cdiff)))
            name = f"{mode}:{right}_minus_{left}"
            result = _contrast(regret_values, seed=bootstrap_seed + 2,
                               n_boot=bootstrap_resamples)
            result["cumulative_regret"] = _contrast(
                cumulative_values, seed=bootstrap_seed + 3,
                n_boot=bootstrap_resamples)
            model_contrasts[name] = result

    model_adjusted = adjust_bh([
        (name, value["p_raw_normal_approx"])
        for name, value in model_contrasts.items()
    ])
    for name, value in model_contrasts.items():
        value["p_bh"] = model_adjusted[name]

    return {
        "protocol_id": config["protocol_id"],
        "analysis_status": "prospective_post_primary",
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": bootstrap_resamples,
        "n_episode_rows_loaded": len(rows),
        "n_expected_identities": len(expected),
        "n_matched_identities": len(cell),
        "missing_identities": [list(item) for item in missing],
        "duplicate_identities": [list(item) for item in duplicates],
        "unexpected_rows": unexpected,
        "arms": arms,
        "tool_contrasts_bh_family": tool_contrasts,
        "model_contrasts_bh_family": model_contrasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs" / "phase5_model_task_extension.json")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    raw = args.config.read_bytes()
    config = json.loads(raw)
    if config.get("status") != "frozen_before_post_primary_execution":
        raise SystemExit("refusing to analyze a protocol that was not frozen before execution")
    rows = load_rows(args.root)
    report = analyze_rows(config, rows)
    report["protocol_sha256"] = hashlib.sha256(raw).hexdigest()
    if (report["missing_identities"] or report["duplicate_identities"]
            or report["unexpected_rows"]) and not args.allow_incomplete:
        raise SystemExit(
            "extension audit failed: "
            f"{len(report['missing_identities'])} missing, "
            f"{len(report['duplicate_identities'])} duplicate, "
            f"{len(report['unexpected_rows'])} unexpected"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({
        "protocol_id": report["protocol_id"],
        "protocol_sha256": report["protocol_sha256"],
        "n_episode_rows_loaded": report["n_episode_rows_loaded"],
        "n_matched_identities": report["n_matched_identities"],
        "missing": len(report["missing_identities"]),
        "duplicates": len(report["duplicate_identities"]),
        "unexpected": len(report["unexpected_rows"]),
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
