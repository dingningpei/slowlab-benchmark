#!/usr/bin/env python3
"""Measure the v1 component-field bypass on recovered execution histories.

For each completed episode this script fits the same small GP twice to the same
executed design: once with the reported margin and once with
``rev_rate - energy_cost_rate - other_cost_rate``.  In environment v1 the
second target omitted chamber, loop and batch nuisance effects.  It also
compares Transfer recommendations made from the old clean revenue field with
the v2-consistent revenue field.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab.agents import GP, _cands
from slowlab.tasks import TASKS
from slowlab.world import ManagedTomgro


def _episode(path: pathlib.Path):
    doc = json.loads(path.read_text())
    designs = {}
    observations = []
    for event in doc["events"]:
        if event["kind"] == "accepted":
            designs[int(event["design_id"])] = event["payload"]
        elif event["kind"] == "completed":
            observations.extend(event["payload"]["observations"])
    if not observations:
        return None
    task = TASKS[doc["task"]]
    X, rows = [], []
    for obs in observations:
        design = designs[int(obs["design_id"])]
        treatment = design["treatments"][str(obs["treatment"])]
        X.append([float(treatment[f.name]) for f in task.factors])
        direct = float(obs["value"])
        revenue = float(obs["rev_rate"])
        energy = float(obs["energy_cost_rate"])
        other = float(obs["other_cost_rate"])
        reconstructed = revenue - energy - other
        unit = str(obs["unit_id"])
        rows.append({
            "episode": doc["episode_id"], "design": int(obs["design_id"]),
            "treatment": str(obs["treatment"]), "unit": unit,
            "chamber": unit.split("l", 1)[0], "direct": direct,
            "reconstructed": reconstructed, "delta": direct - reconstructed,
            "revenue": revenue, "energy": energy, "other": other,
        })
    return doc, task, np.asarray(X, float), rows


def _recommend(X, y, d, candidates):
    if len(y) == 0:
        return np.full(d, 0.5)
    gp = GP(noise=0.15).fit(X, np.asarray(y, float))
    return candidates[int(np.argmax(gp.predict(candidates)[0]))]


def _regret(world, point, best, *, shock=None):
    if shock is None:
        return float(best - world(point.reshape(1, -1))[0])
    return float(best - world.profit_at(point.reshape(1, -1), shock)[0])


def _within_treatment_variance(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["episode"], row["design"], row["treatment"])].append(row[key])
    squares, dof = 0.0, 0
    for values in grouped.values():
        if len(values) > 1:
            a = np.asarray(values, float)
            squares += float(np.sum((a - a.mean()) ** 2))
            dof += len(a) - 1
    return squares / dof if dof else float("nan")


def _nuisance_correlations(rows):
    """Method-of-moments correlations implied by repeated nuisance deltas."""
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[row["episode"]].append(row)
    delta = np.asarray([r["delta"] for r in rows], float)
    variance = float(np.mean((delta - delta.mean()) ** 2))
    products = defaultdict(list)
    for episode_rows in by_episode.values():
        for i, a in enumerate(episode_rows):
            for b in episode_rows[i + 1:]:
                same_round = a["design"] == b["design"]
                same_chamber = a["chamber"] == b["chamber"]
                same_unit = a["unit"] == b["unit"]
                if same_unit and not same_round:
                    category = "same_unit_across_rounds"
                elif same_chamber and not same_round:
                    category = "same_chamber_across_rounds"
                elif same_round and not same_chamber:
                    category = "same_round_across_chambers"
                elif not same_round and not same_chamber:
                    category = "different_round_and_chamber"
                else:
                    continue
                products[category].append(a["delta"] * b["delta"])
    return {
        category: {
            "n_pairs": len(values),
            "covariance": float(np.mean(values)) if values else float("nan"),
            "correlation": float(np.mean(values) / variance) if values and variance else float("nan"),
        }
        for category, values in products.items()
    }


def _paired_summary(a, b, *, label_a="a", label_b="b"):
    a, b = np.asarray(a, float), np.asarray(b, float)
    delta = a - b
    n = len(delta)
    return {
        f"{label_a}_mean": float(a.mean()),
        f"{label_b}_mean": float(b.mean()),
        f"{label_a}_minus_{label_b}_mean": float(delta.mean()),
        "paired_se": float(delta.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        "paired_median": float(np.median(delta)),
        f"{label_a}_better_fraction": float(np.mean(a < b)),
        "different_fraction": float(np.mean(np.abs(delta) > 1e-12)),
    }


def run(events_dir: pathlib.Path, n_candidates: int = 8000):
    files = sorted(events_dir.glob("events_*.json"))
    candidates = {}
    best_cache = {}
    transfer_best_cache = {}
    episode_rows, all_rows = [], []
    for path in files:
        parsed = _episode(path)
        if parsed is None:
            continue
        doc, task, X, rows = parsed
        d = task.d
        C = candidates.setdefault(d, _cands(d, n_candidates, np.random.default_rng(7)))
        direct = np.asarray([r["direct"] for r in rows])
        recon = np.asarray([r["reconstructed"] for r in rows])
        p_direct = _recommend(X, direct, d, C)
        p_recon = _recommend(X, recon, d, C)
        world = ManagedTomgro(seed=int(doc["seed"]), factors=task.factors,
                              cycle_days=task.cycle_days)
        world_key = (task.name, int(doc["seed"]))
        if world_key not in best_cache:
            best_cache[world_key] = float(world.oracle()[1])
        result = {
            "episode": doc["episode_id"], "model": doc["model"],
            "task": doc["task"], "seed": int(doc["seed"]),
            "n_observations": len(rows),
            "direct_regret": _regret(world, p_direct, best_cache[world_key]),
            "reconstructed_regret": _regret(world, p_recon, best_cache[world_key]),
        }
        if task.transfer_energy_shock is not None:
            if world_key not in transfer_best_cache:
                transfer_best_cache[world_key] = float(
                    world.oracle_at(task.transfer_energy_shock)[1])
            revenue_clean = np.asarray([r["revenue"] for r in rows])
            revenue_corrected = revenue_clean + np.asarray([r["delta"] for r in rows])
            energy = np.asarray([r["energy"] for r in rows])
            other = np.asarray([r["other"] for r in rows])
            ratio = float(task.transfer_energy_shock / world.econ.energy_shock)
            ge = GP(noise=.02).fit(X, energy).predict(C)[0]
            go = GP(noise=.02).fit(X, other).predict(C)[0]
            clean_objective = (GP(noise=.15).fit(X, revenue_clean).predict(C)[0]
                               - ratio * ge - go)
            corrected_objective = (GP(noise=.15).fit(X, revenue_corrected).predict(C)[0]
                                   - ratio * ge - go)
            p_clean = C[int(np.argmax(clean_objective))]
            p_corrected = C[int(np.argmax(corrected_objective))]
            result["transfer_regret_clean_components"] = _regret(
                world, p_clean, transfer_best_cache[world_key],
                shock=task.transfer_energy_shock)
            result["transfer_regret_corrected_components"] = _regret(
                world, p_corrected, transfer_best_cache[world_key],
                shock=task.transfer_energy_shock)
        episode_rows.append(result)
        all_rows.extend(rows)

    by_task = {}
    for task_name in sorted({r["task"] for r in episode_rows}):
        subset = [r for r in episode_rows if r["task"] == task_name]
        direct = np.asarray([r["direct_regret"] for r in subset])
        recon = np.asarray([r["reconstructed_regret"] for r in subset])
        item = {"n": len(subset), **_paired_summary(
            recon, direct, label_a="reconstructed_regret", label_b="direct_regret")}
        if "transfer_regret_clean_components" in subset[0]:
            clean = np.asarray([r["transfer_regret_clean_components"] for r in subset])
            corrected = np.asarray([r["transfer_regret_corrected_components"] for r in subset])
            item.update({"transfer": _paired_summary(
                clean, corrected, label_a="clean_components", label_b="corrected_components")})
        by_task[task_name] = item

    by_model = {}
    for model in sorted({r["model"] for r in episode_rows}):
        subset = [r for r in episode_rows if r["model"] == model]
        by_model[model] = {
            "n": len(subset),
            **_paired_summary(
                [r["reconstructed_regret"] for r in subset],
                [r["direct_regret"] for r in subset],
                label_a="reconstructed_regret", label_b="direct_regret"),
        }

    delta = np.asarray([r["delta"] for r in all_rows], float)
    return {
        "input": str(events_dir), "n_candidates": int(n_candidates),
        "n_event_files": len(files),
        "n_completed_episodes": len(episode_rows), "n_observations": len(all_rows),
        "observation_bypass": {
            "nuisance_delta_mean": float(delta.mean()),
            "nuisance_delta_sd": float(delta.std(ddof=1)),
            "within_treatment_variance_direct": _within_treatment_variance(all_rows, "direct"),
            "within_treatment_variance_reconstructed": _within_treatment_variance(all_rows, "reconstructed"),
            "variance_ratio_reconstructed_over_direct": (
                _within_treatment_variance(all_rows, "reconstructed")
                / _within_treatment_variance(all_rows, "direct")),
            "block_correlations": _nuisance_correlations(all_rows),
        },
        "by_task": by_task, "by_model": by_model,
        "episodes": episode_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("events_dir", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--n-candidates", type=int, default=8000)
    args = parser.parse_args()
    result = run(args.events_dir, args.n_candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
