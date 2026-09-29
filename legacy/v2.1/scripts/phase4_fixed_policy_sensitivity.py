#!/usr/bin/env python3
"""Post-hoc fixed-policy external-validity analysis for Phase 4.

This script recovers the *physical* final recommendation from every frozen
primary transcript and evaluates that same recommendation in perturbed worlds.
It never calls an LLM and never lets a policy see observations from the changed
world.  The estimand is therefore recommendation robustness, not how the LLM
would adapt its experiments or inference under the perturbation.

The primary matrix contains two provider generation seeds at each site.  They
are averaged within site before uncertainty is calculated, so the site remains
the independent unit, matching the preregistered Phase 4 analysis.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import pathlib
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slowlab import SlowLabEnv, TASKS
from slowlab.llm import _extract_json
from slowlab.tomgro import TomgroParams
from slowlab.world import (CROP_LATENT_DIMENSION, MANAGEMENT_FACTORS,
                           ManagedTomgro, crop_params_from_latent)


PRICE_FIELDS = ("price_per_kg_fw", "elec_price", "heat_price", "co2_price")
TOOL_MODES = ("bare", "design", "inference", "both")
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_RESAMPLES = 10_000


@dataclass(frozen=True)
class RecommendationRecord:
    site: int
    generation_seed: int
    tool_mode: str
    physical: dict[str, float]
    recorded_regret: float
    transcript: str


def _final_recommendation(turns: list[dict]) -> tuple[str, dict[str, float]]:
    """Recover the value returned by LLMAgent.run, including early stopping."""
    for turn in reversed(turns):
        try:
            blob = _extract_json(turn["assistant"])
        except Exception:
            continue
        if "recommendation" in blob:
            return "recommendation", blob["recommendation"]
        if "current_best" in blob:
            return "current_best", blob["current_best"]
    raise ValueError("transcript has no parseable recommendation or current_best")


def load_primary_recommendations(root: pathlib.Path) -> tuple[list[RecommendationRecord], dict]:
    records: list[RecommendationRecord] = []
    source_counts: dict[str, int] = defaultdict(int)
    for episode_path in sorted((root / "primary").rglob("episodes_*.json")):
        tag = episode_path.stem.removeprefix("episodes_")
        for row in json.loads(episode_path.read_text()):
            if row.get("task") != "T3":
                continue
            site = int(row["seed"])
            transcript_path = episode_path.parent / f"transcript_{tag}_T3_s{site}.json"
            if not transcript_path.exists():
                raise FileNotFoundError(transcript_path)
            source, physical = _final_recommendation(
                json.loads(transcript_path.read_text()))
            source_counts[source] += 1
            records.append(RecommendationRecord(
                site=site,
                generation_seed=int(row["generation_seed"]),
                tool_mode=str(row["tool_mode"]),
                physical={key: float(value) for key, value in physical.items()},
                recorded_regret=float(row["regret"]),
                transcript=str(transcript_path),
            ))
    identities = {(r.tool_mode, r.site, r.generation_seed) for r in records}
    if len(identities) != len(records):
        raise ValueError("duplicate primary recommendation identity")
    return records, {"source_counts": dict(source_counts),
                     "n_records": len(records), "n_identities": len(identities)}


def _normal_pvalue(values: np.ndarray) -> float:
    if len(values) < 2 or values.std(ddof=1) == 0:
        return 1.0 if not len(values) or values.mean() == 0 else 0.0
    z = abs(values.mean() / (values.std(ddof=1) / np.sqrt(len(values))))
    return float(math.erfc(z / np.sqrt(2)))


def paired_summary(rows: list[dict], *, n_boot: int = BOOTSTRAP_RESAMPLES,
                   seed: int = BOOTSTRAP_SEED) -> dict:
    """Contrast each mode with bare after averaging generation seeds per site."""
    cells = {(row["tool_mode"], row["site"], row["generation_seed"]): row["regret"]
             for row in rows}
    sites = sorted({row["site"] for row in rows})
    gseeds = sorted({row["generation_seed"] for row in rows})
    out = {}
    for mode in TOOL_MODES[1:]:
        by_site = []
        for site in sites:
            diffs = [cells[(mode, site, gseed)] - cells[("bare", site, gseed)]
                     for gseed in gseeds
                     if (mode, site, gseed) in cells and ("bare", site, gseed) in cells]
            if len(diffs) == len(gseeds):
                by_site.append(float(np.mean(diffs)))
        values = np.asarray(by_site, float)
        rng = np.random.default_rng(seed)
        means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(1)
        out[f"{mode}_minus_bare"] = {
            "n_sites": len(values),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "ci95": [float(np.quantile(means, 0.025)),
                     float(np.quantile(means, 0.975))],
            "p_raw_normal_approx": _normal_pvalue(values),
            "point_direction": ("tool_better" if values.mean() < 0
                                else "tool_worse_or_equal"),
        }
    return out


def _site_latent(seed: int, *, fruit_span_fraction: float = 1.0) -> np.ndarray:
    """Recreate the frozen site draw, optionally contracting fruit spread.

    Fractions in [0, 1] stay inside the published J99 three-site envelope.  A
    value of zero collapses every fruit coordinate to the envelope midpoint;
    one exactly reproduces the frozen distribution.
    """
    if not 0.0 <= fruit_span_fraction <= 1.0:
        raise ValueError("fruit_span_fraction must lie in [0, 1]")
    rng = np.random.default_rng(20_000 + seed)
    vegetative = rng.normal(size=4)
    fruit_and_e = rng.random(CROP_LATENT_DIMENSION - len(vegetative))
    fruit_and_e = 0.5 + fruit_span_fraction * (fruit_and_e - 0.5)
    return np.concatenate([vegetative, fruit_and_e])


def physiological_params(seed: int, *, veg_cv: float = 0.06,
                         fruit_span_fraction: float = 1.0,
                         j99_site: str | None = None) -> TomgroParams:
    if j99_site is not None:
        return TomgroParams.from_j99(j99_site)
    params = crop_params_from_latent(
        _site_latent(seed, fruit_span_fraction=fruit_span_fraction),
        veg_cv=veg_cv)
    params.site = f"sensitivity#{seed}"
    return params


@contextmanager
def factor_width(factor_name: str | None = None, multiplier: float = 1.0):
    if factor_name is None:
        yield
        return
    spec = next(item for item in MANAGEMENT_FACTORS if item.name == factor_name)
    old_low, old_high = spec.low, spec.high
    midpoint = (old_low + old_high) / 2
    half_width = (old_high - old_low) * multiplier / 2
    spec.low, spec.high = midpoint - half_width, midpoint + half_width
    try:
        yield
    finally:
        spec.low, spec.high = old_low, old_high


def _world(task, site: int, scenario: dict) -> ManagedTomgro:
    base = SlowLabEnv(task, seed=site).truth
    econ = copy.deepcopy(base.econ)
    for field, multiplier in scenario.get("econ_multipliers", {}).items():
        setattr(econ, field, float(getattr(econ, field)) * float(multiplier))

    physiology = scenario.get("physiology")
    params = None
    if physiology is not None:
        params = physiological_params(site, **physiology)
    world = ManagedTomgro(seed=site, econ=econ, params=params,
                          factors=task.factors, cycle_days=task.cycle_days)
    # The disk key omits economics and parameter values.  Only the exact frozen
    # baseline may use the existing oracle cache.
    if (scenario.get("econ_multipliers") or physiology is not None
            or scenario.get("factor_width")):
        world._oracle_cacheable = False
    return world


def score_scenario(records: list[RecommendationRecord], scenario: dict) -> dict:
    task = TASKS["Optimise"]
    scored = []
    clipped = 0
    by_site: dict[int, list[RecommendationRecord]] = defaultdict(list)
    for record in records:
        by_site[record.site].append(record)

    width = scenario.get("factor_width", {})
    with factor_width(width.get("factor"), float(width.get("multiplier", 1.0))):
        factors = task.factors
        for site, site_records in sorted(by_site.items()):
            world = _world(task, site, scenario)
            _, best = world.oracle()
            points = []
            for record in site_records:
                raw = np.asarray([f.norm(record.physical[f.name]) for f in factors], float)
                clipped += int(np.any((raw < 0.0) | (raw > 1.0)))
                points.append(np.clip(raw, 0.0, 1.0))
            values = np.asarray(world(np.asarray(points)), float)
            for record, value in zip(site_records, values):
                scored.append({
                    "site": site,
                    "generation_seed": record.generation_seed,
                    "tool_mode": record.tool_mode,
                    "regret": float(best - value),
                })
    scenario_seed = BOOTSTRAP_SEED + int(hashlib.sha256(
        json.dumps(scenario, sort_keys=True).encode()).hexdigest()[:6], 16)
    return {
        "scenario": scenario,
        "n_recommendations": len(scored),
        "n_clipped_recommendations": clipped,
        "mean_regret": {
            mode: float(np.mean([row["regret"] for row in scored
                                if row["tool_mode"] == mode]))
            for mode in TOOL_MODES
        },
        "paired_contrasts": paired_summary(scored, seed=scenario_seed),
        "rows": scored,
    }


def scenario_catalogue() -> dict[str, dict[str, dict]]:
    return {
        "frozen_baseline": {
            "baseline": {"label": "frozen environment"},
        },
        "price_basket": {
            str(mult): {"label": f"price basket x{mult}",
                        "econ_multipliers": {field: mult for field in PRICE_FIELDS}}
            for mult in (0.85, 1.0, 1.15)
        },
        "ambient_light": {
            str(mult): {"label": f"ambient PAR x{mult}",
                        "econ_multipliers": {"par_ambient": mult}}
            for mult in (0.7, 1.0, 1.3)
        },
        "vegetative_dispersion": {
            str(cv): {"label": f"vegetative log-SD {cv}",
                      "physiology": {"veg_cv": cv, "fruit_span_fraction": 1.0}}
            for cv in (0.0, 0.06, 0.10)
        },
        "fruit_parameter_span": {
            str(span): {"label": f"J99 fruit span fraction {span}",
                        "physiology": {"veg_cv": 0.06,
                                       "fruit_span_fraction": span}}
            for span in (0.0, 0.5, 1.0)
        },
        "j99_site_calibration": {
            site: {"label": f"fixed J99 {site} calibration",
                   "physiology": {"j99_site": site}}
            for site in ("Gainesville", "LakeCity", "Avignon")
        },
        "day_temp_range": {
            str(mult): {"label": f"day_temp width x{mult}",
                        "factor_width": {"factor": "day_temp", "multiplier": mult}}
            for mult in (0.8, 1.0, 1.2)
        },
        "density_range": {
            str(mult): {"label": f"density width x{mult}",
                        "factor_width": {"factor": "density", "multiplier": mult}}
            for mult in (0.8, 1.0, 1.2)
        },
    }


def canonical_scenario(scenario: dict) -> str:
    """Deduplicate scenarios that are exactly the frozen environment."""
    substantive = {key: value for key, value in scenario.items() if key != "label"}
    econ = substantive.get("econ_multipliers")
    if econ is not None and all(float(value) == 1.0 for value in econ.values()):
        substantive = {}
    physiology = substantive.get("physiology")
    if physiology == {"veg_cv": 0.06, "fruit_span_fraction": 1.0}:
        substantive = {}
    width = substantive.get("factor_width")
    if width is not None and float(width.get("multiplier", 1.0)) == 1.0:
        substantive = {}
    return json.dumps(substantive, sort_keys=True, separators=(",", ":"))


def add_holm_adjustment(results: dict) -> dict:
    """Adjust the unique scenario-by-contrast family and annotate all aliases."""
    unique = {}
    for scenarios in results.values():
        for result in scenarios.values():
            scenario_key = canonical_scenario(result["scenario"])
            for contrast, values in result["paired_contrasts"].items():
                unique[(scenario_key, contrast)] = float(values["p_raw_normal_approx"])
    ordered = sorted(unique.items(), key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, (identity, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p_value))
        adjusted[identity] = running
    for scenarios in results.values():
        for result in scenarios.values():
            scenario_key = canonical_scenario(result["scenario"])
            for contrast, values in result["paired_contrasts"].items():
                values["p_holm_post_hoc_sensitivity_family"] = adjusted[
                    (scenario_key, contrast)]
    return {
        "method": "Holm family-wise error control",
        "family": "all unique fixed-policy scenario-by-tool contrasts",
        "n_tests": total,
        "equivalent frozen-baseline aliases_counted_once": True,
    }


def _base_reconstruction(records: list[RecommendationRecord]) -> dict:
    result = score_scenario(records, {"label": "frozen environment"})
    keyed = {(row["tool_mode"], row["site"], row["generation_seed"]): row["regret"]
             for row in result.pop("rows")}
    errors = [abs(keyed[(r.tool_mode, r.site, r.generation_seed)] - r.recorded_regret)
              for r in records]
    return {"n_checked": len(errors), "max_abs_error": float(max(errors, default=0.0)),
            "exact_matches": int(sum(error == 0.0 for error in errors)),
            "result": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path,
                        default=pathlib.Path("/Users/dingningpei/Desktop/slowlab/output/reviews"
                                             "/phase4_confirmatory"))
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("/Users/dingningpei/Desktop/slowlab/output/reviews"
                                             "/phase4_fixed_policy_sensitivity.json"))
    parser.add_argument("--families", nargs="*", default=None,
                        help="optional subset of scenario family names")
    parser.add_argument("--sites", type=int, default=None,
                        help="development-only prefix of sites; omit for the full matrix")
    args = parser.parse_args()

    records, load_audit = load_primary_recommendations(args.root)
    if args.sites is not None:
        keep = set(sorted({record.site for record in records})[:args.sites])
        records = [record for record in records if record.site in keep]
    reconstruction = _base_reconstruction(records)
    if reconstruction["max_abs_error"] > 1e-12:
        raise SystemExit("frozen-environment reconstruction failed: "
                         f"max error {reconstruction['max_abs_error']}")

    catalogue = scenario_catalogue()
    family_names = args.families or list(catalogue)
    unknown = sorted(set(family_names) - set(catalogue))
    if unknown:
        raise SystemExit(f"unknown scenario families: {unknown}")
    results = {}
    result_cache = {canonical_scenario({}): reconstruction["result"]}
    for family in family_names:
        results[family] = {}
        for name, scenario in catalogue[family].items():
            print(f"{family}/{name}", flush=True)
            key = canonical_scenario(scenario)
            if key in result_cache:
                result = copy.deepcopy(result_cache[key])
                result["scenario"] = scenario
                result["reused_equivalent_scenario"] = True
            else:
                result = score_scenario(records, scenario)
                result.pop("rows")
                result_cache[key] = copy.deepcopy(result)
            results[family][name] = result
    multiplicity = add_holm_adjustment(results)

    report = {
        "analysis_family": "post_hoc_fixed_policy_external_validity",
        "estimand": ("Regret of each frozen Phase 4 primary recommendation when the same "
                     "physical action is applied to a perturbed environment."),
        "interpretation_limit": ("The LLM never sees observations from the perturbed world. "
                                 "This measures recommendation robustness, not adaptive policy "
                                 "robustness and not a confirmatory rerun."),
        "scenario_range_status": ("The multipliers are documented scenario ranges, not "
                                  "probability or credible intervals. The reported 95% intervals "
                                  "are bootstrap uncertainty intervals for paired site effects."),
        "noise_scale_limit": ("Fixed-policy final simple regret depends on the response mean, "
                              "not observation noise. Noise robustness requires a new adaptive "
                              "LLM run and is not inferred here."),
        "factor_range_action_rule": ("A frozen physical recommendation outside a narrowed "
                                     "action range is projected coordinatewise to the nearest "
                                     "feasible boundary. n_clipped_recommendations reports how "
                                     "often this fallback changes the action."),
        "input_audit": load_audit,
        "reconstruction_audit": reconstruction,
        "bootstrap": {"unit": "site after averaging generation seeds",
                      "seed_base": BOOTSTRAP_SEED,
                      "resamples": BOOTSTRAP_RESAMPLES},
        "multiplicity": multiplicity,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"input_audit": load_audit,
                      "reconstruction_audit": {
                          key: value for key, value in reconstruction.items()
                          if key != "result"},
                      "families": family_names}, indent=2))


if __name__ == "__main__":
    main()
