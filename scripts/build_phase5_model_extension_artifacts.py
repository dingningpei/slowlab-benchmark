#!/usr/bin/env python3
"""Publish aggregate A3 results without exposing private transcripts or site rows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_phase5_model_extension import rows_and_cost  # noqa: E402


def compact_contrast(value: dict) -> dict:
    out = {key: value[key] for key in (
        "n_sites", "mean", "median", "sd", "standardized_effect", "ci95",
        "p_raw_normal_approx", "p_bh")}
    out["cumulative_regret"] = {
        key: item for key, item in value["cumulative_regret"].items()
        if key != "site_values"
    }
    return out


def signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def interval(values: list[float], digits: int = 4) -> str:
    return f"[{signed(values[0], digits)}, {signed(values[1], digits)}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs" / "phase5_model_task_extension_a3.json")
    parser.add_argument("--execution-root", type=Path, required=True,
                        help="A3 root containing only DeepSeek and Qwen formal episodes")
    parser.add_argument("--summary-out", type=Path,
                        default=ROOT / "results" / "phase5_model_extension_summary_env2.1.0.json")
    parser.add_argument("--table-out", type=Path,
                        default=ROOT / "paper" / "sections" / "_phase5_model_extension_table.tex")
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text())
    config = json.loads(args.config.read_text())
    _, a3_cost = rows_and_cost(args.execution_root, config)
    prior = float(config["cost_plan_usd"]["prior_recorded_spend"])

    summary = {
        "environment_version": "2.1.0",
        "analysis_status": "prospective_post_primary",
        "protocol_id": analysis["protocol_id"],
        "protocol_sha256": analysis["protocol_sha256"],
        "analysis_id": "slowlab-v2-phase5-model-extension-analysis-2026-09-20-a3",
        "bootstrap_seed": analysis["bootstrap_seed"],
        "bootstrap_resamples": analysis["bootstrap_resamples"],
        "episodes": {
            "expected": analysis["n_expected_identities"],
            "matched": analysis["n_matched_identities"],
            "missing": len(analysis["missing_identities"]),
            "duplicates": len(analysis["duplicate_identities"]),
            "unexpected": len(analysis["unexpected_rows"]),
        },
        "cost_audit_usd": {
            "a3_deepseek_and_qwen": a3_cost,
            "prior_valid_and_excluded_runs": prior,
            "all_recorded_spend": prior + a3_cost,
            "hard_cap": config["cost_plan_usd"]["hard_pause_before"],
            "direct_deepseek_policy": "peak cache-miss input and peak output rate upper bound",
        },
        "execution_deviations": [
            "The 22-episode DeepSeek OpenRouter pilot and four-episode direct low-thinking pilot were excluded before effect inspection.",
            "DeepSeek formal runs used the direct API, native JSON mode, and non-thinking mode after non-reserved interface diagnostics.",
            "Qwen requested and actual model IDs matched, but OpenRouter used multiple underlying providers.",
            "Provider-hidden reasoning was disabled for all formal A3 model conditions; the already completed Luna cells used the same setting.",
        ],
        "arms": analysis["arms"],
        "tool_contrasts_bh_family": {
            key: compact_contrast(value)
            for key, value in analysis["tool_contrasts_bh_family"].items()
        },
        "model_contrasts_bh_family": {
            key: compact_contrast(value)
            for key, value in analysis["model_contrasts_bh_family"].items()
        },
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")

    labels = {"luna": "Luna", "deepseek": "DeepSeek V4.1 Flash", "qwen": "Qwen 3.8 27B"}
    rows = []
    for model in ("luna", "deepseek", "qwen"):
        bare = analysis["arms"][f"{model}|bare"]["final_simple_regret"]["mean"]
        inference = analysis["arms"][f"{model}|inference"]["final_simple_regret"]["mean"]
        effect = analysis["tool_contrasts_bh_family"][f"{model}:inference_minus_bare"]
        p = effect["p_bh"]
        p_text = "$<0.001$" if p < 0.001 else f"{p:.3f}"
        rows.append(
            f"{labels[model]} & {bare:.4f} & {inference:.4f} & "
            f"{signed(effect['mean'])} {interval(effect['ci95'])} & {p_text} \\\\")
    table = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Bare & Inference & $\Delta$ [95\% CI] & BH $p$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Prospective post-primary model extension on \textsc{Optimise}. Values are mean final simple regret (lower is better); $\Delta$ is inference minus bare. Each arm contains 24 common sites and two generation seeds, averaged within site before the 10,000-resample bootstrap. BH correction covers the three within-model contrasts.}",
        r"\label{tab:phase5-model-extension}",
        r"\end{table}",
        "",
    ])
    args.table_out.parent.mkdir(parents=True, exist_ok=True)
    args.table_out.write_text(table)
    print(json.dumps({
        "summary": str(args.summary_out), "table": str(args.table_out),
        "episodes": summary["episodes"],
        "all_recorded_spend": summary["cost_audit_usd"]["all_recorded_spend"],
    }, indent=2))


if __name__ == "__main__":
    main()
