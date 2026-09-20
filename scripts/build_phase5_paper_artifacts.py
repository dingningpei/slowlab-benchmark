#!/usr/bin/env python3
"""Build the public Phase 5 summary and LaTeX tables from frozen analyses.

The input directory is deliberately gitignored: it may contain raw transcripts and
internal review notes.  This script reads only named JSON analysis products and writes
the compact, numerical summary that supports the paper.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 20260913
BOOTSTRAP_RESAMPLES = 10_000
ROW_END = r"\\"


def read_json(path: Path):
    return json.loads(path.read_text())


def bootstrap_mean(values: list[float]) -> list[float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = x[rng.integers(0, len(x), (BOOTSTRAP_RESAMPLES, len(x)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def mean_se(values: list[float]) -> dict[str, float | int | list[float]]:
    x = np.asarray(values, dtype=float)
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "se": float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else 0.0,
        "ci95": bootstrap_mean(values),
    }


def phase4_rows(analysis_dir: Path, matrix: str) -> list[dict]:
    rows: list[dict] = []
    folder = analysis_dir / "phase4_confirmatory" / matrix
    for path in sorted(folder.rglob("episodes_*.json")):
        rows.extend(read_json(path))
    return rows


def grouped_episode_summary(rows: list[dict], *, task: str) -> dict[str, dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["task"] == task:
            grouped[(row["model"], row["tool_mode"])].append(row)
    out = {}
    for (model, mode), cell in grouped.items():
        out[f"{model}|{mode}"] = {
            "model": model,
            "tool_mode": mode,
            "n_episodes": len(cell),
            "n_sites": len({int(row["seed"]) for row in cell}),
            "generation_seeds": sorted({int(row["generation_seed"]) for row in cell}),
            "final_simple_regret": mean_se([float(row["regret"]) for row in cell]),
            "cumulative_regret": mean_se([float(row["cumulative_regret"]) for row in cell]),
            "campaign_cash": mean_se([float(row["cash"]) for row in cell]),
            "infeasible_submissions": int(sum(int(row["infeasible"]) for row in cell)),
        }
    return out


def scripted_t3(raw_dir: Path) -> dict[str, dict]:
    report = read_json(raw_dir / "phase3_design_reader_replay.json")
    rows = [row for row in report["rows"] if row["task"] == "T3" and
            row["reader"] == "ordinary_gp" and not row["privileged"]]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["design"]].append(float(row["regret"]))
    return {name: mean_se(values) for name, values in grouped.items()}


def transfer_summary(analysis_dir: Path) -> dict:
    rows = [row for row in phase4_rows(analysis_dir, "task_heterogeneity")
            if row["task"] == "T4"]
    arms: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        arms[row["tool_mode"]].append(row)
    arm_summary = {}
    for mode, cell in arms.items():
        arm_summary[mode] = {
            "n_episodes": len(cell),
            "n_sites": len({int(row["seed"]) for row in cell}),
            "pre_shock_regret": mean_se([float(row["regret"]) for row in cell]),
            "post_shock_regret": mean_se([float(row["transfer_regret"]) for row in cell]),
        }
    cell = {(row["tool_mode"], int(row["seed"]), int(row["generation_seed"])): row
            for row in rows}
    by_site: dict[int, list[float]] = defaultdict(list)
    for (_, site, gseed), row in cell.items():
        if row["tool_mode"] != "inference":
            continue
        bare = cell[("bare", site, gseed)]
        by_site[site].append(float(row["transfer_regret"] - bare["transfer_regret"]))
    differences = [float(np.mean(values)) for values in by_site.values()]
    return {
        "arms": arm_summary,
        "post_shock_inference_minus_bare": {
            **mean_se(differences),
            "analysis_status": "descriptive_not_preregistered_for_inference",
        },
    }


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def interval(cell: dict) -> str:
    lo, hi = cell["ci95"]
    return f"[{lo:+.4f}, {hi:+.4f}]"


def write_main_table(path: Path, summary: dict) -> None:
    scripted = summary["scripted_t3_fixed_history"]
    primary = summary["phase4_t3_episodes"]
    contrasts = summary["primary_tool_contrasts"]
    strong = summary["strong_model_contrast"]
    names = {
        "random_spread": "Random spread",
        "split_plot_doe": "Split-plot DOE",
        "constraint_aware_batch_bo": "Batch BO",
    }
    lines = [
        r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{@{}llrrll@{}}", r"\toprule",
        r"Policy & Reader / aid & Sites $\times$ repeats & Mean regret & $\Delta$ vs. bare [95\% CI] & adjusted $p$ " + ROW_END,
        r"\midrule",
    ]
    for key in ("random_spread", "split_plot_doe", "constraint_aware_batch_bo"):
        cell = scripted[key]
        lines.append(f"{names[key]} & ordinary GP & {cell['n']} $\\times$ 1 & {fmt(cell['mean'])} & descriptive & --- " + ROW_END)
    lines.append(r"\addlinespace")
    luna = "openai/gpt-5.6-luna"
    labels = {"bare": "Luna", "design": "Luna", "inference": "Luna", "both": "Luna"}
    aids = {"bare": "none", "design": "design", "inference": "inference", "both": "both"}
    for mode in ("bare", "design", "inference", "both"):
        cell = primary[f"{luna}|{mode}"]
        if mode == "bare":
            delta, p = "reference", "---"
        else:
            con = contrasts[f"{mode}_minus_bare"]
            delta, p = f"{con['mean']:+.4f} {interval(con)}", f"{con['p_holm']:.3f}"
        lines.append(f"{labels[mode]} & {aids[mode]} & {cell['n_sites']} $\\times$ 2 & "
                     f"{fmt(cell['final_simple_regret']['mean'])} & {delta} & {p} " + ROW_END)
    lines.append(r"\addlinespace")
    sol = "openai/gpt-5.6-sol"
    for mode in ("bare", "inference"):
        cell = summary["phase4_t3_strong_episodes"][f"{sol}|{mode}"]
        if mode == "bare":
            delta, p = "reference", "---"
        else:
            con = strong
            delta, p = f"{con['mean']:+.4f} {interval(con)}", f"{con['p_bh']:.3f}"
        lines.append(f"Sol & {'none' if mode == 'bare' else 'inference'} & {cell['n_sites']} $\\times$ 2 & "
                     f"{fmt(cell['final_simple_regret']['mean'])} & {delta} & {p} " + ROW_END)
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Final simple regret on \textsc{Optimise} (lower is better). Scripted rows are fixed-history reader replays on eight Phase~3 sites and are descriptive; they are included as reference points, not tested against the independently sampled model runs. Phase~4 model contrasts average two provider generations within each site before inference. Luna tool contrasts use Holm correction across the three preregistered comparisons; the Sol replication uses the prespecified secondary-family adjustment.}",
        r"\label{tab:phase4-main}", r"\end{table*}", "",
    ])
    path.write_text("\n".join(lines))


def write_secondary_table(path: Path, summary: dict) -> None:
    rows = []
    for label, cell, correction in [
        ("Luna, T1: inference $-$ bare", summary["task_heterogeneity"]["T1:inference_minus_bare"], "BH"),
        ("Luna, T4: inference $-$ bare", summary["task_heterogeneity"]["T4:inference_minus_bare"], "BH"),
        ("Luna, within-cycle $-$ terminal-only", summary["within_cycle"], "BH"),
        ("Luna, 0.5$\\times$ noise: design $-$ bare", summary["noise_0_5x_design_minus_bare"], "BH"),
        ("Luna, 2$\\times$ noise: design $-$ bare", summary["noise_2x"]["design_minus_bare"], "Holm"),
        ("Luna, 2$\\times$ noise: inference $-$ bare", summary["noise_2x"]["inference_minus_bare"], "Holm"),
        ("Luna, 2$\\times$ noise: both $-$ bare", summary["noise_2x"]["both_minus_bare"], "Holm"),
    ]:
        p = cell["p_holm"] if correction == "Holm" else cell["p_bh"]
        rows.append(f"{label} & {cell['n_sites']} & {cell['mean']:+.4f} & {interval(cell)} & {p:.3f} ({correction}) " + ROW_END)
    lines = [
        r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrl@{}}", r"\toprule",
        r"Contrast & sites & $\Delta$ regret & 95\% CI & adjusted $p$ " + ROW_END, r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}",
        r"\caption{Held-out secondary and robustness contrasts (positive values are worse). The half-noise design-aid contrast is the only listed interval that excludes zero.}",
        r"\label{tab:phase4-secondary}", r"\end{table}", "",
    ]
    path.write_text("\n".join(lines))


def write_transfer_table(path: Path, summary: dict) -> None:
    arms = summary["transfer"]["arms"]
    diff = summary["transfer"]["post_shock_inference_minus_bare"]
    lines = [
        r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
        r"Condition & sites $\times$ repeats & pre-shock & post-shock " + ROW_END, r"\midrule",
    ]
    for mode in ("bare", "inference"):
        cell = arms[mode]
        lines.append(f"Luna {mode} & {cell['n_sites']} $\\times$ 2 & "
                     f"{cell['pre_shock_regret']['mean']:.4f} $\\pm$ {cell['pre_shock_regret']['se']:.4f} & "
                     f"{cell['post_shock_regret']['mean']:.4f} $\\pm$ {cell['post_shock_regret']['se']:.4f} " + ROW_END)
    lines.extend([
        r"\midrule",
        f"Inference $-$ bare (post-shock) & {diff['n']} sites & \\multicolumn{{2}}{{r}}{{{diff['mean']:+.4f} [{diff['ci95'][0]:+.4f}, {diff['ci95'][1]:+.4f}]}} " + ROW_END,
        r"\bottomrule", r"\end{tabular}",
        r"\caption{\textsc{Transfer} regret before and after the held-out $2.2\times$ energy-price shock (mean $\pm$ site-level SE). The post-shock paired interval is descriptive because this contrast was not a preregistered inferential family.}",
        r"\label{tab:transfer-v2}", r"\end{table}", "",
    ])
    path.write_text("\n".join(lines))


def build(analysis_dir: Path, raw_dir: Path) -> dict:
    primary = read_json(analysis_dir / "phase4_primary_analysis.json")
    strong = read_json(analysis_dir / "phase4_strong_model_analysis.json")
    hetero = read_json(analysis_dir / "phase4_task_heterogeneity_analysis.json")
    within = read_json(analysis_dir / "phase4_within_cycle_analysis.json")
    noise = read_json(analysis_dir / "phase4_noise_robustness_analysis.json")
    return {
        "environment_version": "2.0.0",
        "generated_from": {
            "phase3_protocol": "phase3-design-reader-replay-0.1",
            "phase4_protocol": primary["protocol_id"],
            "noise_protocol": noise["protocol_id"],
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "scripted_t3_fixed_history": scripted_t3(raw_dir),
        "phase4_t3_episodes": grouped_episode_summary(phase4_rows(raw_dir, "primary"), task="T3"),
        "primary_tool_contrasts": primary["primary"]["contrasts"],
        "phase4_t3_strong_episodes": grouped_episode_summary(phase4_rows(raw_dir, "strong_model_replication"), task="T3"),
        "strong_model_contrast": strong["result"]["contrasts"]["inference_minus_bare"],
        "task_heterogeneity": hetero["result"]["contrasts"],
        "within_cycle": within["result"]["contrasts"]["within_cycle_minus_terminal_only"],
        "noise_2x": noise["contrasts_by_multiplier"]["2.0"],
        "noise_0_5x_design_minus_bare": noise["contrasts_by_multiplier"]["0.5"]["design_minus_bare"],
        "transfer": transfer_summary(raw_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path,
                        help="raw episode and Phase 3 directory (defaults to analysis-dir)")
    parser.add_argument("--summary-out", type=Path,
                        default=ROOT / "results" / "phase5_paper_summary_env2.0.0.json")
    parser.add_argument("--sections-dir", type=Path,
                        default=ROOT / "paper" / "sections")
    args = parser.parse_args()
    summary = build(args.analysis_dir, args.raw_dir or args.analysis_dir)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    write_main_table(args.sections_dir / "_phase4_main_table.tex", summary)
    write_secondary_table(args.sections_dir / "_phase4_secondary_table.tex", summary)
    write_transfer_table(args.sections_dir / "_phase4_transfer_table.tex", summary)
    print(f"wrote {args.summary_out} and three LaTeX tables")


if __name__ == "__main__":
    main()
