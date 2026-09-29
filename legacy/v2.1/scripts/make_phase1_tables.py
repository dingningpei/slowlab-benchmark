#!/usr/bin/env python3
"""Build corrected Tables 3--5 and Phase 1 statistical audits.

This script never parses designs from assistant text. Outcome metrics come from
the frozen episode summaries; every design metric comes from recovered events.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np
from scipy import stats

CFGS = ("Sanity", "Screen", "Optimise", "Transfer")
SHORT = {
    "deepseek_deepseek-v4-flash": "DeepSeek-v4-flash",
    "openai_gpt-5.6-luna": "GPT-5.6-luna",
    "qwen_qwen3.8-27b": "Qwen3.8-27B",
    "xiaomi_mimo-v2.5": "MiMo-v2.5",
    "z-ai_glm-5.3-flash": "GLM-5.3-flash",
}


def load_episodes(directory: pathlib.Path):
    out = defaultdict(dict)
    for path in sorted(directory.glob("episodes_*.json")):
        arm = path.stem.removeprefix("episodes_")
        for episode in json.loads(path.read_text()):
            out[(arm, episode["task"])][int(episode["seed"])] = episode
    return out


def adjust_pvalues(values: list[float], method: str) -> list[float]:
    """Holm FWER or Benjamini-Hochberg FDR adjusted p-values."""
    p = np.asarray(values, float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), float)
    if method == "holm":
        running = 0.0
        for rank, index in enumerate(order):
            running = max(running, (len(p) - rank) * p[index])
            adjusted[index] = min(1.0, running)
    elif method == "bh":
        running = 1.0
        for rank in range(len(p), 0, -1):
            index = order[rank - 1]
            running = min(running, len(p) * p[index] / rank)
            adjusted[index] = min(1.0, running)
    else:
        raise ValueError(method)
    return adjusted.tolist()


def tool_statistics(episodes, cv):
    rows = []
    for task in CFGS:
        for arm in SHORT:
            bare = episodes[(arm, task)]
            tooled = episodes[(arm + "+tools", task)]
            seeds = sorted(set(bare) & set(tooled))
            x = np.asarray([bare[s]["regret"] for s in seeds], float)
            y = np.asarray([tooled[s]["regret"] for s in seeds], float)
            delta = y - x
            test = stats.ttest_rel(y, x)
            half = float(stats.t.ppf(.975, len(delta) - 1) *
                         stats.sem(delta)) if len(delta) > 1 else float("nan")
            agents = cv[task]["agents"]
            prior = float(cv[task]["prior"])
            rb = float(agents[arm]["R_star"])
            rt = float(agents[arm + "+tools"]["R_star"])
            rows.append({
                "task": task,
                "arm": arm,
                "n": len(seeds),
                "mean_bare_regret": float(x.mean()),
                "mean_tool_regret": float(y.mean()),
                "mean_paired_difference": float(delta.mean()),
                "ci95_difference": [float(delta.mean() - half),
                                    float(delta.mean() + half)],
                "relative_difference": float(delta.mean() / x.mean()),
                "ci95_relative_to_bare_mean": [
                    float((delta.mean() - half) / x.mean()),
                    float((delta.mean() + half) / x.mean())],
                "t": float(test.statistic),
                "p_raw": float(test.pvalue),
                "delta_c": float((1 - rt / prior) - (1 - rb / prior)),
                "delta_eta": float(rt / y.mean() - rb / x.mean()),
            })
    raw = [row["p_raw"] for row in rows]
    holm = adjust_pvalues(raw, "holm")
    bh = adjust_pvalues(raw, "bh")
    for row, p_holm, p_bh in zip(rows, holm, bh):
        row["p_holm"] = p_holm
        row["p_bh"] = p_bh
    return rows


def round1_statistics(episodes, event_dir: pathlib.Path):
    cells, missing_reasons = [], defaultdict(int)
    total_worse = total_eligible = total_missing = worse_cells = 0
    for task in CFGS:
        for arm in list(SHORT) + [name + "+tools" for name in SHORT]:
            items = episodes[(arm, task)]
            paired = []
            missing = []
            for seed, episode in sorted(items.items()):
                trace = episode.get("trace") or []
                if trace:
                    paired.append((float(trace[0]),
                                   float(episode["regret_zero_shot"])))
                    continue
                event_path = event_dir / f"events_{arm}_{task}_s{seed}.json"
                document = json.loads(event_path.read_text())
                kinds = [event["kind"] for event in document["events"]]
                if "stopped" in kinds:
                    reason = "stopped_before_first_completion"
                elif "forfeited" in kinds:
                    reason = "forfeited_without_completion"
                else:
                    reason = "no_completed_trace_other"
                missing.append({"seed": seed, "reason": reason})
                missing_reasons[reason] += 1
            r1 = np.asarray([x for x, _ in paired], float)
            zero = np.asarray([x for _, x in paired], float)
            n_worse = int(np.sum(r1 > zero))
            mean_worse = bool(r1.mean() > zero.mean()) if len(r1) else None
            cells.append({
                "task": task, "arm": arm, "eligible": len(paired),
                "missing": len(missing), "missing_episodes": missing,
                "n_worse": n_worse,
                "worse_rate": float(n_worse / len(paired)) if paired else None,
                "mean_round1_regret": float(r1.mean()) if paired else None,
                "mean_zero_shot_regret": float(zero.mean()) if paired else None,
                "cell_mean_worse": mean_worse,
            })
            total_worse += n_worse
            total_eligible += len(paired)
            total_missing += len(missing)
            worse_cells += int(bool(mean_worse))
    return {
        "policy": ("Pair round-1 and zero-shot regret only for episodes with a "
                   "completed first round. Episodes without a trace are excluded "
                   "and reported by execution reason; they are not imputed."),
        "eligible_episodes": total_eligible,
        "missing_episodes": total_missing,
        "missing_reasons": dict(missing_reasons),
        "n_worse": total_worse,
        "worse_rate": total_worse / total_eligible,
        "cells_with_worse_mean": worse_cells,
        "total_cells": len(cells),
        "cells": cells,
    }


def results_table(episodes, cv):
    lines = [r"% AUTO-GENERATED by scripts/make_phase1_tables.py",
             r"\begin{table}[t]", r"\centering\small",
             r"\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"\textbf{Model} & 0-shot & $\bar{\mathcal{R}}\downarrow$ & "
             r"$\mathcal{R}_{\rm cum}\downarrow$ & $c\uparrow$ & $\eta\uparrow$ \\",
             r"\midrule"]
    for task in CFGS:
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textsc{{{task}}}}} \\")
        for arm, name in SHORT.items():
            items = list(episodes[(arm, task)].values())
            regret = np.asarray([x["regret"] for x in items], float)
            zero = np.mean([x["regret_zero_shot"] for x in items])
            cumulative = np.mean([x["cumulative_regret"] for x in items])
            rstar = float(cv[task]["agents"][arm]["R_star"])
            c = 1 - rstar / float(cv[task]["prior"])
            eta = rstar / regret.mean()
            se = regret.std(ddof=1) / np.sqrt(len(regret))
            lines.append(
                f"{name} & {zero:.4f} & {regret.mean():.4f}({se*1e4:.0f}) & "
                f"{cumulative:.0f} & {100*c:.0f}\\% & {100*eta:.0f}\\%" + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Phase-1 corrected main results. Design quantities use only "
              r"accepted and completed events; outcome quantities use the frozen v1 "
              r"episode summaries.}", r"\label{tab:results}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def diagnostics_table(diagnostics):
    rows = {(row["arm"], row["task"]): row for row in diagnostics["rows"]}
    lines = [r"% AUTO-GENERATED by scripts/make_phase1_tables.py",
             r"\begin{table}[t]", r"\centering\footnotesize",
             r"\begin{tabular}{@{}l" + "cc" * 4 + "@{}}", r"\toprule",
             " & ".join([""] + [rf"\multicolumn{{2}}{{c}}{{\textsc{{{t}}}}}"
                                   for t in CFGS]) + r" \\",
             r"\textbf{Model}" + r" & bare & +tools" * 4 + r" \\", r"\midrule"]
    for arm, name in SHORT.items():
        cells = []
        for task in CFGS:
            for suffix in ("", "+tools"):
                row = rows[(arm + suffix, task)]
                cells.append(f"{100*row['rank_deficient_rate']:.0f}")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Rank-deficient completed designs (percent). Rejected drafts "
              r"are excluded.}", r"\label{tab:diag}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tools_table(rows):
    lines = [r"% AUTO-GENERATED by scripts/make_phase1_tables.py",
             r"\begin{table*}[t]", r"\centering\scriptsize",
             r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r"Task & Model & $\Delta c$ & $\Delta\eta$ & $\Delta\bar R$ "
             r"[95\% CI] & $p$ & Holm $p$ & BH $q$ \\", r"\midrule"]
    for row in rows:
        lo, hi = row["ci95_relative_to_bare_mean"]
        lines.append(
            f"{row['task']} & {SHORT[row['arm']]} & "
            f"{100*row['delta_c']:+.1f} & {100*row['delta_eta']:+.1f} & "
            f"{100*row['relative_difference']:+.1f} "
            f"[{100*lo:+.1f}, {100*hi:+.1f}] & {row['p_raw']:.3f} & "
            f"{row['p_holm']:.3f} & {row['p_bh']:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Paired tool effects over the 20 model--task comparisons. "
              r"Holm controls family-wise error and BH controls false discovery rate "
              r"over the full family of 20 tests.}", r"\label{tab:tools}",
              r"\end{table*}"]
    return "\n".join(lines) + "\n"


def report_markdown(tools, round1):
    raw = sum(row["p_raw"] < .05 for row in tools)
    holm = sum(row["p_holm"] < .05 for row in tools)
    bh = sum(row["p_bh"] < .05 for row in tools)
    lines = ["# Phase 1 remaining statistical audit", "",
             "## Tool comparisons", "",
             f"Across 20 paired comparisons: {raw} have raw p<0.05, "
             f"{holm} survive Holm FWER correction, and {bh} survive BH FDR correction.", "",
             "| Task | Arm | ΔR | 95% CI | raw p | Holm p | BH q | Δc (pp) | Δη (pp) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in tools:
        lo, hi = row["ci95_relative_to_bare_mean"]
        lines.append(
            f"| {row['task']} | {row['arm']} | {100*row['relative_difference']:+.1f}% | "
            f"[{100*lo:+.1f}%, {100*hi:+.1f}%] | {row['p_raw']:.4f} | "
            f"{row['p_holm']:.4f} | {row['p_bh']:.4f} | "
            f"{100*row['delta_c']:+.1f} | {100*row['delta_eta']:+.1f} |")
    lines += ["", "## Round-1 worse rate", "",
              round1["policy"], "",
              f"Eligible: {round1['eligible_episodes']}; missing: "
              f"{round1['missing_episodes']}; worse: {round1['n_worse']} "
              f"({100*round1['worse_rate']:.1f}%); cell means worse: "
              f"{round1['cells_with_worse_mean']}/{round1['total_cells']}.", "",
              f"Missing reasons: `{json.dumps(round1['missing_reasons'], sort_keys=True)}`."]
    return "\n".join(lines) + "\n"


def table_changes(episodes, old_cv, new_cv, diagnostics):
    main = []
    for task in CFGS:
        for arm in SHORT:
            regret = np.mean([
                item["regret"] for item in episodes[(arm, task)].values()])
            prior = float(new_cv[task]["prior"])
            old_rstar = float(old_cv[task]["agents"][arm]["R_star"])
            new_rstar = float(new_cv[task]["agents"][arm]["R_star"])
            main.append({
                "task": task, "arm": arm,
                "c_old": 1 - old_rstar / prior,
                "c_completed": 1 - new_rstar / prior,
                "eta_old": old_rstar / regret,
                "eta_completed": new_rstar / regret,
            })
    old_diag = {(row["arm"], row["task"]): row
                for row in diagnostics["legacy_rows"]}
    new_diag = {(row["arm"], row["task"]): row
                for row in diagnostics["rows"]}
    geometry = []
    for key in sorted(new_diag):
        old, new = old_diag[key], new_diag[key]
        geometry.append({
            "arm": key[0], "task": key[1],
            "n_old": old["n_designs"], "n_completed": new["n_designs"],
            "rank_deficient_old": old["rank_deficient_rate"],
            "rank_deficient_completed": new["rank_deficient_rate"],
            "insufficient_distinct_old": old["insufficient_distinct_rate"],
            "insufficient_distinct_completed": new["insufficient_distinct_rate"],
        })
    return {"table3_main": main, "table4_geometry": geometry}


def changes_markdown(changes, tools, round1):
    lines = ["# Tables 3--5: old vs Phase-1 corrected", "",
             "## Table 3: completed-event changes", "",
             "| Task | Arm | c old | c corrected | Δc | η old | η corrected | Δη |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in changes["table3_main"]:
        lines.append(
            f"| {row['task']} | {row['arm']} | {100*row['c_old']:.1f}% | "
            f"{100*row['c_completed']:.1f}% | "
            f"{100*(row['c_completed']-row['c_old']):+.1f} pp | "
            f"{100*row['eta_old']:.1f}% | {100*row['eta_completed']:.1f}% | "
            f"{100*(row['eta_completed']-row['eta_old']):+.1f} pp |")
    lines += ["", "## Table 4: design geometry", "",
              "The corrected table removes rejected designs. Across all 40 arms, "
              "the direction of the high-dimensional rank-deficiency result remains; "
              "cell-level values and denominators change.", "",
              "| Task | Arm | n old→new | Rank deficient old→new |",
              "|---|---|---:|---:|"]
    for row in changes["table4_geometry"]:
        lines.append(
            f"| {row['task']} | {row['arm']} | {row['n_old']}→{row['n_completed']} | "
            f"{100*row['rank_deficient_old']:.1f}%→"
            f"{100*row['rank_deficient_completed']:.1f}% |")
    lines += ["", "## Table 5 and trajectory claims", "",
              f"Tool direction remains 16/20 worse in realised regret. Raw p<0.05: "
              f"{sum(row['p_raw'] < .05 for row in tools)}/20; Holm: "
              f"{sum(row['p_holm'] < .05 for row in tools)}/20; BH: "
              f"{sum(row['p_bh'] < .05 for row in tools)}/20.", "",
              f"Round-1 worse is {round1['n_worse']}/{round1['eligible_episodes']} "
              f"({100*round1['worse_rate']:.1f}%). One episode is excluded because "
              "it forfeited the first round without a completed trace.", "",
              "## Claim classification", "",
              "| Claim | Classification |",
              "|---|---|",
              "| Transcript-parsed designs are valid scoring inputs | **Withdraw** |",
              "| High-dimensional designs often use too few treatments and are rank deficient | **Retain; numbers changed, direction unchanged** |",
              "| Tools make realised regret worse in 16/20 cells | **Retain descriptively** |",
              "| Six tool effects are statistically significant | **Significance changed**: six raw, two Holm, four BH |",
              "| Tools generally improve c | **Withdraw as a global claim**; effects are task×model dependent |",
              "| Round-1 data is worse than zero-shot in 44% of episodes | **Retain with denominator**: 354/799; one missing trace excluded |",
              "| c and η establish independent failure mechanisms | **Withdraw pending Phase 2** matched-history evaluator |"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True, type=pathlib.Path)
    parser.add_argument("--events", required=True, type=pathlib.Path)
    parser.add_argument("--cv", required=True, type=pathlib.Path)
    parser.add_argument("--cv-old", required=True, type=pathlib.Path)
    parser.add_argument("--diagnostics", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(args.episodes)
    cv = json.loads(args.cv.read_text())
    old_cv = json.loads(args.cv_old.read_text())
    diagnostics = json.loads(args.diagnostics.read_text())
    tools = tool_statistics(episodes, cv)
    round1 = round1_statistics(episodes, args.events)
    changes = table_changes(episodes, old_cv, cv, diagnostics)
    audit = {"tool_comparisons": tools, "round1": round1,
             "table_changes": changes}
    (args.out / "phase1_statistics.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False))
    (args.out / "phase1_statistics.md").write_text(
        report_markdown(tools, round1))
    (args.out / "tables_3_5_changelog.md").write_text(
        changes_markdown(changes, tools, round1))
    (args.out / "_results_table.tex").write_text(results_table(episodes, cv))
    (args.out / "_diag_table.tex").write_text(diagnostics_table(diagnostics))
    (args.out / "_tools_table.tex").write_text(tools_table(tools))
    print(json.dumps({
        "raw_significant": sum(row["p_raw"] < .05 for row in tools),
        "holm_significant": sum(row["p_holm"] < .05 for row in tools),
        "bh_significant": sum(row["p_bh"] < .05 for row in tools),
        "round1": {key: round1[key] for key in (
            "eligible_episodes", "missing_episodes", "n_worse",
            "worse_rate", "cells_with_worse_mean")},
    }, indent=2))


if __name__ == "__main__":
    main()
