#!/usr/bin/env python3
"""Generate Phase 1 figures from corrected completed-event statistics."""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

TASKS = ("Sanity", "Screen", "Optimise", "Transfer")
MARKERS = {"Sanity": "o", "Screen": "^", "Optimise": "s", "Transfer": "D"}
SHORT = {
    "deepseek_deepseek-v4-flash": "DeepSeek",
    "openai_gpt-5.6-luna": "GPT",
    "qwen_qwen3.8-27b": "Qwen",
    "xiaomi_mimo-v2.5": "MiMo",
    "z-ai_glm-5.3-flash": "GLM",
}
COLORS = dict(zip(SHORT, ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")))


def episode_means(directory: pathlib.Path):
    values = {}
    for path in directory.glob("episodes_*.json"):
        arm = path.stem.removeprefix("episodes_")
        grouped = {}
        for item in json.loads(path.read_text()):
            grouped.setdefault(item["task"], []).append(float(item["regret"]))
        for task, regret in grouped.items():
            values[(arm, task)] = float(np.mean(regret))
    return values


def efficiency_figure(cv, regrets, out: pathlib.Path):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True, sharey=True)
    for ax, suffix, title in zip(
            axes, ("", "+tools"), ("Bare interface", "With tools")):
        for task in TASKS:
            for arm, label in SHORT.items():
                key = arm + suffix
                rstar = float(cv[task]["agents"][key]["R_star"])
                c = 100 * (1 - rstar / float(cv[task]["prior"]))
                eta = 100 * rstar / regrets[(key, task)]
                ax.scatter(c, eta, marker=MARKERS[task], s=48,
                           edgecolor="black", facecolor=COLORS[arm], linewidth=.7)
        ax.axhline(0, color="0.7", linewidth=.6)
        ax.axvline(0, color="0.7", linewidth=.6)
        ax.grid(alpha=.2, linewidth=.5)
        ax.set_title(title)
        ax.set_xlabel("design efficiency c (%)")
    axes[0].set_ylabel("decision efficiency η (%)")
    task_handles = [Line2D([], [], marker=MARKERS[task], linestyle="",
                           markeredgecolor="black", markerfacecolor="white",
                           label=task) for task in TASKS]
    model_handles = [Line2D([], [], marker="o", linestyle="", color=color,
                            label=SHORT[arm]) for arm, color in COLORS.items()]
    first = axes[1].legend(handles=task_handles, frameon=False, fontsize=7,
                           loc="upper left", title="Task", title_fontsize=7)
    axes[1].add_artist(first)
    axes[1].legend(handles=model_handles, frameon=False, fontsize=7,
                   loc="upper right", title="Model", title_fontsize=7)
    fig.suptitle("Phase 1: accepted and completed designs only", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_efficiency_completed_events.{ext}",
                    bbox_inches="tight", dpi=220)
    plt.close(fig)


def tool_figure(rows, out: pathlib.Path):
    ordered = sorted(rows, key=lambda row: row["relative_difference"])
    labels = [f"{row['task']} / {SHORT[row['arm']]}" for row in ordered]
    center = 100 * np.asarray([row["relative_difference"] for row in ordered])
    ci = np.asarray([row["ci95_relative_to_bare_mean"] for row in ordered]) * 100
    errors = np.vstack([center - ci[:, 0], ci[:, 1] - center])
    colors = ["#2166ac" if row["delta_c"] > 0 else "#b2182b" for row in ordered]
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    y = np.arange(len(ordered))
    for i in range(len(ordered)):
        ax.errorbar(center[i], y[i], xerr=errors[:, i:i+1], fmt="o",
                    color=colors[i], ecolor=colors[i], capsize=2, markersize=4)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("paired change in realised regret (% of bare mean; 95% CI)")
    ax.grid(axis="x", alpha=.2)
    ax.set_title("Tool effects; blue means c increased, red means c decreased",
                 fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_tool_effects_completed_events.{ext}",
                    bbox_inches="tight", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True, type=pathlib.Path)
    parser.add_argument("--cv", required=True, type=pathlib.Path)
    parser.add_argument("--statistics", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cv = json.loads(args.cv.read_text())
    statistics = json.loads(args.statistics.read_text())
    efficiency_figure(cv, episode_means(args.episodes), args.out)
    tool_figure(statistics["tool_comparisons"], args.out)
    print("wrote corrected efficiency and tool-effect figures")


if __name__ == "__main__":
    main()
