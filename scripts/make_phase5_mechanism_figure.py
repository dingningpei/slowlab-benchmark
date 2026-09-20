#!/usr/bin/env python3
"""Plot the four Phase 5 mechanism checks from frozen aggregate analyses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path,
                        default=ROOT / "results" / "phase5_paper_summary_env2.0.0.json")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "paper" / "figures" / "fig3_phase5_mechanisms.pdf")
    args = parser.parse_args()
    replay = load(args.analysis_dir / "phase3_design_reader_replay.json")
    calibration = load(args.analysis_dir / "phase2_history_evaluator_calibration.json")
    summary = load(args.summary)

    designs = ["constraint_aware_batch_bo", "random_spread", "split_plot_doe"]
    design_labels = ["Batch BO", "Random", "Split-plot"]
    readers = ["ordinary_gp", "block_aware_gp", "component_gp"]
    reader_labels = ["Ordinary", "Block", "Component"]
    cells = replay["overall_effects"]["cell_means"]
    matrix = np.array([[cells[f"{d}|{r}"] for r in readers] for d in designs])

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    ax = axes[0, 0]
    means = [replay["overall_effects"]["design_means"][d] for d in designs]
    ax.bar(design_labels, means, color=["#0072B2", "#56B4E9", "#E69F00"])
    ax.set_ylabel("Mean final regret")
    ax.set_title("A  Executed designs", loc="left", fontweight="bold")
    ax.tick_params(axis="x", rotation=18)

    ax = axes[0, 1]
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="white" if matrix[i, j] > 0.036 else "black", fontsize=8)
    ax.set_xticks(range(3), reader_labels)
    ax.set_yticks(range(3), design_labels)
    ax.set_title("B  Design × reader", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=.045, pad=.03, label="Final regret")

    ax = axes[1, 0]
    effect = summary["within_cycle"]
    lo, hi = effect["ci95"]
    ax.errorbar([effect["mean"]], [0],
                xerr=[[effect["mean"] - lo], [hi - effect["mean"]]],
                fmt="o", color="#D55E00", capsize=4)
    ax.axvline(0, color="#777777", lw=1, ls="--")
    ax.set_yticks([0], ["Within-cycle − terminal"])
    ax.set_xlabel("Paired change in final regret (95% CI)")
    ax.set_title("C  Feedback mechanism", loc="left", fontweight="bold")

    ax = axes[1, 1]
    product = calibration["factorized_product"]
    runs = product["runs"]
    actions = [row["bayes_action"] for row in runs]
    ax.scatter([0, 1], actions, s=45, color="#009E73", zorder=3)
    ax.plot([0, 1], actions, color="#009E73", lw=1)
    ax.set_xticks([0, 1], ["Run 1", "Run 2"])
    ax.set_ylabel("Bayes action")
    ax.set_ylim(min(actions) - .004, max(actions) + .004)
    ax.text(.5, min(actions) - .0028,
            f"response-curve RMSE = {product['between_run_response_curve_rmse']:.5f}",
            ha="center", fontsize=8)
    ax.set_title("D  Evaluator calibration", loc="left", fontweight="bold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
