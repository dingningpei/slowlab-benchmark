#!/usr/bin/env python3
"""Rebuild Phase 5 analyses, tables, figures, checks, and PDF from frozen inputs."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(*args: str, cwd: Path = ROOT) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True,
                        help="directory containing frozen Phase 3/4 JSON and episode trees")
    args = parser.parse_args()
    raw = args.analysis_dir.resolve()
    build = ROOT / "build" / "phase5-analysis"
    build.mkdir(parents=True, exist_ok=True)
    config = ROOT / "configs" / "phase4_preregistered.json"

    run(PYTHON, "-B", "scripts/analyze_phase4.py", "--config", str(config),
        "--root", str(raw / "phase4_confirmatory"),
        "--out", str(build / "phase4_primary_analysis.json"))
    for matrix, name in [
        ("strong_model_replication", "phase4_strong_model_analysis.json"),
        ("task_heterogeneity", "phase4_task_heterogeneity_analysis.json"),
        ("within_cycle_feedback", "phase4_within_cycle_analysis.json"),
    ]:
        run(PYTHON, "-B", "scripts/analyze_phase4_secondary.py", "--config", str(config),
            "--root", str(raw / "phase4_confirmatory"), "--matrix", matrix,
            "--out", str(build / name))
    run(PYTHON, "-B", "scripts/analyze_phase4_noise_robustness.py",
        "--config", str(ROOT / "configs" / "phase4_noise_robustness.json"),
        "--root", str(raw / "phase4_noise_robustness"),
        "--out", str(build / "phase4_noise_robustness_analysis.json"))
    run(PYTHON, "-B", "scripts/build_phase5_paper_artifacts.py",
        "--analysis-dir", str(build), "--raw-dir", str(raw))
    run(PYTHON, "-B", "scripts/make_overview.py")
    run(PYTHON, "-B", "scripts/make_phase5_mechanism_figure.py",
        "--analysis-dir", str(raw))
    run(PYTHON, "-B", "scripts/make_fig_worked_example.py")
    run(PYTHON, "-B", "scripts/check_crossrefs.py")
    run(PYTHON, "-B", "scripts/verify_results_claims.py")

    for executable in ("pdflatex", "bibtex"):
        if not shutil.which(executable):
            raise SystemExit(f"{executable} is required to build the paper")
    run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex",
        cwd=ROOT / "paper")
    run("bibtex", "main", cwd=ROOT / "paper")
    for _ in range(2):
        run("pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex",
            cwd=ROOT / "paper")
    print(f"complete: {ROOT / 'paper' / 'main.pdf'}")


if __name__ == "__main__":
    main()
