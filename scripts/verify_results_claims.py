#!/usr/bin/env python3
"""Verify that the Phase 5 manuscript agrees with its frozen public summary."""
from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "phase5_paper_summary_env2.0.0.json"


def close(got: float, expected: float, label: str, tol: float = 5e-7) -> None:
    if not math.isclose(got, expected, rel_tol=0.0, abs_tol=tol):
        raise AssertionError(f"{label}: got {got}, expected {expected}")


def require(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise AssertionError(f"{label}: missing {needle!r}")


def main() -> None:
    data = json.loads(SUMMARY.read_text())
    primary = data["primary_tool_contrasts"]
    close(primary["design_minus_bare"]["mean"], 0.0001580172060754977,
          "primary design contrast")
    close(primary["inference_minus_bare"]["mean"], -0.0019644177945116505,
          "primary inference contrast")
    close(primary["both_minus_bare"]["mean"], -0.00022584818071818443,
          "primary combined contrast")
    close(data["strong_model_contrast"]["mean"], 0.0005261737158492671,
          "Sol inference contrast")
    close(data["within_cycle"]["mean"], 0.0076778205407092405,
          "within-cycle contrast")
    close(data["noise_0_5x_design_minus_bare"]["mean"], 0.011326575017979339,
          "half-noise design contrast")
    close(data["transfer"]["post_shock_inference_minus_bare"]["mean"],
          0.04518249211654407, "post-shock transfer contrast")

    paper = "\n".join(path.read_text() for path in [
        ROOT / "paper" / "sections" / "00_abstract.tex",
        ROOT / "paper" / "sections" / "07_results.tex",
        ROOT / "paper" / "sections" / "09_conclusion.tex",
        ROOT / "paper" / "sections" / "_phase4_main_table.tex",
        ROOT / "paper" / "sections" / "_phase4_secondary_table.tex",
        ROOT / "paper" / "sections" / "_phase4_transfer_table.tex",
    ])
    for needle, label in [
        ("Across 928 episodes", "abstract episode count"),
        ("384-episode", "noise-extension count"),
        ("$+0.00016$", "primary design text"),
        ("$-0.00196$", "primary inference text"),
        ("$+0.00053$", "Sol text"),
        ("$+0.00768$", "within-cycle text"),
        ("$0.0452$", "Transfer post-shock text"),
        ("$p=0.0096$", "half-noise adjusted p value"),
    ]:
        require(paper, needle, label)

    forbidden = [
        "sixteen of twenty cells",
        "44\\% of episodes",
        "design efficiency $c$",
        "decision efficiency $\\eta$",
        "defer empirical claims",
    ]
    for phrase in forbidden:
        if phrase in paper:
            raise AssertionError(f"withdrawn claim remains in compiled Phase 5 sections: {phrase}")
    print("OK -- Phase 5 numerical claims match the frozen public summary")


if __name__ == "__main__":
    main()
