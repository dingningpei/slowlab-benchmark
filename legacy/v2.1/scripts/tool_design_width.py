#!/usr/bin/env python3
"""The cleanest evidence that the tool inherits the design: how many distinct points
the same inference routine is handed.

This claim used to rest on within-task rank correlations alone (-0.24 / -0.31, and
not significant on the other two tasks), which invites the question of whether the
mechanism simply fails on Screen. Here it is replaced by a comparison that holds on
all four tasks: the same `_gp_posterior_max`, given the points it sees when it picks
its own design, against the points it sees when fed the LLM's design.

The spread of the point counts is recorded too: on Screen they range from 3 to 25
(CV 0.32), comparable to Optimise, so the missing correlation there is not a
restricted range but the fact that in six dimensions extra points buy nothing.

    python scripts/tool_design_width.py
"""
from __future__ import annotations
import json, glob, pathlib, re, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import slowlab
from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent
from eig_of_llm import designs_from

R = ROOT / "results" / f"llm_env{slowlab.ENV_VERSION}"
CFGS = ("Sanity", "Screen", "Optimise", "Transfer")
SLUGS = ("deepseek_deepseek-v4-flash", "openai_gpt-5.6-luna", "qwen_qwen3.8-27b",
         "xiaomi_mimo-v2.5", "z-ai_glm-5.3-flash")
REF, N_REF_SEEDS = "gp_ucb_rep1", 8


def distinct(P):
    return len(np.unique(np.round(np.asarray(P, float), 6), axis=0))


def main():
    out = {}
    print(f"{'task':10s}{'d':>3}{'LLM+tools':>26}{'reference':>11}{'ratio':>8}")
    for cfg in CFGS:
        t = TASKS[cfg]
        llm = []
        for slug in SLUGS:
            for f in sorted(glob.glob(str(R / f"transcript_{slug}+tools_{cfg}_s*.json"))):
                seed = int(re.search(r"_s(\d+)\.json$", f).group(1))
                ds = designs_from(f, SlowLabEnv(t, seed=seed))
                if ds:
                    llm.append(distinct(np.vstack([np.atleast_2d(p) for p, _ in ds])))
        ref = []
        for seed in range(N_REF_SEEDS):
            e = SlowLabEnv(t, seed=seed)
            ag = make_agent(REF)
            ag.run(e)
            X, _ = e.as_arrays()
            ref.append(distinct(X))
        llm = np.array(llm); ref = np.array(ref)
        a, b = float(np.median(llm)), float(np.median(ref))
        out[cfg] = {"d": len(t.factors), "llm_median": a, "ref_median": b,
                    "ratio": b / a, "llm_min": int(llm.min()), "llm_max": int(llm.max()),
                    "llm_cv": float(llm.std() / llm.mean()), "n_llm": len(llm),
                    "reference": REF}
        cell = f"{a:.0f}  (range {llm.min()}-{llm.max()}, CV {out[cfg]['llm_cv']:.2f})"
        print(f"{cfg:10s}{len(t.factors):3d}{cell:>26}{b:11.0f}{b/a:7.1f}x")
    rs = [v["ratio"] for k, v in out.items() if v["d"] > 1]
    out["summary"] = {"ratio_min": min(rs), "ratio_max": max(rs),
                      "note": "Sanity excluded from the range: d=1 leaves nothing to spread"}
    p = ROOT / "results" / f"tool_design_width_env{slowlab.ENV_VERSION}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nOutside Sanity (d=1), the reference policy hands the same routine "
          f"{min(rs):.1f}-{max(rs):.1f} times as many points as the LLM designs do")
    print("wrote", p.name)


if __name__ == "__main__":
    main()
