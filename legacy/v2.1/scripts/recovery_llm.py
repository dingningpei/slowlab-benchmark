#!/usr/bin/env python3
"""The language-model half of the irreversibility ablation, paired by site.

The scripted half (scripts/recovery_sweep.py) found that withdrawing capacity after heat
damage does not change what a reference strategy achieves. That measurement has a boundary:
none of the references replans around lost units, so it prices the loss to an agent that
ignores it. A language model does see the shortened available_units list and could respond,
which is what this compares.

Both arms are \\textsc{Optimise}, five models, the same twenty sites, bare interface. The
only difference is recovery_days: 0 in results/llm_env<version>/ (the setting of every
number in the paper's main tables) against 210 in the directory given below.

Per-model differences are not separable from rerun noise -- the models are sampled at
temperature 0.7, so the same cell rerun at the same setting would also move -- so the
pooled paired test over all hundred episodes is the one to read.

    python scripts/recovery_llm.py --arm results/llm_recovery210_env1.0.0
"""
from __future__ import annotations
import sys, json, glob, pathlib, re, argparse
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import slowlab
from slowlab import SlowLabEnv, TASKS
from eig_of_llm import designs_from

TASK = "Optimise"


def episodes(d, task=TASK):
    E = {}
    for f in sorted(glob.glob(f"{d}/episodes_*.json")):
        if "+tools" in f:
            continue
        slug = pathlib.Path(f).stem[len("episodes_"):].replace(f"@{task}", "")
        for e in json.loads(pathlib.Path(f).read_text()):
            if e["task"] == task:
                E.setdefault(slug, {})[e["seed"]] = e
    return E


def units_per_round(d, task=TASK):
    """How wide each round actually was, from the transcripts.

    This is the only place the dial's effect on capacity is visible for the first run of
    the ablation: run_llm.py did not record lost_unit_days at the time.
    """
    t = TASKS[task]
    by = {}
    for f in sorted(glob.glob(f"{d}/transcript_*_{task}_s*.json")):
        if "+tools" in f:
            continue
        s = int(re.search(r"_s(\d+)\.json$", f).group(1))
        env = SlowLabEnv(t, seed=s)
        for i, (p, _) in enumerate(designs_from(f, env)):
            if i < t.n_rounds:
                by.setdefault(i, []).append(len(np.atleast_2d(p)))
    return {i: float(np.mean(v)) for i, v in sorted(by.items())}


def main(base=None, arm=None, out=None):
    base = base or f"results/llm_env{slowlab.ENV_VERSION}"
    arm = arm or f"results/llm_recovery210_env{slowlab.ENV_VERSION}"
    A, B = episodes(ROOT / base), episodes(ROOT / arm)

    print(f"{'model':28s}{'rd=0':>9}{'rd=210':>9}{'delta':>8}{'paired p':>10}"
          f"{'infeasible 0 / 210':>20}")
    rows, ax, ay = [], [], []
    for m in sorted(B):
        if m not in A:
            print(f"  {m}: no rd=0 arm"); continue
        ks = sorted(set(A[m]) & set(B[m]))
        x = np.array([A[m][k]["regret"] for k in ks])
        y = np.array([B[m][k]["regret"] for k in ks])
        i0 = np.mean([A[m][k]["infeasible"] for k in ks])
        i1 = np.mean([B[m][k]["infeasible"] for k in ks])
        p = stats.ttest_rel(y, x).pvalue
        ax += list(x); ay += list(y)
        rows.append({"model": m, "n": len(ks), "rd0": float(x.mean()),
                     "rd210": float(y.mean()), "p": float(p)})
        print(f"{m:28s}{x.mean():>9.4f}{y.mean():>9.4f}"
              f"{100*(y.mean()-x.mean())/x.mean():>7.0f}%{p:>10.3f}"
              f"{i0:>12.2f} /{i1:5.2f}")

    x, y = np.array(ax), np.array(ay)
    t = stats.ttest_rel(y, x)
    print(f"\npooled  n={len(x)}  rd=0 {x.mean():.4f}  rd=210 {y.mean():.4f}  "
          f"{100*(y.mean()-x.mean())/x.mean():+.0f}%  "
          f"paired t={t.statistic:+.2f}  p={t.pvalue:.3f}  "
          f"worse in {int((y > x).sum())}/{len(x)}")

    u0, u1 = units_per_round(ROOT / base), units_per_round(ROOT / arm)
    print("\nunits submitted per round (the dial's effect on capacity)")
    for i in sorted(u0):
        print(f"  round {i+1}: {u0[i]:.1f} -> {u1.get(i, float('nan')):.1f}"
              f"   ({100*(u1.get(i, np.nan)-u0[i])/u0[i]:+.0f}%)")

    res = {"task": TASK, "base": base, "arm": arm, "per_model": rows,
           "pooled": {"n": len(x), "rd0": float(x.mean()), "rd210": float(y.mean()),
                      "t": float(t.statistic), "p": float(t.pvalue),
                      "n_worse": int((y > x).sum())},
           "units_per_round": {"rd0": u0, "rd210": u1}}
    fp = ROOT / (out or f"results/recovery_llm_env{slowlab.ENV_VERSION}.json")
    fp.write_text(json.dumps(res, indent=1))
    print("\nwrote", fp.name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base")
    ap.add_argument("--arm")
    a = ap.parse_args()
    main(base=a.base, arm=a.arm)
