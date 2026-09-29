#!/usr/bin/env python3
"""Turn "how bad the inference tool's own proposal is" into a reproducible file.

This analysis used to live only in a one-off session, with no script behind the
slope and correlation the paper reports. Here it is: TOOL 2's posterior maximum
is written into every round's prompt, so parse it as given, score it with the
same oracle, and write the result to results/tool_regret_env*.json.

    python scripts/tool_regret.py
"""
from __future__ import annotations
import json, glob, pathlib, re, sys
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import slowlab
from slowlab import SlowLabEnv, TASKS

R = ROOT / "results" / f"llm_env{slowlab.ENV_VERSION}"
CFGS = ("Sanity", "Screen", "Optimise", "Transfer")
SLUGS = ("deepseek_deepseek-v4-flash", "openai_gpt-5.6-luna", "qwen_qwen3.8-27b",
         "xiaomi_mimo-v2.5", "z-ai_glm-5.3-flash")
PAT = re.compile(r"posterior maximum is at:\s*\n((?:\s*\w+=[-\d.]+,?\s*)+)")


def tool_point(user_text, env):
    """The point TOOL 2 reports in the last prompt, normalised to [0,1]^d. Returns None if absent."""
    m = PAT.search(user_text)
    if not m:
        return None
    kv = dict(re.findall(r"(\w+)=(-?[\d.]+)", m.group(1)))
    out = []
    for f in env.task.factors:
        if f.name not in kv:
            return None
        out.append((float(kv[f.name]) - f.low) / (f.high - f.low))
    return np.clip(np.array(out), 0, 1)



def verbatim_copy_rate():
    """How often the current_best the model reports *is* the tool's point.

    The slope of 0.29 is an average across cells and does not show how literal the
    deference can be. This compares round by round: the point TOOL 2 reports in a
    prompt against the current_best the model writes in that same round's reply.
    """
    tot = exact = 0
    per = {}
    for cfg in CFGS:
        t = TASKS[cfg]
        names = [f.name for f in t.factors]
        a = b = 0
        for slug in SLUGS:
            for f in sorted(glob.glob(str(R / f"transcript_{slug}+tools_{cfg}_s*.json"))):
                seed = int(re.search(r"_s(\d+)\.json$", f).group(1))
                env = SlowLabEnv(t, seed=seed)
                for turn in json.loads(pathlib.Path(f).read_text()):
                    tp = tool_point(turn["user"], env)
                    m = re.search(r'"current_best"\s*:\s*(\{[^}]*\})', turn["assistant"])
                    if tp is None or not m:
                        continue
                    try:
                        cb = json.loads(m.group(1))
                    except Exception:
                        continue
                    if any(n not in cb for n in names):
                        continue
                    x = np.array([(float(cb[n]) - fa.low) / (fa.high - fa.low)
                                  for n, fa in zip(names, t.factors)])
                    a += 1
                    b += float(np.abs(x - tp).max()) < 1e-6
        per[cfg] = {"exact": b, "comparable": a}
        tot += a; exact += b
    return {"comparable_turns": tot, "exact": exact, "rate": exact / tot,
            "by_task": per}


def main():
    per_cell, rows = {}, []
    for slug in SLUGS:
        for cfg in CFGS:
            t = TASKS[cfg]
            vals = []
            for f in sorted(glob.glob(str(R / f"transcript_{slug}+tools_{cfg}_s*.json"))):
                seed = int(re.search(r"_s(\d+)\.json$", f).group(1))
                env = SlowLabEnv(t, seed=seed)
                d = json.loads(pathlib.Path(f).read_text())
                # The last prompt has seen the most data and is the one the model saw before submitting
                x = None
                for turn in reversed(d):
                    x = tool_point(turn["user"], env)
                    if x is not None:
                        break
                if x is None:
                    continue
                _, best = env.truth.oracle()
                vals.append(float(best - env.truth(x.reshape(1, -1))[0]))
            if len(vals) >= 10:
                per_cell[f"{slug}|{cfg}"] = {"regret_tool": float(np.mean(vals)),
                                             "n": len(vals)}
    if not per_cell:
        raise SystemExit("no TOOL 2 point could be parsed -- the prompt format changed; look at a transcript first")

    # Pair with the measured bare / tooled regret
    E = {}
    for f in sorted(glob.glob(str(R / "episodes_*.json"))):
        slug = pathlib.Path(f).stem[len("episodes_"):]
        for e in json.loads(pathlib.Path(f).read_text()):
            E.setdefault((slug, e["task"]), {})[e["seed"]] = e

    xs, ys = [], []
    for key, v in sorted(per_cell.items()):
        slug, cfg = key.split("|")
        a, b = E.get((slug, cfg), {}), E.get((slug + "+tools", cfg), {})
        ks = sorted(set(a) & set(b))
        if len(ks) < 10:
            continue
        bare = float(np.mean([a[k]["regret"] for k in ks]))
        tool = float(np.mean([b[k]["regret"] for k in ks]))
        adv = (v["regret_tool"] - bare) / bare      # how much worse the tool is than the model
        eff = (tool - bare) / bare                  # how much worse things got once the tool was given
        xs.append(adv); ys.append(eff)
        rows.append({"model": slug, "task": cfg, "n": len(ks),
                     "regret_tool": v["regret_tool"], "bare": bare, "tooled": tool,
                     "tool_disadvantage": adv, "tool_effect": eff})

    lr = stats.linregress(xs, ys)
    sign_ok = sum(1 for a, e in zip(xs, ys) if np.sign(a) == np.sign(e))
    out = {"cells": rows, "n_cells": len(rows),
           "slope": lr.slope, "stderr": lr.stderr, "r": lr.rvalue, "p": lr.pvalue,
           "sign_predicted": sign_ok,
           "tool_worse_than_model": sum(1 for a in xs if a > 0),
           "by_task": {c: float(np.mean([r["regret_tool"] for r in rows
                                         if r["task"] == c]))
                       for c in CFGS if any(r["task"] == c for r in rows)}}
    out["verbatim"] = verbatim_copy_rate()
    p = ROOT / "results" / f"tool_regret_env{slowlab.ENV_VERSION}.json"
    p.write_text(json.dumps(out, indent=1))

    print(f"{'model':22s}{'task':10s}{'R_tool':>9}{'bare':>9}{'tooled':>9}"
          f"{'adv':>8}{'eff':>8}")
    for r in rows:
        print(f"{r['model'][:21]:22s}{r['task']:10s}{r['regret_tool']:9.4f}"
              f"{r['bare']:9.4f}{r['tooled']:9.4f}{r['tool_disadvantage']:+8.2f}"
              f"{r['tool_effect']:+8.2f}")
    print(f"\nn={len(rows)}  slope {lr.slope:.2f}±{lr.stderr:.2f}  "
          f"r={lr.rvalue:.2f}  p={lr.pvalue:.4f}  "
          f"sign correct {sign_ok}/{len(rows)}  tool worse than model {out['tool_worse_than_model']}")
    print("tool's own regret per task:", {k: round(v, 4) for k, v in out["by_task"].items()})
    v = out["verbatim"]
    print(f"verbatim copies: in {v['exact']} of {v['comparable_turns']} turns "
          f"({100*v['rate']:.0f}%) the model's current_best is the tool's point")
    print("wrote", p.name)


if __name__ == "__main__":
    main()
