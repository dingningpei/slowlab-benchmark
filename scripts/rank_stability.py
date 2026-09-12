#!/usr/bin/env python3
"""Would the leaderboard come back the same on a different twenty sites?

Two of the four tasks have a prize barely larger than the noise a full budget leaves
behind (Table 2: the ratio is 0.8 on Screen and 1.0 on Transfer), which invites the
objection that any ordering they produce is a coin toss. This measures it rather than
arguing about it.

Method: split the twenty instances into two disjoint halves of ten, rank the five models
on each half by mean realised regret, and correlate the two rankings. Every model sees the
same sites within a half, so the split is paired and the only thing varying is which sites
were drawn. Repeated over many random splits, the mean rank correlation is how much of the
leaderboard survives resampling the site draw, and the top-1 agreement rate is how often
the winner is the same model.

    python scripts/rank_stability.py
"""
from __future__ import annotations
import sys, json, glob, pathlib, argparse
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import slowlab

R = ROOT / "results" / f"llm_env{slowlab.ENV_VERSION}"
CFGS = ("Sanity", "Screen", "Optimise", "Transfer")
SLUGS = ("deepseek_deepseek-v4-flash", "openai_gpt-5.6-luna", "qwen_qwen3.8-27b",
         "xiaomi_mimo-v2.5", "z-ai_glm-5.3-flash")


def load():
    E = {}
    for f in sorted(glob.glob(str(R / "episodes_*.json"))):
        slug = pathlib.Path(f).stem[len("episodes_"):]
        for e in json.loads(pathlib.Path(f).read_text()):
            E.setdefault((slug, e["task"]), {})[e["seed"]] = e
    return E


def split_half(E, cfg, arms, field="regret", n_boot=4000, half=10, seed=0):
    """Mean rank correlation between the two halves, and top-1 agreement."""
    seeds = sorted(set.intersection(*[set(E[(a, cfg)]) for a in arms
                                      if (a, cfg) in E]))
    if len(seeds) < 2 * half:
        half = len(seeds) // 2
    Y = np.array([[E[(a, cfg)][s][field] for s in seeds] for a in arms])  # arms x seeds
    rng = np.random.default_rng(seed)
    rhos, top1 = [], 0
    for _ in range(n_boot):
        p = rng.permutation(len(seeds))
        A, B = p[:half], p[half:2 * half]
        ma, mb = Y[:, A].mean(1), Y[:, B].mean(1)
        rho = stats.spearmanr(ma, mb).statistic
        if not np.isnan(rho):
            rhos.append(rho)
        top1 += int(np.argmin(ma) == np.argmin(mb))
    rhos = np.array(rhos)
    return {"n_seeds": len(seeds), "half": half, "field": field,
            "rho_mean": float(rhos.mean()),
            "rho_lo": float(np.percentile(rhos, 2.5)),
            "rho_hi": float(np.percentile(rhos, 97.5)),
            "p_rho_positive": float((rhos > 0).mean()),
            "top1_agree": top1 / n_boot}


def load_rstar():
    """R*(D) per (arm, task, seed), averaged over the rounds of that episode.

    The cross-validated file keys designs as transcript_<arm>_<task>_s<seed>.json#<round>,
    so the same split of the twenty sites can be applied to the design measure.
    """
    import re
    fp = ROOT / "results" / f"achievable_cv_env{slowlab.ENV_VERSION}_M320.json"
    d = json.loads(fp.read_text())
    out, prior = {}, {}
    for cfg, slot in d.items():
        prior[cfg] = slot["prior"]
        for key, v in slot["raw"].items():
            m = re.match(r"transcript_(.+)_[A-Za-z]+_s(\d+)\.json#\d+$", key)
            if not m: continue
            out.setdefault((m.group(1), cfg), {}).setdefault(int(m.group(2)), []).append(v)
    return {k: {s: float(np.mean(v)) for s, v in d_.items()} for k, d_ in out.items()}, prior


def split_half_design(E, RS, prior, cfg, arms, n_boot=4000, half=10, seed=0):
    """The same split applied to design efficiency c and decision efficiency eta."""
    arms = [a for a in arms if (a, cfg) in RS]
    if len(arms) < 3: return None
    seeds = sorted(set.intersection(*[set(RS[(a, cfg)]) & set(E[(a, cfg)]) for a in arms]))
    if len(seeds) < 4: return None
    half = min(half, len(seeds) // 2)
    Rst = np.array([[RS[(a, cfg)][s] for s in seeds] for a in arms])
    Rbar = np.array([[E[(a, cfg)][s]["regret"] for s in seeds] for a in arms])
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("c", "eta"):
        rhos, top1 = [], 0
        for _ in range(n_boot):
            p = rng.permutation(len(seeds)); A, B = p[:half], p[half:2*half]
            if name == "c":
                va = 1 - Rst[:, A].mean(1)/prior[cfg]; vb = 1 - Rst[:, B].mean(1)/prior[cfg]
            else:
                va = Rst[:, A].mean(1)/Rbar[:, A].mean(1)
                vb = Rst[:, B].mean(1)/Rbar[:, B].mean(1)
            rho = stats.spearmanr(va, vb).statistic
            if not np.isnan(rho): rhos.append(rho)
            top1 += int(np.argmax(va) == np.argmax(vb))     # larger is better for both
        rhos = np.array(rhos)
        out[name] = {"n_seeds": len(seeds), "half": half,
                     "rho_mean": float(rhos.mean()),
                     "rho_lo": float(np.percentile(rhos, 2.5)),
                     "rho_hi": float(np.percentile(rhos, 97.5)),
                     "p_rho_positive": float((rhos > 0).mean()),
                     "top1_agree": top1 / n_boot}
    return out


def main(n_boot=4000, out=None):
    E = load()
    out = out or f"results/rank_stability_env{slowlab.ENV_VERSION}.json"
    res = {}
    for field in ("regret", "cumulative_regret"):
        print(f"=== ranking by {field}   (twenty sites split 10 / 10, "
              f"{n_boot} random splits)")
        print(f"{'task':10s}{'arm':8s}{'rho (95% CI)':>26}{'P(rho>0)':>10}{'top-1 agree':>13}")
        for cfg in CFGS:
            for label, arms in (("bare", SLUGS),
                                ("+tools", tuple(s + "+tools" for s in SLUGS))):
                if not all((a, cfg) in E for a in arms):
                    continue
                r = split_half(E, cfg, arms, field=field, n_boot=n_boot)
                res[f"{field}|{cfg}|{label}"] = r
                print(f"{cfg:10s}{label:8s}"
                      f"{r['rho_mean']:+.2f} [{r['rho_lo']:+.2f}, {r['rho_hi']:+.2f}]".rjust(26)
                      + f"{r['p_rho_positive']:>10.2f}{r['top1_agree']:>13.2f}")
        print()
    RS, prior = load_rstar()
    print("=== ranking by design efficiency c and decision efficiency eta")
    print(f"{'task':10s}{'arm':8s}{'measure':9s}{'rho (95% CI)':>26}{'P(rho>0)':>10}"
          f"{'top-1 agree':>13}")
    for cfg in CFGS:
        for label, arms in (("bare", SLUGS),
                            ("+tools", tuple(s + "+tools" for s in SLUGS))):
            r = split_half_design(E, RS, prior, cfg, arms, n_boot=n_boot)
            if not r: continue
            for name, v in r.items():
                res[f"{name}|{cfg}|{label}"] = v
                print(f"{cfg:10s}{label:8s}{name:9s}"
                      + f"{v['rho_mean']:+.2f} [{v['rho_lo']:+.2f}, {v['rho_hi']:+.2f}]".rjust(26)
                      + f"{v['p_rho_positive']:>10.2f}{v['top1_agree']:>13.2f}")
    print()
    (ROOT / out).write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=4000)
    a = ap.parse_args()
    main(n_boot=a.n_boot)
