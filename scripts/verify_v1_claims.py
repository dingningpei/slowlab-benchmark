#!/usr/bin/env python3
"""Recompute every number in the Results section and print, line by line, what the
paper says against what the data now gives.

The code and the evaluation have both changed several times. This script exists
so that any sentence carrying a number has a corresponding line of output here.
A sentence with no corresponding line either gets one, or gets deleted.

    python scripts/verify_results_claims.py
"""
from __future__ import annotations
import json, glob, pathlib, re, sys
import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import slowlab
from slowlab import SlowLabEnv, TASKS
from eig_of_llm import designs_from

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


def say(claim, got):
    print(f"  paper: {claim}\n  now  : {got}\n")


def main():
    E = load()
    CV = json.loads((ROOT / "results" /
                     f"achievable_cv_env{slowlab.ENV_VERSION}_M320.json").read_text())

    print("=" * 78)
    print("A. Episodes per cell (the paper says all 40 cells have their full 20)")
    short = [(s, c, len(E.get((s, c), {}))) for s in SLUGS for c in CFGS
             if len(E.get((s, c), {})) != 20]
    say("every one of the forty cells has its full twenty instances",
        "full" if not short else f"short: {short}")

    print("=" * 78)
    print("B. Was the campaign worth running (bare vs the model's own zero-shot)")
    ARMS = list(SLUGS) + [s + "+tools" for s in SLUGS]   # the paper's 40 cells are bare plus tooled
    worse = []
    for s in ARMS:
        for c in CFGS:
            es = list(E.get((s, c), {}).values())
            if not es: continue
            m = np.mean([e["regret"] for e in es])
            z = np.mean([e["regret_zero_shot"] for e in es])
            if m >= z: worse.append((s, c, m, z))
    say("31 of 40 cells end below 0-shot; the remaining 9 end above it",
        f"{40-len(worse)} below 0-shot, {len(worse)} above ({len(ARMS)*len(CFGS)} cells in total)")
    say("all nine are on Sanity and Optimise",
        f"tasks the above-0-shot cells fall on: {sorted({c for _, c, _, _ in worse})}")
    # The paper says "not one of the forty arms reaches it", so all forty have to be
    # checked here: an earlier version of this block iterated the twenty bare cells while
    # reporting against a denominator of forty.
    ceil = [(s, c, np.mean([e["regret"] for e in E[(s, c)].values()]), CV[c]["prior"])
            for s in ARMS for c in CFGS if E.get((s, c))]
    bare_r = [m/p for s, _, m, p in ceil if "+tools" not in s]
    tool_r = [m/p for s, _, m, p in ceil if "+tools" in s]
    say("no cell on any task reaches R*(∅); bare arms sit 1.8x to 23.4x above it",
        f"{sum(1 for _,_,m,p in ceil if m < p)} / {len(ceil)} cells have R-bar below "
        f"R*(none); bare {min(bare_r):.1f}x..{max(bare_r):.1f}x, "
        f"tooled {min(tool_r):.1f}x..{max(tool_r):.1f}x")

    refs = json.loads((ROOT / "results" /
                       f"reference_table_env{slowlab.ENV_VERSION}.json").read_text())
    rr = [(d["realised"]/v["prior"], t, a)
          for t, v in refs["tasks"].items() for a, d in v["agents"].items()]
    say("the scripted references are closer but also short, at 1.6x to 22.3x",
        f"{min(rr)[0]:.1f}x ({min(rr)[1]}/{min(rr)[2]}) .. "
        f"{max(rr)[0]:.1f}x ({max(rr)[1]}/{max(rr)[2]})")

    print("=" * 78)
    print("C. Design diagnostics (rank deficiency; too few distinct treatments)")
    # Panel (b) of Table 4 counts designs with fewer distinct treatments than the task
    # has factors plus one -- k < d+1 -- rather than designs with exactly two. On
    # Sanity (d=1) two treatments are exactly enough, so the old criterion called 67%
    # of them failures when the true "too few" rate is 3%.
    diag = {}
    for slug in list(SLUGS) + [s + "+tools" for s in SLUGS]:
        for c in CFGS:
            few = rd = tot = 0
            ks, reps = [], []
            t = TASKS[c]; d = len(t.factors)
            for f in sorted(glob.glob(str(R / f"transcript_{slug}_{c}_s*.json"))):
                seed = int(re.search(r"_s(\d+)\.json$", f).group(1))
                env = SlowLabEnv(t, seed=seed)
                for p, _ in designs_from(f, env):
                    p = np.atleast_2d(p)
                    u = np.unique(np.round(p, 6), axis=0)
                    tot += 1; few += (len(u) < d + 1); ks.append(len(u))
                    reps.append(len(p) / len(u))
                    rd += (np.linalg.matrix_rank(np.c_[np.ones(len(u)), u]) < d + 1)
            if tot: diag[(slug, c)] = (100*few/tot, 100*rd/tot, tot, float(np.median(ks)),
                                       float(np.median(reps)))
    for c in CFGS:
        b = np.mean([diag[(s, c)][1] for s in SLUGS if (s, c) in diag])
        t_ = np.mean([diag[(s+"+tools", c)][1] for s in SLUGS if (s+"+tools", c) in diag])
        fw = np.mean([diag[(s, c)][0] for s in SLUGS if (s, c) in diag])
        md = np.median([diag[(s, c)][3] for s in SLUGS if (s, c) in diag])
        rp = np.median([diag[(s, c)][4] for s in SLUGS if (s, c) in diag])
        print(f"  {c:9s} rank-deficient bare {b:5.1f}%  tooled {t_:5.1f}%   "
              f"k < d+1 bare {fw:5.1f}%   median distinct treatments {md:.0f}   "
              f"median units per treatment {rp:.0f}")
    print()
    say("unaided, 99% rank-deficient on Screen, 92% on Transfer, 54% on Optimise; "
        "with tools 59% and 64%; k < d+1 in 98% / 87% / 50% of those designs; "
        "the median design uses two distinct treatments, each carrying a median of "
        "three to twelve units depending on the task",
        "see the table above")

    print("=" * 78)
    print("D. Tools (paired)")
    dirn = {"worse": 0, "better": 0, "sig": 0, "sig_worse": 0}
    per_task = {}
    for c in CFGS:
        rows = []
        for s in SLUGS:
            a, b = E.get((s, c), {}), E.get((s+"+tools", c), {})
            ks = sorted(set(a) & set(b))
            if len(ks) < 10: continue
            x = np.array([a[k]["regret"] for k in ks])
            y = np.array([b[k]["regret"] for k in ks])
            t = stats.ttest_rel(y, x)
            P = CV[c]["prior"]; A = CV[c]["agents"]
            dc = 100*((1-A[s+"+tools"]["R_star"]/P) - (1-A[s]["R_star"]/P))
            dr = 100*(y.mean()-x.mean())/x.mean()
            rows.append((s, dc, dr, t.pvalue))
            dirn["worse" if dr > 0 else "better"] += 1
            if t.pvalue < .05:
                dirn["sig"] += 1; dirn["sig_worse"] += (dr > 0)
        per_task[c] = rows
    say("16 of 20 worse, 4 better, 6 at p<0.05 — four worse, two better (both Optimise)",
        f"{dirn['worse']} worse / {dirn['better']} better; {dirn['sig']} significant, "
        f"of which {dirn['sig_worse']} worse")
    for c in ("Screen", "Transfer"):
        dr = [r[2] for r in per_task[c]]
        dc = [r[1] for r in per_task[c]]
        print(f"  {c:9s} ΔR̄ {min(dr):+.0f}%..{max(dr):+.0f}%  "
              f"({sum(1 for v in dr if v>0)}/{len(dr)} worse)   "
              f"dc {min(dc):+.1f}..{max(dc):+.1f} ({sum(1 for v in dc if v>0)}/{len(dc)} up)")
    say("realised regret rises in all five Screen and all five Transfer cells, by 4% to 59%;"
        " design efficiency rises in seven of ten, by up to 6.5 points, others −0.5..−0.2",
        "see the two lines above")

    print("=" * 78)
    print("E. The deference regression (produced by scripts/tool_regret.py)")
    tp = ROOT / "results" / f"tool_regret_env{slowlab.ENV_VERSION}.json"
    if not tp.exists():
        raise SystemExit("run python scripts/tool_regret.py first")
    T = json.loads(tp.read_text())
    say("slope 0.29 ± 0.09, r = 0.62, p = 0.004 over twenty cells; sign predicted in 18/20",
        f"slope {T['slope']:.2f} ± {T['stderr']:.2f}, r = {T['r']:.2f}, "
        f"p = {T['p']:.4f}, n = {T['n_cells']}; sign correct in {T['sign_predicted']}, "
        f"tool worse than model in {T['tool_worse_than_model']}")
    say("tool's own proposal carries 0.0219 on Optimise and 0.0901 on Transfer",
        {k: round(v, 4) for k, v in T["by_task"].items()})
    say("on Sanity the tool is worse than making no observation at all",
        f"Sanity tool {T['by_task']['Sanity']:.4f} vs R*(none) {CV['Sanity']['prior']:.4f}")

    print("=" * 78)
    print("F. Transfer after the shock")
    fac = []
    for s in SLUGS:
        for suf in ("", "+tools"):
            es = list(E.get((s+suf, "Transfer"), {}).values())
            if not es: continue
            pre = np.mean([e["regret"] for e in es])
            post = [e.get("transfer_regret") for e in es]
            if any(v is None for v in post): continue
            fac.append((s+suf, np.mean(post)/pre))
    if fac:
        v = [f for _, f in fac]
        say("every model worse after the shock, by factors of 1.06 to 1.98",
            f"{min(v):.2f}x .. {max(v):.2f}x, n = {len(v)}; "
            f"{sum(1 for x in v if x < 1)} of them below 1")
        bare = {n: f for n, f in fac if not n.endswith("+tools")}
        tool = {n[:-6]: f for n, f in fac if n.endswith("+tools")}
        both = [(k, bare[k], tool[k]) for k in bare if k in tool]
        say("tooled arms worse than bare in every case",
            f"tooled worse in {sum(1 for _, b_, t_ in both if t_ > b_)}/{len(both)}")
    else:
        say("post-shock factors", "episodes carry no regret_transfer field")
    tp = ROOT / "results" / f"transfer_env{slowlab.ENV_VERSION}.json"
    if tp.exists():
        T4 = json.loads(tp.read_text())
        pr, co = T4["gp_ucb_profit"], T4["gp_ucb_components"]
        say("two reference agents differ by 1.47x after the shock (0.0605 vs 0.0411, t=5.5)",
            f"{pr['transfer']:.4f} / {co['transfer']:.4f} = "
            f"{pr['transfer']/co['transfer']:.2f}x, t = {T4['paired']['t']:.1f}, "
            f"{T4['paired']['n_worse']}/{T4['paired']['n']} sites")

    print("=" * 78)
    print("G. The first round of data makes the recommendation worse")
    ep = worse_cell = tot_ep = 0
    for s in ARMS:
        for c in CFGS:
            es = list(E.get((s, c), {}).values())
            if not es: continue
            r1 = [e["trace"][0] for e in es if e.get("trace")]
            z = [e["regret_zero_shot"] for e in es if e.get("trace")]
            if not r1: continue
            ep += sum(1 for a, b_ in zip(r1, z) if a > b_); tot_ep += len(r1)
            worse_cell += (np.mean(r1) > np.mean(z))
    if tot_ep:
        say("in 44% of episodes round-1 is worse than 0-shot; 24 of 40 cells worse on the mean",
            f"{100*ep/tot_ep:.0f}% of {tot_ep} episodes; {worse_cell}/40 cells")
    else:
        say("round-1 regret", "episodes carry no regret_by_round")

    print("=" * 78)
    print("H. Within-task rank correlations (the two regrets, and c against eta)")
    # These carry the claim that the measures are not restatements of each other, so
    # they need a line here like every other number in the section.
    for c in CFGS:
        P = CV[c]["prior"]; A = CV[c]["agents"]
        rows = []
        for s in SLUGS:
            es = list(E.get((s, c), {}).values())
            if not es or s not in A: continue
            rbar = np.mean([e["regret"] for e in es])
            cum = np.mean([e["cumulative_regret"] for e in es])
            rows.append((rbar, cum, 1 - A[s]["R_star"]/P, A[s]["R_star"]/rbar))
        if len(rows) < 3: continue
        rbar, cum, cc, eta = map(np.array, zip(*rows))
        print(f"  {c:9s} rho(R-bar, R_cum) = {stats.spearmanr(rbar, cum).statistic:+.2f}   "
              f"rho(c, eta) = {stats.spearmanr(cc, eta).statistic:+.2f}   n = {len(rows)}")
    print()
    say("R-bar and R_cum rank-correlate at -0.80 on Sanity and -0.60 on Optimise; "
        "c and eta at +0.80 on Optimise and -0.80 on Transfer",
        "see the table above")


if __name__ == "__main__":
    main()
