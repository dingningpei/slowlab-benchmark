#!/usr/bin/env python3
"""Read local transcripts and measure what the designs LLMs submit actually look like.

The scores tell you how well an agent did, never *why*. This script reconstructs
every accepted design and computes four things:

  1. collinearity between factors -- do two factors move together (together means
     their effects are not separable)
  2. within-chamber variation -- how many values a loop-level factor takes inside
     each chamber (1 means it is confounded with the chamber)
  3. replication structure -- pure-error degrees of freedom, sum of (n_i - 1)
  4. the specification checklist -- per-item pass rates of the diagnostic
     checklist in the appendix

Usage:
    python3 scripts/diagnose_llm.py --model deepseek-chat
    python3 scripts/diagnose_llm.py --model deepseek-chat --out diag.json
"""
from __future__ import annotations
import argparse, json, pathlib, re, sys
from collections import Counter, defaultdict
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab import SlowLabEnv, TASKS
from slowlab.llm import _extract_json, _to_design
from slowlab.validity import score_validity, ValidityReport


def analyse_design(env, blob):
    """Diagnose one model reply at the design level. Returns None if the reply was not a design submission."""
    if "treatments" not in blob or "allocation" not in blob:
        return None
    task = env.task
    tr, al = blob["treatments"], blob["allocation"]
    names = [f.name for f in task.factors]
    try:
        pts = np.array([[float(tr[t][n]) for n in names] for t in tr], float)
    except Exception:
        return None
    out = {"n_treatments": len(tr), "n_units": sum(len(v) for v in al.values())}

    # 1. Largest |correlation| between factors. Two treatments are collinear by
    #    necessity (two points define a line), so the number carries no
    #    information there; it is computed only when there are >=3 treatments, and
    #    designs with fewer are counted separately. Recording those as correlation
    #    0, as we did at first, was wrong -- it laundered collinear designs into
    #    non-collinear ones.
    out["corr_defined"] = bool(pts.shape[0] > 2 and len(names) > 1)
    worst, pair = None, None
    if out["corr_defined"]:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = pts[:, i], pts[:, j]
                if a.std() < 1e-9 or b.std() < 1e-9:
                    continue
                r = abs(float(np.corrcoef(a, b)[0, 1]))
                if worst is None or r > worst:
                    worst, pair = r, (names[i], names[j])
    out["max_factor_corr"] = worst
    out["corr_pair"] = pair

    # 2. How many values a loop-level factor takes inside each chamber
    loop_names = [f.name for f in task.factors if f.control_level == "loop"]
    within = {}
    for fn in loop_names:
        by_ch = defaultdict(set)
        for tid, us in al.items():
            for u in us:
                m = re.match(r"c(\d+)l(\d+)", str(u))
                if m:
                    by_ch[m.group(1)].add(round(float(tr[tid][fn]), 9))
        within[fn] = sorted(len(v) for v in by_ch.values())
    out["within_chamber_levels"] = within

    # 3. Pure-error degrees of freedom
    reps = Counter({t: len(v) for t, v in al.items()})
    out["pure_error_df"] = sum(max(0, n - 1) for n in reps.values())
    out["min_reps"] = min(reps.values()) if reps else 0

    # 4. The specification checklist
    try:
        d = _to_design(env, blob)
        rep = score_validity(d, env.facility, task)
        out["checks"] = {k: bool(rep.checks.get(k, False))
                         for k in ValidityReport.MANDATORY}
    except Exception:
        out["checks"] = None
    return out


def recommendation_vs_own_best(env, turns):
    """How far is the final recommendation from the best treatment the agent itself
    observed?

    If the designs are sound but regret looks like random search, the failure may
    lie in the reading rather than the design: not even adopting the best point it
    measured. Distance is Euclidean in the *normalised* factor space, and we also
    report the difference between the recommendation's true value and the best
    observed point's true value, in margin units.
    """
    obs = env.observations()
    if not obs:
        return None
    best_o = max(obs, key=lambda o: o.value)
    d = env._designs[best_o.design_id]
    x_obs = np.array([d.treatments[best_o.treatment][f.name] for f in env.task.factors])
    rec = None
    for turn in reversed(turns):
        try:
            b = _extract_json(turn["assistant"])
        except Exception:
            continue
        fv = b.get("recommendation") or b.get("current_best")
        if fv:
            try:
                rec = np.array([(float(fv[f.name]) - f.low) / (f.high - f.low)
                                for f in env.task.factors])
                break
            except Exception:
                continue
    if rec is None:
        return None
    rec = np.clip(rec, 0, 1)
    y_rec = float(env.truth(rec.reshape(1, -1))[0])
    y_obs = float(env.truth(x_obs.reshape(1, -1))[0])
    return {"dist_to_own_best": float(np.linalg.norm(rec - x_obs)),
            "profit_gap_vs_own_best": y_obs - y_rec,
            "rec_worse_than_own_best": bool(y_rec < y_obs)}


def coverage(env, turns):
    """How much of each factor's range the agent actually probed, normalised."""
    xs = []
    for o in env.observations():
        d = env._designs[o.design_id]
        xs.append([d.treatments[o.treatment][f.name] for f in env.task.factors])
    if not xs:
        return None
    a = np.array(xs, float)
    return float(np.mean(a.max(0) - a.min(0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--dir", default="results/llm")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    outdir = ROOT / a.dir
    files = sorted(outdir.glob("transcript_*.json"))
    if not files:
        raise SystemExit(f"no transcript_*.json under {outdir}")

    per_task = defaultdict(list)
    per_ep = defaultdict(list)
    for f in files:
        m = re.match(r"transcript_(.+)_s(\d+)\.json", f.name)
        if not m:
            continue
        tname, seed = m.group(1), int(m.group(2))
        if tname not in TASKS:
            continue
        env = SlowLabEnv(TASKS[tname], seed=seed)
        turns = json.loads(f.read_text())
        # Replay the episode so env holds the observations (for recommendation vs best observed)
        for turn in turns:
            try:
                b = _extract_json(turn["assistant"])
                des = _to_design(env, b)
            except Exception:
                continue
            if isinstance(env.submit_design(des), int):
                env.advance()
        ep = recommendation_vs_own_best(env, turns)
        if ep:
            ep["coverage"] = coverage(env, turns)
            per_ep[tname].append(ep)
        env = SlowLabEnv(TASKS[tname], seed=seed)
        for turn in turns:
            try:
                blob = _extract_json(turn["assistant"])
            except Exception:
                continue
            r = analyse_design(env, blob)
            if r:
                per_task[tname].append(r)

    summary = {}
    print(f"model {a.model} -- {sum(len(v) for v in per_task.values())} designs in total\n")
    hdr = (f"{'config':<12}{'designs':>9}{'collinear':>11}{'>0.9':>7}"
           f"{'flat-in-ch':>12}{'pure-df':>9}{'checklist':>11}")
    print(hdr); print("-" * len(hdr.encode('gbk', 'ignore')))
    for t, rows in per_task.items():
        cr = np.array([r["max_factor_corr"] for r in rows
                       if r.get("corr_defined") and r["max_factor_corr"] is not None])
        n_two = sum(1 for r in rows if not r.get("corr_defined"))
        df = np.array([r["pure_error_df"] for r in rows])
        flat = []          # loop-level factor takes one value in *every* chamber -> confounded with chamber
        for r in rows:
            w = r["within_chamber_levels"]
            flat.append(bool(w) and all(max(v) <= 1 for v in w.values() if v))
        ok = [r["checks"] for r in rows if r["checks"]]
        pass_rate = np.mean([all(c.values()) for c in ok]) if ok else float("nan")
        summary[t] = {
            "n_designs": len(rows),
            "n_corr_defined": int(len(cr)),
            "frac_two_treatment_designs": float(n_two / max(len(rows), 1)),
            "mean_max_factor_corr": float(cr.mean()) if len(cr) else None,
            "frac_corr_gt_0.9": float((cr > 0.9).mean()) if len(cr) else None,
            "frac_loop_factor_flat_in_all_chambers": float(np.mean(flat)),
            "mean_pure_error_df": float(df.mean()),
            "frac_pure_error_df_lt_2": float((df < 2).mean()),
            "checklist_pass_rate": float(pass_rate),
            "per_check_pass": {k: float(np.mean([c[k] for c in ok])) for k in ok[0]} if ok else {},
        }
        s_row = summary[t]
        c_txt = "n/a" if s_row["mean_max_factor_corr"] is None else f"{s_row['mean_max_factor_corr']:.2f}"
        g_txt = "n/a" if s_row["frac_corr_gt_0.9"] is None else f"{s_row['frac_corr_gt_0.9']*100:.0f}%"
        print(f"{t:<12}{len(rows):>7}{c_txt:>10}{g_txt:>10}"
              f"{s_row['frac_loop_factor_flat_in_all_chambers']*100:>11.0f}%"
              f"{s_row['mean_pure_error_df']:>10.1f}{s_row['checklist_pass_rate']*100:>9.0f}%")

    print("\n(collinearity is computed only on designs with >=3 treatments; two-treatment designs are collinear by necessity and are listed separately)")
    for t, s_ in summary.items():
        print(f"  {t:<12} two-treatment designs {s_['frac_two_treatment_designs']*100:>3.0f}%"
              f"   designs where correlation is defined: {s_['n_corr_defined']}")

    print("\nRecommendation vs the best point the agent itself observed:")
    print(f"  {'config':<12}{'factor dist':>13}{'margin gap':>12}{'rec is worse':>14}{'coverage':>10}")
    for t, eps in per_ep.items():
        d_ = np.array([e["dist_to_own_best"] for e in eps])
        g_ = np.array([e["profit_gap_vs_own_best"] for e in eps])
        w_ = np.mean([e["rec_worse_than_own_best"] for e in eps])
        c_ = np.mean([e["coverage"] for e in eps if e["coverage"] is not None])
        summary[t].update({"dist_to_own_best": float(d_.mean()),
                           "profit_gap_vs_own_best": float(g_.mean()),
                           "frac_rec_worse_than_own_best": float(w_),
                           "coverage": float(c_)})
        print(f"  {t:<12}{d_.mean():>14.3f}{g_.mean():>+12.4f}{w_*100:>15.0f}%{c_:>10.2f}")

    print("\nPer-item pass rates:")
    for t, row in summary.items():
        if row["per_check_pass"]:
            print(f"  {t:<12}" + "  ".join(f"{k}={v*100:.0f}%" for k, v in row["per_check_pass"].items()))

    out = pathlib.Path(a.out) if a.out else outdir / f"diagnostics_{a.model}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
