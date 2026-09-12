#!/usr/bin/env python3
"""Rescore after the oracle fix, without rerunning any agent.

Nothing the agent sees -- factor ranges, facility, budget, observations -- passes
through the oracle, so its designs and recommendations are unchanged word for
word and rerunning would only add sampling noise. What changed is the reference
point f(x*). Since every regret-like quantity is f(x*) - f(x) and f(x) is
unchanged, one shift per (task, seed) suffices:

    Δ(task, seed) = oracle_new - oracle_old

Every entry of the trace, regret and regret_zero_shot shift by the same amount;
cumulative_regret is delta_days x sum_units [f(x*) - f(a)], so its shift is
multiplied by the cycle length and the number of units actually occupied.

We do this rather than leave it alone because the spread in one table already uses
the new oracle while the regret in another still used the old one -- mixing two
versions is exactly the class of error we have been caught by before.
"""
from __future__ import annotations
import sys, json, glob, pathlib, re
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab import SlowLabEnv, TASKS


def _oracle_old(w, n_sample=6000, seed=0):
    """The pre-fix implementation: single start, two-stage refinement. Kept verbatim so the shift can be computed."""
    rng = np.random.default_rng(seed)
    Xs = rng.random((n_sample, w.d)); y = w(Xs)
    i = int(np.argmax(y)); x0, best = Xs[i], float(y[i])
    for r in (0.10, 0.03):
        Xl = np.clip(x0 + rng.normal(0, r, (1500, w.d)), 0, 1)
        yl = w(Xl); j = int(np.argmax(yl))
        if yl[j] > best:
            x0, best = Xl[j], float(yl[j])
    return best


def deltas(task_name, seeds):
    t = TASKS[task_name]
    out = {}
    for s in seeds:
        w = SlowLabEnv(t, seed=s).truth
        out[s] = float(w.oracle()[1]) - _oracle_old(w)
    return out


def main(out="results/llm"):
    d = ROOT / out
    files = sorted(glob.glob(str(d / "episodes_*.json")))
    tasks = {}
    for f in files:
        rows = json.loads(pathlib.Path(f).read_text())
        for r in rows:
            tasks.setdefault(r["task"], set()).add(r["seed"])
    D = {T: deltas(T, sorted(ss)) for T, ss in tasks.items()}
    print("Shift per task (median / maximum)")
    for T, m in D.items():
        v = np.array(list(m.values()))
        print(f"  {T:10} {np.median(v):+.6f} / {np.abs(v).max():+.6f}")

    n = 0
    for f in files:
        p = pathlib.Path(f)
        rows = json.loads(p.read_text())
        for r in rows:
            dd = D[r["task"]][r["seed"]]
            t = TASKS[r["task"]]
            env = SlowLabEnv(t, seed=0)
            units = min(t.units_per_round, len(env.facility.units))
            for k in ("regret", "regret_zero_shot", "transfer_regret"):
                if r.get(k) is not None and np.isfinite(r[k]):
                    r[k] += dd
            if r.get("trace"):
                r["trace"] = [x + dd for x in r["trace"]]
            if r.get("cumulative_regret") is not None:
                r["cumulative_regret"] += (dd * t.cycle_days * units
                                           * r.get("rounds_submitted", t.n_rounds))
            n += 1
        p.write_text(json.dumps(rows, indent=1))
    print(f"\nrescored {n} episodes, rewrote {len(files)} files")

    # summary_*.json is derived from episodes, so recompute it in step
    for f in sorted(glob.glob(str(d / "summary_*.json"))):
        p = pathlib.Path(f)
        tag = re.sub(r"^summary_|\.json$", "", p.name)
        ep = d / f"episodes_{tag}.json"
        if not ep.exists():
            continue
        rows = json.loads(ep.read_text())
        s = json.loads(p.read_text())
        for T in s:
            v = [x for x in rows if x["task"] == T]
            if not v:
                continue
            g = lambda k: float(np.mean([x[k] for x in v if x.get(k) is not None]))
            s[T]["regret"] = g("regret")
            s[T]["regret_se"] = float(np.std([x["regret"] for x in v], ddof=1)
                                      / len(v) ** .5)
            s[T]["regret_zero_shot"] = g("regret_zero_shot")
            s[T]["cumulative_regret"] = g("cumulative_regret")
        p.write_text(json.dumps(s, indent=1))
    print("summary_*.json brought up to date")


if __name__ == "__main__":
    main()
