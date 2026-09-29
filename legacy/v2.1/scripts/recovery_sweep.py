#!/usr/bin/env python3
"""Price the irreversibility axis: what does a recovery period after heat damage cost?

`recovery_days` is a hyperparameter in days. A unit whose day temperature exceeds
TCRIT + damage_margin enters a recovery period at the end of its round, graded by how far
it went over and capped at recovery_days. A cycle is duration_days, so:
  0                 off
  < duration_days   blocks part of the next round
  > duration_days   blocks two rounds in a row
The lost unit-days cannot be bought back with money, which is what separates this axis from
the Costly one.

All four tasks of v1.0.0 set it to zero, so the eight hundred episodes the paper reports
carry irreversibility through commitment alone. This sweep turns the dial on for the
scripted references, which are deterministic given a seed and need no model calls, so that
the condition stops being an assertion about the domain and becomes a measurement. The
language-model half of the same ablation is a separate run; see --recovery-days in
scripts/run_llm.py.

The sweep does not change the frozen environment: it constructs modified copies of the
tasks and writes to its own file. Nothing here is comparable with Table 3, which is at
recovery_days = 0.

    python scripts/recovery_sweep.py
"""
from __future__ import annotations
import sys, pathlib, json, copy, time, argparse
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import slowlab
from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent

AGENTS = ["random_spread", "classical_doe", "gp_ucb_rep1", "gp_ucb_rep2"]
CFGS = ("Sanity", "Screen", "Optimise", "Transfer")


def run(task, a, seeds=20):
    R, L, C, J, U = [], [], [], [], []
    for s in range(seeds):
        e = SlowLabEnv(task, seed=s)
        ag = make_agent(a)
        r = e.submit_recommendation(ag.run(e), a)
        R.append(r.simple_regret)
        L.append(r.lost_unit_days)
        C.append(r.campaign_cash)
        J.append(len(r.rejections))
        U.append(r.unit_days_used)
    n = len(R)
    return {"regret": float(np.mean(R)),
            "regret_se": float(np.std(R, ddof=1) / np.sqrt(n)),
            "lost_unit_days": float(np.mean(L)),
            "damaged_episodes": int(sum(1 for x in L if x > 0)),
            "unit_days_used": float(np.mean(U)),
            "cash": float(np.mean(C)),
            "rejections": float(np.mean(J)),
            "per_seed_regret": [float(x) for x in R],
            "n": n}


def main(seeds=20, sweep=(0, 60, 210), only=None, out=None):
    out = out or f"results/recovery_sweep_env{slowlab.ENV_VERSION}.json"
    fp = ROOT / out
    # Merge rather than overwrite: the sandbox caps a single call well below the time
    # the four tasks need, so this is run one task at a time.
    res = json.loads(fp.read_text()) if fp.exists() else {}
    t0 = time.time()
    for cfg in (only or CFGS):
        base = TASKS[cfg]
        print(f"\n=== {cfg}   (cycle {base.duration_days} days, "
              f"{base.n_rounds} rounds x {base.units_per_round} units)")
        print(f"{'agent':16}{'rec_days':>9}{'regret':>18}{'lost u-d':>10}"
              f"{'damaged':>9}{'paired vs 0':>13}")
        res[cfg] = {}
        for a in AGENTS:
            base_row = None
            for rd in sweep:
                t = copy.deepcopy(base)
                t.recovery_days = float(rd)
                row = run(t, a, seeds)
                res[cfg].setdefault(str(rd), {})[a] = row
                if rd == sweep[0]:
                    base_row = row
                    delta = ""
                else:
                    d = np.array(row["per_seed_regret"]) - np.array(base_row["per_seed_regret"])
                    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
                    delta = f"{100*d.mean()/base_row['regret']:+.0f}%"
                    delta += " *" if abs(d.mean()) > 2 * se else ""
                print(f"{a if rd == sweep[0] else '':16}{rd:>9}"
                      f"{row['regret']:>12.4f}±{row['regret_se']:.4f}"
                      f"{row['lost_unit_days']:>10.0f}{row['damaged_episodes']:>7}/{row['n']}"
                      f"{delta:>13}")
    fp.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {out}   ({time.time()-t0:.0f}s)")
    print("Note: recovery_days = 0 is the setting of every result in the main tables.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--sweep", type=int, nargs="+", default=[0, 60, 210])
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    main(seeds=a.seeds, sweep=tuple(a.sweep), only=a.only)
