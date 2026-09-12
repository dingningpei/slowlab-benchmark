#!/usr/bin/env python3
"""T4, transfer of economic conditions: answering correctly requires understanding,
while the score remains regret.

After the campaign, energy prices rise, no new budget is granted, and the only
question is what the agent recommends now. A controlled pair: same algorithm,
same budget, same observations, differing only in whether they model *profit* or
*revenue and cost as two components*.
"""
from __future__ import annotations
import sys, pathlib, json
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import slowlab
from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent


def main(seeds=40, agents=("gp_ucb_profit", "gp_ucb_components")):
    out = {}
    per = {}
    print(f"T4, {seeds} sites. Training energy price x{TASKS['Transfer'].train_energy_shock or 1.0}"
          f" -> asked at x{TASKS['Transfer'].transfer_energy_shock}\n")
    print(f"{'agent':22}{'training regret':>17}{'post-shock regret':>19}")
    for a in agents:
        S, T = [], []
        for s in range(seeds):
            e = SlowLabEnv(TASKS["Transfer"], seed=s)
            ag = make_agent(a)
            x = ag.run(e)
            r = e.submit_recommendation(x, a, x_transfer=getattr(ag, "x_transfer", x))
            S.append(r.simple_regret); T.append(r.transfer_regret)
        per[a] = {"simple": [float(v) for v in S], "transfer": [float(v) for v in T]}
        out[a] = {"simple": float(np.mean(S)), "simple_se": float(np.std(S, ddof=1)/np.sqrt(seeds)),
                  "transfer": float(np.mean(T)), "transfer_se": float(np.std(T, ddof=1)/np.sqrt(seeds))}
        o = out[a]
        print(f"{a:22}{o['simple']:>10.4f}±{o['simple_se']:.4f}{o['transfer']:>10.4f}±{o['transfer_se']:.4f}")
    # Paired comparison: same site, same observations, differing only in the modelling choice
    dS = np.array(per[agents[0]]["simple"]) - np.array(per[agents[1]]["simple"])
    dT = np.array(per[agents[0]]["transfer"]) - np.array(per[agents[1]]["transfer"])
    n = len(dT)
    print(f"\nPaired difference (profit - components) over {n} sites:")
    print(f"  training   {dS.mean():+.4f} +/- {dS.std(ddof=1)/np.sqrt(n):.4f}"
          f"   (largest |difference| = {np.abs(dS).max():.6f})")
    tt = dT.mean() / (dT.std(ddof=1) / np.sqrt(n))
    print(f"  post-shock {dT.mean():+.4f} +/- {dT.std(ddof=1)/np.sqrt(n):.4f}   t = {tt:.2f}")
    print(f"  sites where profit-only is worse after the shock: {int((dT > 0).sum())}/{n}")
    out["per_seed"] = per
    out["paired"] = {"transfer_diff": float(dT.mean()),
                     "transfer_diff_se": float(dT.std(ddof=1)/np.sqrt(n)),
                     "t": float(tt), "n_worse": int((dT > 0).sum()), "n": n}
    (ROOT / "results" / f"transfer_env{slowlab.ENV_VERSION}.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
