#!/usr/bin/env python3
"""Run the baselines and write the results as CSV / JSON for the paper's tables and figures.

    python scripts/run_baselines.py --seeds 30 --out results/
"""
from __future__ import annotations
import argparse, json, csv, sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent, available


def run_episode(task_name: str, agent_name: str, seed: int) -> dict:
    env = SlowLabEnv(TASKS[task_name], seed=seed)
    agent = make_agent(agent_name)
    x = agent.run(env)
    r = env.submit_recommendation(x, agent_name, probes=getattr(agent, "probes", None))
    return {
        "task": task_name, "agent": agent_name, "seed": seed,
        "simple_regret": r.simple_regret,          # Outcome
        "campaign_cash": r.campaign_cash,
        "cumulative_regret": r.cumulative_regret,  # Outcome
        "churn": r.churn,
        "gain": r.gain, "backslide": r.backslide,
        "wasted_fraction": r.wasted_fraction,      # appendix diagnostic
        "wasted_rounds": r.wasted_rounds,
        "committed_rounds": r.committed_rounds,
        "validity_rate": r.validity_rate,          # appendix diagnostic, not in the body
        "n_designs": r.n_designs, "n_valid": r.n_valid,
        "unit_days_used": r.unit_days_used,
        "parallel_utilisation": r.parallel_utilisation,
        "n_rejections": len(r.rejections),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T3", "T4"])
    # The four reference points. The rep2 / transfer pair is a *result*, not a scale
    # marker, and is not run here.
    ap.add_argument("--agents", nargs="+",
                    default=["random_spread", "classical_doe", "gp_ucb_rep1"])
    ap.add_argument("--out", type=str, default="results")
    args = ap.parse_args()

    out = pathlib.Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = [run_episode(t, a, s)
            for t in args.tasks for a in args.agents for s in range(args.seeds)]

    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    summary = {}
    for t in args.tasks:
        summary[t] = {}
        for a in args.agents:
            sub = [r for r in rows if r["task"] == t and r["agent"] == a]
            reg = np.array([r["simple_regret"] for r in sub])
            summary[t][a] = {
                "regret_mean": float(reg.mean()),
                "regret_se": float(reg.std(ddof=1) / np.sqrt(len(reg))),
                "validity_rate": float(np.mean([r["validity_rate"] for r in sub])),
                "cumulative_regret": float(np.mean([r["cumulative_regret"] for r in sub])),
                "cumulative_regret_se": float(np.std([r["cumulative_regret"] for r in sub], ddof=1)
                                              / len(sub) ** .5),
                "churn": float(np.nanmean([r["churn"] for r in sub])),
                "churn_median": float(np.nanmedian([r["churn"] for r in sub])),
                "monotone_frac": float(np.mean([(r["backslide"] or 0) < 1e-12
                                                for r in sub if r["gain"] == r["gain"]])),
                "campaign_cash": float(np.mean([r["campaign_cash"] for r in sub])),
                "wasted_fraction": float(np.nanmean([r["wasted_fraction"] for r in sub])),
                "parallel_utilisation": float(np.mean([r["parallel_utilisation"] for r in sub])),
                "n_seeds": len(sub),
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    for t in args.tasks:
        task = TASKS[t]
        print(f"\n── {t}: {task.d} factors · {task.n_rounds}×{task.units_per_round} units "
              f"· plant CV={task.plant_cv} ──")
        print(f"{'agent':<18}{'Slow: regret':>16}{'Costly: cash':>14}{'Irrev: wasted':>15}")
        print("-" * 63)
        for a in args.agents:
            s = summary[t][a]
            w = s["wasted_fraction"]
            print(f"{a:<18}{s['regret_mean']:>10.4f}±{s['regret_se']:.4f}"
                  f"{s['campaign_cash']:>14.1f}"
                  f"{('n/a' if w != w else f'{w*100:.0f}%'):>15}")
    print(f"\nwrote {out/'episodes.csv'} and {out/'summary.json'}")


if __name__ == "__main__":
    main()
