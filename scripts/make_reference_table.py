"""Reference-policy table: realised regret, R*(D), the two efficiencies, and the
headline metrics *as ranks*.

Re-run after environment v1.0 was frozen. Design measurements are reported only
as ranks (see rule 3 of ENVIRONMENT_v1.0.md), so the levels of c and eta are
marked uncalibrated and only the ordering is a conclusion.

    python scripts/make_reference_table.py            # print + write JSON/LaTeX
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import slowlab
from slowlab import SlowLabEnv, TASKS
from slowlab.registry import make_agent

REFS = ("random_spread", "classical_doe", "gp_ucb_rep1", "gp_ucb_rep2")
CFGS = ("Sanity", "Screen", "Optimise", "Transfer")
LABEL = {"random_spread": "Random search", "classical_doe": "Planned campaign",
         "gp_ucb_rep1": "Bayesian opt.\\ (1 rep)",
         "gp_ucb_rep2": "Bayesian opt.\\ (2 rep)"}


def realised(cfg, agent, seeds):
    """Take both the final regret and the cumulative regret over the active period --
    the latter is what the irreversibility dimension produces."""
    sr, cr = [], []
    for s in range(seeds):
        env = SlowLabEnv(TASKS[cfg], seed=s)
        a = make_agent(agent)
        r = env.submit_recommendation(a.run(env), agent,
                                      probes=getattr(a, "probes", None))
        sr.append(r.simple_regret); cr.append(r.cumulative_regret)
    return np.array(sr, float), np.array(cr, float)


def main(M=320, seeds=8):
    # Cross-validated estimator (tempering + leave-one-out calibration + held-out
    # evaluation). The old in-sample files must not be reused; see the 2026-09-11
    # note at the top of slowlab/achievable.py.
    src = ROOT / "results" / f"achievable_cv_env{slowlab.ENV_VERSION}_M{M}.json"
    if not src.exists():
        raise SystemExit(f"missing {src.name}; run rescore_cv.py --refs-only first")
    ach = json.loads(src.read_text())

    res = {"env_version": slowlab.ENV_VERSION, "M": M, "seeds": seeds, "tasks": {}}
    for cfg in CFGS:
        prior = ach[cfg]["prior"]
        rows = {}
        for ag in REFS:
            Rs = ach[cfg]["agents"][f"ref:{ag}"]["R_star"]
            rr, cc = realised(cfg, ag, seeds)
            rows[ag] = {"realised": float(rr.mean()),
                        "se": float(rr.std(ddof=1) / np.sqrt(len(rr))),
                        "cum": float(cc.mean()),
                        "R_star": Rs,
                        "c": (prior - Rs) / prior,
                        "eta": Rs / rr.mean()}
        # The headline is the *order*: design rank and decision rank (1 = best).
        # Ties share a rank. The tolerance is the display precision, one percentage
        # point: between M=80 and M=320 the estimator holds only the 0.81-0.92 level
        # steady, so differences of a few parts per thousand are far below the
        # resolution and giving them distinct ranks would be false precision.
        for key, field in (("design_rank", "c"), ("decision_rank", "eta")):
            q = {a: round(rows[a][field] * 100) for a in REFS}
            vals = sorted(set(q.values()), reverse=True)
            for a in REFS:
                rows[a][key] = vals.index(q[a]) + 1
        res["tasks"][cfg] = {"prior": prior, "agents": rows}

    (ROOT / "results" / f"reference_table_env{slowlab.ENV_VERSION}.json").write_text(
        json.dumps(res, indent=1))

    print(f"env {slowlab.ENV_VERSION}  M={M}  seeds={seeds}")
    print(f"{'task':9s} {'agent':18s} {'R̄(R)':>9s} {'R*(D)':>8s} "
          f"{'c':>5s} {'rank':>5s} {'η':>5s} {'rank':>5s}")
    for cfg in CFGS:
        d = res["tasks"][cfg]
        print(f"  R*(∅) = {d['prior']:.4f}")
        for ag in REFS:
            r = d["agents"][ag]
            print(f"{cfg:9s} {ag:18s} {r['realised']:9.4f} {r['R_star']:8.4f} "
                  f"{100*r['c']:4.0f}% {r['design_rank']:5d} "
                  f"{100*r['eta']:4.0f}% {r['decision_rank']:5d}")

    # LaTeX
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
         r"\textbf{Task} & \textbf{Strategy} & $\bar{\mathcal{R}}(R)\downarrow$ &"
         r" $\mathcal{R}_{\mathrm{cum}}\downarrow$ & $c\uparrow$ & $\eta\uparrow$ &"
         r" design & decision \\",
         r" & & final answer & campaign cost & & & rank & rank \\", r"\midrule"]
    for cfg in CFGS:
        d = res["tasks"][cfg]
        L.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\textsc{{{cfg}}}"
                 rf" \quad \small $\mathcal{{R}}^\star(\varnothing)={d['prior']:.4f}$}} \\")
        for ag in REFS:
            r = d["agents"][ag]
            L.append(f"& {LABEL[ag]} & {r['realised']:.4f}({r['se']*1e4:.0f}) & "
                     f"{r['cum']:.0f} & {100*r['c']:.0f}\\% & "
                     f"{100*r['eta']:.0f}\\% & "
                     f"{r['design_rank']} & {r['decision_rank']} \\\\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Reference strategies, environment v" + slowlab.ENV_VERSION +
          r", eight instances per task. $\bar{\mathcal{R}}(R)$ is the regret of the final"
          r" recommendation, with its standard error in the last two places;"
          r" $\mathcal{R}_{\mathrm{cum}}$ is the profit the campaign forwent, in"
          r" \euro\,per square metre summed over committed unit-days. $c$ is design"
          r" efficiency against the $\mathcal{R}^\star(\varnothing)$ printed above each"
          r" block. Rank $1$ is best and tied values share a rank; decision rank orders"
          r" $\eta$, whose levels are in Appendix~\ref{app:eig}. We draw conclusions"
          r" from the ranks rather than the levels, for the reason given in"
          r" \S\ref{sec:estimator}.}",
          r"\label{tab:refs}", r"\end{table}"]
    (ROOT / "paper" / "sections" / "_reference_table.tex").write_text(
        "\n".join(L) + "\n")
    print("\nwrote results/reference_table_env*.json and "
          "paper/sections/_reference_table.tex")


if __name__ == "__main__":
    main()
