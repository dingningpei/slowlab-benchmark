#!/usr/bin/env python3
"""Build three tables from the v1.0 LLM results: main results, tools, design diagnostics.

c comes from the cross-validated estimator (results/achievable_cv_*), not the old
in-sample files.

    python scripts/make_llm_tables.py
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
SEC = ROOT / "paper" / "sections"
CFGS = ("Sanity", "Screen", "Optimise", "Transfer")
SHORT = {"deepseek_deepseek-v4-flash": "DeepSeek-v4-flash",
         "openai_gpt-5.6-luna": "GPT-5.6-luna",
         "qwen_qwen3.8-27b": "Qwen3.8-27B",
         "xiaomi_mimo-v2.5": "MiMo-v2.5",
         "z-ai_glm-5.3-flash": "GLM-5.3-flash"}
MIN_N = 10          # cells with fewer episodes than this are printed as ---


def load():
    E = {}
    for f in sorted(glob.glob(str(R / "episodes_*.json"))):
        slug = pathlib.Path(f).stem[len("episodes_"):]
        for e in json.loads(pathlib.Path(f).read_text()):
            E.setdefault((slug, e["task"]), {})[e["seed"]] = e
    return E


def cv():
    fp = ROOT / "results" / f"achievable_cv_env{slowlab.ENV_VERSION}_M320.json"
    return json.loads(fp.read_text())


def mean_se(es, key="regret"):
    v = np.array([e[key] for e in es], float)
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)


def tab_results(E, C):
    """Main results: *all four* quantities defined in the evaluation section.

    We used to report only R-bar and c, so eta and cumulative regret never
    appeared in the body at all -- a benchmark paper whose main table omits
    measures it defined itself. The table is now one block per task, with columns
    0-shot / R-bar / R_cum / c / eta.
    """
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"\textbf{Model} & 0-shot & $\bar{\mathcal{R}}\downarrow$"
         r" & $\mathcal{R}_{\mathrm{cum}}\downarrow$ & $c\uparrow$ & $\eta\uparrow$ \\",
         r"\midrule"]
    for j, cfg in enumerate(CFGS):
        if j:
            L.append(r"\addlinespace")
        L.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textsc{{{cfg}}}\quad"
                 rf"\footnotesize $\mathcal{{R}}^\star(\varnothing) = "
                 rf"{C[cfg]['prior']:.4f}$}} \\")
        for slug, nice in SHORT.items():
            es = list(E.get((slug, cfg), {}).values())
            if len(es) < MIN_N:
                L.append(f"{nice} & " + " & ".join(["---"] * 5) + r" \\"); continue
            m, se, n = mean_se(es)
            z = np.mean([e["regret_zero_shot"] for e in es])
            cum = np.mean([e["cumulative_regret"] for e in es])
            A = C[cfg]["agents"][slug]; Rs = A["R_star"]
            cc = 100 * (1 - Rs / C[cfg]["prior"])
            bold = (r"\textbf{", "}") if m < z else ("", "")
            L.append(f"{nice} & {z:.4f} & {bold[0]}{m:.4f}({se*1e4:.0f}){bold[1]}"
                     f" & {cum:.0f} & {cc:.0f}\\% & {100*Rs/m:.0f}\\%" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{The four quantities of \S\ref{sec:eval}, for language models"
          r" without tools, $20$ instances per cell. \emph{0-shot} is the model's own"
          r" recommendation before it sees any data, and is not one of the four."
          r" $\bar{\mathcal{R}}$ \eqref{eq:regret} is realised regret after the full"
          r" campaign, standard error in the last two places, in bold where it beats"
          r" that model's own 0-shot; $\mathcal{R}_{\mathrm{cum}}$ \eqref{eq:cumregret}"
          r" is what the experiments themselves forwent, in the same currency summed over"
          r" occupied slot-days; $c$ and $\eta$ \eqref{eq:capture} are design and"
          r" decision efficiency under the cross-validated estimator of"
          r" \S\ref{sec:estimator}, to be read as ranks. $\eta$ near $100\%$ would mean the"
          r" recommendation is as good as the design allowed. \textsc{Transfer} is scored"
          r" at the training prices here so the four tasks are comparable; the post-shock"
          r" number the task exists for is in \S\ref{sec:results}. No cell reaches"
          r" $\mathcal{R}^\star(\varnothing)$.}",
          r"\label{tab:results}", r"\end{table}"]
    (SEC / "_results_table.tex").write_text("\n".join(L) + "\n")


def tab_tools(E, C):
    """The tools table: design efficiency and realised regret move in opposite directions under one intervention."""
    L = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{@{}ll" + "cccc" + r"@{}}", r"\toprule",
         r"\textbf{Task} & \textbf{Model} & $\Delta c$ & $\Delta\eta$"
         r" & $\Delta\bar{\mathcal{R}}$ & paired $t$ \\",
         r" & & design & decision & final answer & \\", r"\midrule"]
    for cfg in CFGS:
        L.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textsc{{{cfg}}}}} \\")
        for slug, nice in SHORT.items():
            a = E.get((slug, cfg), {}); b = E.get((slug + "+tools", cfg), {})
            ks = sorted(set(a) & set(b))
            if len(ks) < MIN_N:
                L.append(rf"& {nice} & \multicolumn{{4}}{{c}}{{incomplete}} \\")
                continue
            x = np.array([a[k]["regret"] for k in ks])
            y = np.array([b[k]["regret"] for k in ks])
            t = stats.ttest_rel(y, x)
            A = C[cfg]["agents"]; P = C[cfg]["prior"]
            Rb, Rt = A[slug]["R_star"], A[slug + "+tools"]["R_star"]
            dc = 100 * ((1 - Rt / P) - (1 - Rb / P))
            deta = Rt / y.mean() - Rb / x.mean()
            L.append(f"& {nice} & {dc:+.1f} & {100*deta:+.1f}"
                     f" & {100*(y.mean()-x.mean())/x.mean():+.0f}\\%"
                     f" & {t.statistic:+.1f}" + ("$^\\ast$" if t.pvalue < .05 else "")
                     + r" \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{What the two tools do, paired by instance, on three of the four"
          r" quantities of Table~\ref{tab:results}. $\Delta c$ is the change in design"
          r" efficiency in percentage points, $\Delta\eta$ the change in decision"
          r" efficiency also in percentage points, and $\Delta\bar{\mathcal{R}}$ the"
          r" relative change in the regret of the"
          r" final recommendation. A positive $\Delta c$ with a positive"
          r" $\Delta\bar{\mathcal{R}}$ is a better design with a worse answer, which"
          r" is what most cells show."
          r" $^\ast$ marks $p<0.05$ on a paired $t$-test over instances: six cells, four"
          r" of them a worse answer and two a better one, both on \textsc{Optimise}.}",
          r"\label{tab:tools}", r"\end{table}"]
    (SEC / "_tools_table.tex").write_text("\n".join(L) + "\n")


def tab_diag(E):
    """The diagnostics table: the two-treatment habit and rank deficiency, one column each for bare and tooled."""
    rows = {}
    for slug in list(SHORT) + [s + "+tools" for s in SHORT]:
        for cfg in CFGS:
            tw = rd = tot = 0
            t = TASKS[cfg]; d = len(t.factors)
            ks = []
            for f in sorted(glob.glob(str(R / f"transcript_{slug}_{cfg}_s*.json"))):
                seed = int(re.search(r"_s(\d+)\.json$", f).group(1))
                env = SlowLabEnv(t, seed=seed)
                for p, _ in designs_from(f, env):
                    u = np.unique(np.round(np.atleast_2d(p), 6), axis=0)
                    tot += 1
                    ks.append(len(u))
                    # The criterion for "too few" has to track the task's dimension:
                    # k distinct treatments give rank at most k, so k >= d+1 is
                    # required. The old version used "exactly 2", which was both
                    # arbitrary and misleading -- Sanity has one factor, where two
                    # treatments are exactly enough, yet it was counted as a failure.
                    tw += (len(u) < d + 1)
                    rd += (np.linalg.matrix_rank(np.c_[np.ones(len(u)), u]) < d + 1)
            if tot:
                rows[(slug, cfg)] = (100 * tw / tot, 100 * rd / tot, tot,
                                     float(np.median(ks)))
    # The two columns under each task are bare / with tools. The old version buried
    # that in a bracket inside a multicolumn row, with nothing above the columns
    # themselves, so a reader could not tell what they were. Add a sub-header row.
    L = [r"\begin{table}[t]", r"\centering\small",
         r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{@{}l" + "cc" * 4 + r"@{}}", r"\toprule",
         " & ".join([""] + [rf"\multicolumn{{2}}{{c}}{{\textsc{{{c}}}}}" for c in CFGS])
         + r" \\",
         " & ".join([""] + [rf"\multicolumn{{2}}{{c}}{{\footnotesize needs "
                            rf"$k\!\ge\!{len(TASKS[c].factors)+1}$}}" for c in CFGS])
         + r" \\[1pt]",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"\textbf{Model}" + r" & \textit{bare} & \textit{+tools}" * 4 + r" \\",
         r"\midrule",
         r"\multicolumn{9}{@{}l}{\textbf{(a)} \% of designs that cannot separate the"
         r" main effects (rank-deficient)} \\[2pt]"]
    for slug, nice in SHORT.items():
        cells = []
        for cfg in CFGS:
            a_ = rows.get((slug, cfg)); b_ = rows.get((slug + "+tools", cfg))
            cells += [f"{a_[1]:.0f}" if a_ else "---", f"{b_[1]:.0f}" if b_ else "---"]
        L.append(f"{nice} & " + " & ".join(cells) + r" \\")
    L += [r"\midrule",
          r"\multicolumn{9}{@{}l}{\textbf{(b)} \% of designs with fewer than $d+1$"
          r" distinct treatments --- why (a) happens} \\[2pt]"]
    for slug, nice in SHORT.items():
        cells = []
        for cfg in CFGS:
            a_ = rows.get((slug, cfg)); b_ = rows.get((slug + "+tools", cfg))
            cells += [f"{a_[0]:.0f}" if a_ else "---", f"{b_[0]:.0f}" if b_ else "---"]
        L.append(f"{nice} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{What the submitted designs look like. Every cell is a percentage of"
          r" that model's designs on that task, pooled over every round of all $20$"
          r" instances; \emph{bare} is the interface alone and \emph{+tools} the same"
          r" model with the two tools. \textbf{(a)} A design is rank-deficient when its"
          r" distinct treatments, stacked beside an intercept column, do not reach rank"
          r" $d+1$, so the $d$ main effects cannot be estimated separately."
          r" \textbf{(b)} The immediate cause, and almost the whole of it: $k$ distinct"
          r" treatments give rank at most $k$, so a design needs $k \ge d+1$ before"
          r" replication can buy anything. The threshold is the task's, not a fixed"
          r" number --- \textsc{Sanity} needs two and \textsc{Screen} seven --- and (b)"
          r" tracks (a) to within a few points on every task."
          r" The screening tool repairs much of (a) on the two higher-dimensional tasks"
          r" and none of it on \textsc{Sanity}, which has one factor and nothing to"
          r" screen.}",
          r"\label{tab:diag}", r"\end{table}"]
    (SEC / "_diag_table.tex").write_text("\n".join(L) + "\n")
    return rows


if __name__ == "__main__":
    E, C = load(), cv()
    tab_results(E, C); tab_tools(E, C); d = tab_diag(E)
    print("wrote _results_table.tex, _tools_table.tex, _diag_table.tex")
    for cfg in CFGS:
        bare = np.mean([d[(s, cfg)][1] for s in SHORT if (s, cfg) in d])
        tool = np.mean([d[(s + "+tools", cfg)][1] for s in SHORT
                        if (s + "+tools", cfg) in d])
        print(f"  {cfg:9s} rank-deficient: bare {bare:3.0f}%  tooled {tool:3.0f}%")
