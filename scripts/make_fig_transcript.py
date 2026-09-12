#!/usr/bin/env python3
"""[Withdrawn from the paper on 2026-09-11] Qualitative side-by-side figure of one
episode's transcript.

Why it was withdrawn: once the body text was compressed, the two things the
figure showed were each said in one clause of prose, and the one thing only it
carried -- the model copying the tool's point verbatim -- is now reported by
tool_regret.py's verbatim_copy_rate() as 384 of 693 turns (55%), which is far
stronger than a single episode. A third of a page to illustrate a sentence that
already has a statistic behind it is not worth it.

The script is kept: its instance-selection logic (of 400 paired instances, 238
have a strictly wider tooled design and 143 of those also a worse answer) and its
assert are still useful, and it can be rerun if the figure is ever wanted again.
The generated sections/_fig_transcript.tex was removed and the body no longer
inputs it.

    python scripts/make_fig_transcript.py
"""
from __future__ import annotations
import json, pathlib, re, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import slowlab
from slowlab import SlowLabEnv, TASKS
from eig_of_llm import designs_from
from tool_regret import tool_point

R = ROOT / "results" / f"llm_env{slowlab.ENV_VERSION}"
SEC = ROOT / "paper" / "sections"
CFG, SEED, SLUG, NICE = "Screen", 8, "openai_gpt-5.6-luna", "GPT-5.6-luna"


def tex(s):
    return (str(s).replace("\\", "\\textbackslash ").replace("_", r"\_")
            .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#"))


def arm(tag):
    """Read one arm into a dict: per-round treatment count, rank, trajectory, and the final recommendation."""
    t = TASKS[CFG]; d = len(t.factors)
    env = SlowLabEnv(t, seed=SEED)
    f = R / f"transcript_{SLUG}{tag}_{CFG}_s{SEED}.json"
    turns = json.loads(f.read_text())
    ds = designs_from(f, env)
    rounds = []
    for p, _ in ds:
        P = np.atleast_2d(p)
        u = np.unique(np.round(P, 6), axis=0)
        rounds.append({"units": len(P), "distinct": len(u),
                       "rank": int(np.linalg.matrix_rank(np.c_[np.ones(len(u)), u]))})
    allp = np.vstack([np.atleast_2d(p) for p, _ in ds])
    uu = np.unique(np.round(allp, 6), axis=0)
    ep = json.loads((R / f"episodes_{SLUG}{tag}.json").read_text())
    e = [x for x in ep if x["task"] == CFG and x["seed"] == SEED][0]
    # The final recommendation and each round's current_best
    cb = []
    for turn in turns:
        m = re.search(r'"current_best"\s*:\s*(\{[^}]*\})', turn["assistant"])
        cb.append(json.loads(m.group(1)) if m else None)
    fin = None
    for turn in reversed(turns):
        m = re.search(r'"recommendation"\s*:\s*(\{[^}]*\})', turn["assistant"])
        if m:
            fin = json.loads(m.group(1)); break
    tool = None
    for turn in reversed(turns):
        tp = tool_point(turn["user"], env)
        if tp is not None:
            tool = tp; break
    return {"rounds": rounds, "distinct": len(uu),
            "rank": int(np.linalg.matrix_rank(np.c_[np.ones(len(uu)), uu])),
            "d": d, "regret": e["regret"], "zero": e["regret_zero_shot"],
            "trace": e["trace"], "current_best": cb, "final": fin, "tool": tool,
            "env": env, "turns": turns}



def selection_pool():
    """How many comparable instances this one was chosen from -- the caption must say, otherwise it is cherry-picking."""
    cfgs = ("Sanity", "Screen", "Optimise", "Transfer")
    slugs = ("deepseek_deepseek-v4-flash", "openai_gpt-5.6-luna", "qwen_qwen3.8-27b",
             "xiaomi_mimo-v2.5", "z-ai_glm-5.3-flash")
    tot = wider = both = 0
    for cfg in cfgs:
        tk = TASKS[cfg]
        for sl in slugs:
            for sd in range(20):
                fa = R / f"transcript_{sl}_{cfg}_s{sd}.json"
                fb = R / f"transcript_{sl}+tools_{cfg}_s{sd}.json"
                if not (fa.exists() and fb.exists()):
                    continue
                ev = SlowLabEnv(tk, seed=sd)
                da, db = designs_from(fa, ev), designs_from(fb, ev)
                if not da or not db:
                    continue
                ua = len(np.unique(np.round(np.vstack(
                    [np.atleast_2d(q) for q, _ in da]), 6), axis=0))
                ub = len(np.unique(np.round(np.vstack(
                    [np.atleast_2d(q) for q, _ in db]), 6), axis=0))
                ea = [x for x in json.loads((R / f"episodes_{sl}.json").read_text())
                      if x["task"] == cfg and x["seed"] == sd][0]
                eb = [x for x in json.loads((R / f"episodes_{sl}+tools.json").read_text())
                      if x["task"] == cfg and x["seed"] == sd][0]
                tot += 1
                wider += ub > ua
                both += (ub > ua) and eb["regret"] > ea["regret"]
    return tot, wider, both


def main():
    t = TASKS[CFG]
    names = [f.name for f in t.factors]
    bare, tool = arm(""), arm("+tools")
    env = tool["env"]
    _, best = env.truth.oracle()

    def norm(dd):
        return np.array([(float(dd[n]) - f.low) / (f.high - f.low)
                         for n, f in zip(names, t.factors)])

    def reg(x):
        return float(best - env.truth(np.asarray(x, float).reshape(1, -1))[0])

    # The central assertion: after round two, the model's current_best is the tool's point
    cb2 = tool["current_best"][1]
    assert cb2 is not None, "the tooled arm has no current_best in round two -- the transcript changed"
    gap = float(np.abs(norm(cb2) - tool["tool"]).max())
    assert gap < 1e-6, f"the model's turn-2 recommendation no longer matches the tool's ({gap:.2e} apart); the figure's claim must change"
    r_tool = reg(tool["tool"])

    n_tot, n_wider, n_both = selection_pool()
    ceiling = json.loads((ROOT / "results" /
        f"achievable_cv_env{slowlab.ENV_VERSION}_M320.json").read_text())[CFG]["prior"]

    short = {"day_temp": "day", "night_temp": "night", "co2": "CO$_2$",
             "par": "PAR", "density": "dens.", "lai_max": "LAI"}
    hdr = " & ".join(short[n] for n in names)

    def row(dd):
        return " & ".join(f"{float(dd[n]):g}" for n in names)

    def rk(a):
        return (rf"rank {a['rank']}/{a['d']+1}" +
                ("" if a["rank"] == a["d"] + 1 else r"\,---\,deficient"))

    # ---- Layout: a task header on top, two columns below; the right one on a pale
    # amber ground to mark it as the tooled arm ----
    def treat_rows(T, limit=3):
        """Rows of the treatment table. Beyond limit they are elided, stating how many were dropped as the transcripts do."""
        out, items = [], list(T.items())
        for k, v in items[:limit]:
            out.append(rf"{tex(k)} & " + row(v) + r"\\")
        if len(items) > limit:
            out.append(rf"\multicolumn{{{len(names)+1}}}{{@{{}}l}}{{\itshape "
                       rf"\dots\ {len(items)-limit} further treatments\dots}}\\")
        return out

    def treatments_of(turns, i):
        m = re.search(r'"treatments"\s*:\s*(\{.*?\})\s*,\s*"allocation"',
                      turns[i]["assistant"], re.S)
        return json.loads(m.group(1)) if m else None

    def panel(a, title, tint, is_tool):
        P = [rf"\begin{{minipage}}[t]{{0.485\linewidth}}",
             rf"\colorbox{{{tint}}}{{\parbox{{\dimexpr\linewidth-2\fboxsep}}{{\strut"
             rf"\textbf{{{title}}}\hfill $\bar{{\mathcal{{R}}}} = {a['regret']:.4f}$}}}}",
             r"\\[4pt]"]
        for i, r_ in enumerate(a["rounds"]):
            cb = a["current_best"][i]
            note = (r" \emph{--- the supplied screening design, verbatim}"
                    if is_tool and r_["distinct"] > 2 else "")
            P.append(rf"\textsc{{round {i+1}}}\quad {r_['units']} units, "
                     rf"\textbf{{{r_['distinct']}}} distinct treatments{note}\\[1pt]")
            T = treatments_of(a["turns"], i)
            if T:
                P += [r"\begin{tabular}{@{}l" + "r" * len(names) + r"@{}}",
                      r"& " + hdr + r"\\\midrule", *treat_rows(T),
                      r"\end{tabular}\\[3pt]"]
            if is_tool and i == 1 and cb:
                P += [r"\begin{tabular}{@{}l" + "r" * len(names) + r"@{}}",
                      r"\midrule",
                      r"\emph{tool proposes} & " +
                      " & ".join(f"{v:g}" for v in
                                 [f.low + x * (f.high - f.low)
                                  for x, f in zip(a["tool"], t.factors)]) + r"\\",
                      r"\emph{agent answers} & " + row(cb) + r"\\\midrule",
                      r"\end{tabular}\\[1pt]",
                      r"\centerline{\small$\underbrace{\hphantom{\text{identical to six"
                      r" digits}}}_{\textstyle\text{\small identical to six digits}}$}"
                      r"\\[3pt]"]
            if cb:
                P.append(rf"\emph{{interim recommendation}} $\to$ "
                         rf"$\mathcal{{R}} = {reg(norm(cb)):.4f}$\\[5pt]")
        P += [r"\vfill",
              rf"\emph{{whole design}}: {a['distinct']} distinct, {rk(a)}\\[2pt]",
              rf"\colorbox{{{tint}}}{{\parbox{{\dimexpr\linewidth-2\fboxsep}}{{\strut\ "
              rf"final recommendation\hfill $\mathcal{{R}} = {a['regret']:.4f}$}}}}",
              r"\end{minipage}"]
        return P

    L = [r"%% Figure: generated by scripts/make_fig_transcript.py. Do not edit by hand.",
         r"\begin{figure}[t]", r"\centering\footnotesize",
         r"\setlength{\tabcolsep}{3pt}",
         r"\colorbox{black!12}{\parbox{\dimexpr\linewidth-2\fboxsep}{\strut"
         rf"\textbf{{\textsc{{{CFG}}}}}, instance {SEED}, {NICE} \quad"
         rf" {len(names)} factors, {t.n_rounds} rounds $\times$ {t.units_per_round} units,"
         rf" ceiling $\mathcal{{R}}^\star(\varnothing) = {ceiling:.4f}$, zero-shot"
         rf" ${bare['zero']:.4f}$ \hfill \emph{{tools: a screening-design generator and a"
         rf" posterior-maximisation routine}}}}}}",
         r"\\[6pt]",
         *panel(bare, "No tools", "black!7", False),
         r"\hfill",
         *panel(tool, "With tools", "yellow!14", True),
         rf"\caption{{One instance of \textsc{{{CFG}}}, the same model and the same site,"
         rf" run with and without the two tools; everything else identical."
         rf" \emph{{Left}}: four distinct treatments over two rounds, {rk(bare)} over the six"
         rf" factors --- a design that cannot separate the main effects it was commissioned"
         rf" to separate. \emph{{Right}}: the agent adopts the supplied screening design,"
         rf" reaching {tool['rounds'][0]['distinct']} distinct treatments and full rank, so"
         rf" the six main effects become estimable and the design scores better on $c$."
         rf" It then answers"
         rf" ${tool['regret']/bare['regret']:.1f}\times$ worse. The boxed rows are why: after"
         rf" round~2 the agent's stated recommendation is the inference tool's proposed point"
         rf" to six digits, and carries that tool's regret of ${r_tool:.4f}$. In the final turn"
         rf" it abandons the tool for its best observed treatment, recovering to"
         rf" ${tool['regret']:.4f}$ --- still worse than it managed with no tool at all."
         rf" A better design and a worse answer are not a contradiction here: full rank"
         rf" makes the main effects estimable, but {tool['distinct']} points over six factors"
         rf" is close to one level a factor, and the same routine is handed twice as many"
         rf" when it chooses its own design (\S\ref{{sec:results:findings}}). This"
         rf" episode is not exceptional: of the {n_tot} instances run both ways, the tooled"
         rf" arm submits the strictly wider design in {n_wider} and, in {n_both} of those,"
         rf" also the worse answer. We show this one because the copying is exact."
         rf" Table~\ref{{tab:tools}} aggregates the effect over all $20$ model--task pairs.}}",
         r"\label{fig:transcript}", r"\end{figure}"]

    # Written to results/ rather than sections/: the figure is withdrawn, and a
    # generated file left in sections/ will eventually be input by mistake
    out = ROOT / "results" / "fig_transcript.tex"
    out.write_text("\n".join(L) + "\n")
    print(f"bare : {bare['distinct']} distinct, rank {bare['rank']}/{bare['d']+1}, "
          f"R = {bare['regret']:.4f}")
    print(f"tools: {tool['distinct']} distinct, rank {tool['rank']}/{tool['d']+1}, "
          f"R = {tool['regret']:.4f}   tool's own {r_tool:.4f}")
    print(f"turn 2 copies the tool; largest coordinate difference {gap:.2e}")
    print("wrote", out.name)


if __name__ == "__main__":
    main()
