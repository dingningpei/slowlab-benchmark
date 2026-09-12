#!/usr/bin/env python3
"""Run one LLM agent on SlowLab.

    cp .env.example .env          # put your DEEPSEEK_API_KEY in it
    python3 scripts/run_llm.py --model deepseek-chat --tasks Optimise --seeds 3

Each finished episode prints a line and is saved immediately; rerunning skips
what is already done, so a Ctrl-C or a dropped connection does not mean starting
over.
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from slowlab import SlowLabEnv, TASKS
from slowlab.llm import LLMAgent, zero_shot_recommendation, scripted_completer


def make_completer(model: str, temperature: float = 0.7, rpm: float = 0.0):
    """fake uses the built-in stub model; anything else goes to an OpenAI-compatible endpoint (DeepSeek / OpenAI / Qwen ...)."""
    if model == "fake":
        return scripted_completer()
    from slowlab.providers import openai_compatible
    return openai_compatible(model, temperature=temperature, rpm=rpm)


def load_done(path: pathlib.Path, drop_incomplete: bool = False) -> dict:
    """Episodes already completed. With drop_incomplete=True, those that did not run
    their full complement of rounds are discarded.

    Exhausting retries during rate limiting voids a round, and such an episode
    measures our network conditions rather than the model; leaving it in the
    results contaminates the score. Dropping them means the next run refills them.
    """
    if not path.exists():
        return {}
    out = {}
    for r in json.loads(path.read_text()):
        if drop_incomplete and r.get("rounds_submitted", 0) < TASKS[r["task"]].n_rounds:
            continue
        out[f"{r['task']}|{r['seed']}"] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="fake")
    ap.add_argument("--tasks", nargs="+", default=["Optimise"])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results/llm")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--rpm", type=float, default=0.0,
                    help="maximum calls per minute (0 = unlimited). With multiple processes, set this to total quota / number of processes")
    ap.add_argument("--fresh", action="store_true", help="ignore existing results and rerun from scratch")
    ap.add_argument("--redo-incomplete", action="store_true",
                    help="rerun only episodes that did not complete their rounds (those voided during rate limiting)")
    ap.add_argument("--tools", action="store_true",
                    help="the tool ablation: put the screening design table and the GP posterior maximum into the prompt")
    ap.add_argument("--recovery-days", type=float, default=None,
                    help="the irreversibility ablation: length of the recovery period a "
                         "heat-damaged unit enters, in days. Every task ships at 0, which "
                         "is the setting of every result in the paper's main tables, so a "
                         "run with this flag is NOT comparable with them -- give it its "
                         "own --out directory")
    a = ap.parse_args()

    # Modified copies, so the frozen task definitions are left alone.
    tasks = dict(TASKS)
    if a.recovery_days is not None:
        import copy
        for tn in a.tasks:
            t = copy.deepcopy(TASKS[tn]); t.recovery_days = float(a.recovery_days)
            tasks[tn] = t
        print(f"recovery_days = {a.recovery_days} (the paper's tables are at 0; "
              f"these episodes are a separate condition)\n")

    complete = make_completer(a.model, a.temperature, a.rpm)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    # When running a subset of tasks, the task name goes into the filename so all
    # four can run in parallel without overwriting each other; make_tables merges
    # them back by model name.
    tag = a.model.replace("/", "_") + ("+tools" if a.tools else "")
    if len(a.tasks) < 4:
        tag += "@" + "-".join(a.tasks)
    ep_path = out / f"episodes_{tag}.json"
    done = {} if a.fresh else load_done(ep_path, drop_incomplete=a.redo_incomplete)
    if done:
        n_all = len(load_done(ep_path))
        extra = (f", of which {n_all - len(done)} did not complete their rounds and will be rerun"
                 if a.redo_incomplete and n_all > len(done) else "")
        print(f"{len(done)} episodes already present, skipping them{extra} (--fresh forces a full rerun)\n")

    rows = list(done.values())
    total = len(a.tasks) * a.seeds
    n = 0
    t_start = time.time()
    for tn in a.tasks:
        for s in range(a.seeds):
            n += 1
            key = f"{tn}|{s}"
            if key in done:
                continue
            t0 = time.time()
            # Recall with no experiment: on the same site, first ask what the model
            # recommends with no data. The gap to its post-campaign score is what
            # this model actually gained from experimenting.
            env0 = SlowLabEnv(tasks[tn], seed=s)
            r0 = env0.submit_recommendation(
                zero_shot_recommendation(env0, complete), a.model + "|zero-shot")

            env = SlowLabEnv(tasks[tn], seed=s)
            ag = LLMAgent(complete=complete, name=a.model, tools=a.tools)
            r = env.submit_recommendation(ag.run(env), a.model,
                                          x_transfer=getattr(ag, "x_transfer", None))
            dt = time.time() - t0
            rows.append({"task": tn, "model": a.model, "seed": s,
                         "regret": float(r.simple_regret),
                         "regret_zero_shot": float(r0.simple_regret),
                         "cumulative_regret": float(r.cumulative_regret),
                         "stopped_early_at": getattr(ag, "stopped_early_at", None),
                         "cash": float(r.campaign_cash),
                         "wasted_fraction": float(r.wasted_fraction),
                         "transfer_regret": float(r.transfer_regret),
                         "rounds_submitted": r.n_designs,
                         # Capacity: zero unless the irreversibility dial is on, but it
                         # has to be on the record either way -- the first run of the
                         # ablation could not be read from its own episode file without it.
                         "lost_unit_days": float(r.lost_unit_days),
                         "unit_days_used": float(r.unit_days_used),
                         "format_failures": ag.format_failures,
                         "infeasible": ag.infeasible_submissions,
                         "trace": [float(v) for v in r.regret_trace],
                         "seconds": round(dt, 1)})
            ep_path.write_text(json.dumps(rows, indent=1))          # saved after every episode
            (out / f"transcript_{tag}_{tn}_s{s}.json").write_text(
                json.dumps(ag.transcript, indent=1, ensure_ascii=False))
            eta = (time.time() - t_start) / max(n - len(done), 1) * (total - n)
            print(f"[{n}/{total}] {tn} seed={s}  regret {r.simple_regret:.4f} "
                  f"(zero-shot {r0.simple_regret:.4f})  "
                  f"cash {r.campaign_cash:+8.1f}  rounds {r.n_designs}/{tasks[tn].n_rounds}  "
                  f"format-fail {ag.format_failures}  infeasible {ag.infeasible_submissions}  "
                  f"{dt:.0f}s   ~{eta/60:.0f} min left", flush=True)

    summary = {}
    for tn in a.tasks:
        sub = [r for r in rows if r["task"] == tn]
        if not sub:
            continue
        g = lambda k: np.array([r[k] for r in sub], float)
        summary[tn] = {
            "n": len(sub),
            "regret": float(g("regret").mean()),
            "regret_se": float(g("regret").std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0,
            "regret_zero_shot": float(g("regret_zero_shot").mean())
            if "regret_zero_shot" in sub[0] else None,
            "cumulative_regret": float(g("cumulative_regret").mean())
            if "cumulative_regret" in sub[0] else None,
            "cash": float(g("cash").mean()),
            "wasted_fraction": float(np.nanmean(g("wasted_fraction"))),
            "rounds_submitted": float(g("rounds_submitted").mean()),
            "format_failures": float(g("format_failures").mean()),
            "infeasible": float(g("infeasible").mean()),
        }
    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 78)
    print(f"{'task':<12}{'zero-shot':>11}{'R(R)':>17}{'R_cum':>10}"
          f"{'rounds':>10}{'format':>8}{'infeas':>8}")
    for tn, d in summary.items():
        zs = d.get("regret_zero_shot")
        rc = d.get("cumulative_regret")
        print(f"{tn:<12}{(f'{zs:.4f}' if zs is not None else '--'):>11}"
              f"{d['regret']:>10.4f}±{d['regret_se']:.4f}"
              f"{(f'{rc:.1f}' if rc is not None else '--'):>10}"
              f"{d['rounds_submitted']:>6.1f}/{TASKS[tn].n_rounds}"
              f"{d['format_failures']:>7.1f}{d['infeasible']:>8.1f}")
    gains = [(d["regret_zero_shot"] / d["regret"]) for d in summary.values()
             if d.get("regret_zero_shot") and d.get("regret")]
    if gains:
        print(f"\nimprovement factor from experimenting (zero-shot / R(R)): "
              f"{', '.join(f'{g:.1f}x' for g in gains)}")
    print(f"\nresults written to {out}/summary_{tag}.json (full transcript per episode included)")


if __name__ == "__main__":
    main()
