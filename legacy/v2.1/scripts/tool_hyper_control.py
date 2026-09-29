#!/usr/bin/env python3
"""A control: is the inference tool weak because its GP hyperparameters are hard-coded?

A reviewer will certainly ask. This control had been run once by hand: the appendix
said "-52% to +11%", the change log recorded four absolute values, the two did not
agree, and neither could be reproduced. Hence this script.

Method: take the observations the LLM actually submitted (designs from the
transcript, replayed through the environment) and score them twice under the same
recommendation rule (the posterior-mean maximiser) -- once with the hard-coded
(ls, noise), once refitting both by marginal likelihood on the same observations.
Only the hyperparameters differ.

    python scripts/tool_hyper_control.py
"""
from __future__ import annotations
import json, glob, pathlib, re, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import slowlab
from slowlab import SlowLabEnv, TASKS
from slowlab.agents import GP, _cands
from eig_of_llm import designs_from

R = ROOT / "results" / f"llm_env{slowlab.ENV_VERSION}"
CFGS = ("Sanity", "Screen", "Optimise", "Transfer")
SLUGS = ("deepseek_deepseek-v4-flash", "openai_gpt-5.6-luna", "qwen_qwen3.8-27b",
         "xiaomi_mimo-v2.5", "z-ai_glm-5.3-flash")
LS_GRID = (0.10, 0.15, 0.22, 0.28, 0.40, 0.60, 0.90)
NOISE_GRID = (0.05, 0.10, 0.15, 0.22, 0.30, 0.45)


def log_ml(X, y, ls, noise):
    """Marginal likelihood after standardisation, matching GP.fit's scale convention."""
    ym = y.mean(); ys = float(np.std(y)) if len(y) > 1 and np.std(y) > 1e-12 else 1.0
    z = (y - ym) / ys
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    K = np.exp(-0.5 * d2 / ls ** 2) + (noise ** 2 + 1e-8) * np.eye(len(X))
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return -np.inf
    a = np.linalg.solve(L.T, np.linalg.solve(L, z))
    return float(-0.5 * z @ a - np.log(np.diag(L)).sum() - 0.5 * len(z) * np.log(2 * np.pi))


def fit_hyper(X, y):
    best, arg = -np.inf, (0.28, 0.30)
    for ls in LS_GRID:
        for nz in NOISE_GRID:
            v = log_ml(X, y, ls, nz)
            if v > best:
                best, arg = v, (ls, nz)
    return arg


def argmax_of(X, y, ls, noise, d, seed=7):
    g = GP(ls=ls, noise=noise).fit(X, y)
    C = _cands(d, 8000, np.random.default_rng(seed))
    return C[int(np.argmax(g.predict(C)[0]))]


def main():
    out = {}
    print(f"{'task':10s}{'fixed':>10}{'refit':>10}{'change':>9}{'n':>5}   median (ls, noise)")
    for cfg in CFGS:
        t = TASKS[cfg]
        fx, rf, hyp = [], [], []
        for slug in SLUGS:
            for f in sorted(glob.glob(str(R / f"transcript_{slug}+tools_{cfg}_s*.json"))):
                seed = int(re.search(r"_s(\d+)\.json$", f).group(1))
                env = SlowLabEnv(t, seed=seed)
                # Replay every design submitted in this episode. Noise is drawn once
                # from a fixed seed, so both arms see the *same* y -- this control
                # allows only the hyperparameters to differ.
                ds = designs_from(f, env)
                if not ds:
                    continue
                X = np.vstack([np.atleast_2d(p) for p, _ in ds])
                rng = np.random.default_rng(10_000 + seed)
                y = (np.asarray(env.truth(X), float)
                     + rng.normal(0, env.truth.response_sd, len(X)))
                _, best = env.truth.oracle()
                nz0 = max(0.15, t.plant_cv)
                x0 = argmax_of(X, y, 0.28, nz0, t.d)
                ls1, nz1 = fit_hyper(X, y)
                x1 = argmax_of(X, y, ls1, nz1, t.d)
                fx.append(float(best - env.truth(x0.reshape(1, -1))[0]))
                rf.append(float(best - env.truth(x1.reshape(1, -1))[0]))
                hyp.append((ls1, nz1))
        a, b = float(np.mean(fx)), float(np.mean(rf))
        out[cfg] = {"fixed": a, "refit": b, "change": (b - a) / a, "n": len(fx),
                    "median_ls": float(np.median([h[0] for h in hyp])),
                    "median_noise": float(np.median([h[1] for h in hyp]))}
        print(f"{cfg:10s}{a:10.4f}{b:10.4f}{100*(b-a)/a:+8.0f}%{len(fx):5d}   "
              f"({out[cfg]['median_ls']:.2f}, {out[cfg]['median_noise']:.2f})")
    ch = [v["change"] for v in out.values()]
    out["summary"] = {"min_change": min(ch), "max_change": max(ch),
                      "n_better": sum(1 for c in ch if c < 0),
                      "n_worse": sum(1 for c in ch if c > 0)}
    p = ROOT / "results" / f"tool_hyper_control_env{slowlab.ENV_VERSION}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nRefitting changes the tool's regret by {100*min(ch):+.0f}% to {100*max(ch):+.0f}%: "
          f"{out['summary']['n_better']} tasks better, {out['summary']['n_worse']} worse")
    print("wrote", p.name)


if __name__ == "__main__":
    main()
