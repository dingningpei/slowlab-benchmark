"""Frozen statistics for the formal analysis (RESEARCH_PLAN section 5).

The unit is the site: repeats are averaged within a site first. Primary tests are site-level
paired mean differences with a two-sided sign-flip permutation test and Holm correction;
intervals are site-level BCa bootstrap 95% intervals; the secondary family is corrected by
Benjamini-Hochberg.
"""
from __future__ import annotations

import math

import numpy as np

SIGN_FLIPS = 100_000
BOOTSTRAPS = 10_000


def sign_flip_test(d, flips: int = SIGN_FLIPS, seed: int = 0) -> float:
    """Two-sided p-value of mean(d) = 0 under random sign flips (Monte Carlo, observed included)."""
    d = np.asarray(d, dtype=float)
    if len(d) == 0:
        return float('nan')
    observed = abs(d.mean())
    rng = np.random.default_rng([seed, 0x5F1])
    hits = 0
    for start in range(0, flips, 10_000):
        n = min(10_000, flips - start)
        signs = rng.choice((-1.0, 1.0), size=(n, len(d)))
        hits += int(np.sum(np.abs(signs @ d / len(d)) >= observed - 1e-12))
    return (1 + hits) / (1 + flips)


def bca_interval(d, level: float = 0.95, boots: int = BOOTSTRAPS, seed: int = 0) -> tuple[float, float]:
    """BCa bootstrap interval for the mean of d (resampling sites)."""
    from scipy.stats import norm
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n < 3:
        return float('nan'), float('nan')
    rng = np.random.default_rng([seed, 0xBCA])
    means = d[rng.integers(0, n, size=(boots, n))].mean(axis=1)
    theta = d.mean()
    share = np.mean(means < theta)
    if share in (0.0, 1.0) or np.allclose(d, d[0]):
        return float(theta), float(theta)
    z0 = norm.ppf(share)
    jack = np.array([np.delete(d, i).mean() for i in range(n)])
    diff = jack.mean() - jack
    accel = float(np.sum(diff ** 3) / (6.0 * np.sum(diff ** 2) ** 1.5)) if np.any(diff) else 0.0
    out = []
    for q in ((1 - level) / 2, 1 - (1 - level) / 2):
        z = norm.ppf(q)
        adj = norm.cdf(z0 + (z0 + z) / (1 - accel * (z0 + z)))
        out.append(float(np.quantile(means, adj)))
    return out[0], out[1]


def holm(pvalues: dict) -> dict:
    """Holm-adjusted p-values (step-down, monotone)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def benjamini_hochberg(pvalues: dict) -> dict:
    """BH-adjusted p-values (step-up, monotone)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 1.0
    for i in range(m - 1, -1, -1):
        k, p = items[i]
        running = min(running, p * m / (i + 1))
        out[k] = min(running, 1.0)
    return out


def minimum_detectable(sd: float, n: int, alpha: float, power: float = 0.8) -> float:
    """Smallest true mean difference a paired t-test detects with the given power (approximation)."""
    from scipy import stats
    if n < 2 or not math.isfinite(sd) or sd <= 0:
        return float('nan')

    def pw(delta):
        df, tc = n - 1, stats.t.ppf(1 - alpha / 2, n - 1)
        nc = delta / (sd / math.sqrt(n))
        return 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
    lo, hi = 0.0, 10 * sd
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pw(mid) < power else (lo, mid)
    return hi


def summarise(d, *, seed: int, alpha_for_mde: float) -> dict:
    d = np.asarray(d, dtype=float)
    lo, hi = bca_interval(d, seed=seed)
    sd = float(d.std(ddof=1)) if len(d) > 1 else float('nan')
    return {'sites': int(len(d)), 'mean': float(d.mean()) if len(d) else float('nan'), 'sd': sd,
            'ci95_bca': [lo, hi], 'p_sign_flip': sign_flip_test(d, seed=seed),
            'minimum_detectable_80': minimum_detectable(sd, len(d), alpha_for_mde)}
