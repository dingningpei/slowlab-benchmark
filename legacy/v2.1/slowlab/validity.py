"""The design validity checker -- W1's only process measure.

Important: this is not env.validate_design().
  * env.validate_design()  -> mechanical feasibility only; free and callable by
                              the agent
  * score_validity()       -> the scientific validity criteria used for scoring;
                              not callable by the agent
Keeping them apart is what stops the "does it have statistical tools" ablation
from being contaminated (proposal section 3.4).
"""
from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict

# Cache of permutation null distributions, keyed on the design's *shape* rather
# than its contents, so the verdict is deterministic and reproducible
_NULL_CACHE: dict = {}


@dataclass
class ValidityReport:
    checks: dict[str, bool] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    advisory: dict[str, float] = field(default_factory=dict)

    # Five *necessary* conditions. They claim no sufficiency: a design can pass
    # all five and still waste the budget. They are reported item by item, never
    # combined into one score and never aggregated with regret.
    MANDATORY = ("replication", "effects_estimable", "randomization_declared",
                 "no_chamber_confounding", "allocation_balanced")

    @property
    def passed(self) -> bool:
        return all(self.checks.get(k, False) for k in self.MANDATORY)

    def failed_checks(self) -> list[str]:
        return [k for k in self.MANDATORY if not self.checks.get(k, False)]


def _assoc_stat(F: np.ndarray, N: np.ndarray) -> float:
    """Largest |correlation| between a factor's values and the facility position coordinates."""
    best = 0.0
    for j in range(F.shape[1]):
        fj = F[:, j]
        if np.std(fj) < 1e-12:
            continue
        for k in range(N.shape[1]):
            nk = N[:, k]
            if np.std(nk) < 1e-12:
                continue
            best = max(best, abs(float(np.corrcoef(fj, nk)[0, 1])))
    return best


def _null_threshold(shape_key, F, N, level, q=0.99, n_perm=400) -> float:
    """Permutation null -- it must be built separately for each control level.

    Free permutation is the wrong null for a chamber-level factor: such a factor
    is constant within a chamber by definition, so its correlation with chamber
    identity is *imposed by the facility*, not chosen by the agent. Comparing
    against a free permutation marks every split-plot design unbalanced, which is
    exactly the mistake we made first.

      level="loop"     a loop-level factor can go on any unit -> freely permute unit positions
      level="chamber"  a chamber-level factor is assigned whole chambers -> permute chamber labels only

    Note that n chambers admit only n! relabellings, so with four chambers the
    smallest attainable p-value is 1/24 ~= 4%: the chamber dimension has *no
    power by construction* in a small facility. That is a statistical fact rather
    than an implementation defect and should be stated in the paper -- the
    randomisation test effectively constrains the loop-level allocation.
    """
    if shape_key in _NULL_CACHE:
        return _NULL_CACHE[shape_key]
    rng = np.random.default_rng(abs(hash(shape_key)) % (2**32))
    stats = np.empty(n_perm)
    if level == "loop":
        idx = np.arange(len(N))
        for i in range(n_perm):
            rng.shuffle(idx)
            stats[i] = _assoc_stat(F, N[idx])
    else:
        ch = N[:, 0].astype(int)
        chambers = np.unique(ch)
        idx_by_ch = {c: np.where(ch == c)[0] for c in chambers}
        for i in range(n_perm):
            Np = N.copy()
            relabel = dict(zip(chambers, rng.permutation(chambers)))
            for c in chambers:
                Np[idx_by_ch[c], 0] = relabel[c]
            stats[i] = _assoc_stat(F, Np)
    thr = float(np.quantile(stats, q))
    _NULL_CACHE[shape_key] = thr
    return thr


def score_validity(design, facility, task) -> ValidityReport:
    r = ValidityReport()
    reps = design.replicates()

    # ── Requirement 1: pure error must be estimable ─────────
    # The criterion is deliberately not "every treatment needs >=2 replicates" --
    # that would mark a standard Plackett-Burman screening design invalid. The
    # real practice is to add replicated centre points. Pure error can only come
    # from replication:
    #   pure_error_df = Σ(n_i − 1)
    # residual degrees of freedom from an unsaturated model confound lack of fit
    # and cannot substitute for pure error.
    min_rep = min(reps.values()) if reps else 0
    pure_df = sum(max(0, n - 1) for n in reps.values())
    r.checks["replication"] = pure_df >= 2
    r.notes["replication"] = (f"pure-error df = {pure_df} (needs >=2); "
                              f"minimum replication {min_rep}, {len(reps)} treatments")
    r.advisory["pure_error_df"] = float(pure_df)

    # ── Requirement 1b: the effects under study must be estimable ────────
    # Pure error alone is gameable: one recipe replicated 12 times has 11
    # pure-error df and passes everything above -- while estimating no effect at
    # all. Replication is not a virtue in itself.
    # The criterion: let V be the set of factors taking >=2 levels in this design,
    # and require
    #   (a) |V| >= 1                      -- the design is not empty
    #   (b) rank([1, X_V]) = |V| + 1      -- the factors in V are not collinear
    # It deliberately does not require that *all* factors vary: holding one factor
    # and attacking another is legitimate in a sequential strategy.
    Xall, cols = [], [f.name for f in task.factors]
    for tid, uids in design.allocation.items():
        for _ in uids:
            Xall.append([design.treatments[tid][c] for c in cols])
    est_ok, est_note = False, "no observations"
    if Xall:
        X = np.asarray(Xall, float)
        varying = [j for j in range(X.shape[1]) if np.ptp(X[:, j]) > 1e-9]
        if not varying:
            est_note = "no factor takes two levels -- this design can estimate nothing"
        else:
            M = np.column_stack([np.ones(len(X))] + [X[:, j] for j in varying])
            rk = int(np.linalg.matrix_rank(M, tol=1e-9))
            est_ok = rk == len(varying) + 1
            est_note = (f"varying factors {[cols[j] for j in varying]}, "
                        f"design matrix rank {rk}/{len(varying)+1}"
                        + ("" if est_ok else " -- collinear, effects not separable"))
    r.checks["effects_estimable"] = est_ok
    r.notes["effects_estimable"] = est_note

    # ── Requirement 2: randomisation must be declared explicitly ────────
    r.checks["randomization_declared"] = design.randomization_seed is not None
    r.notes["randomization_declared"] = (
        "declared" if design.randomization_seed is not None else "no randomization_seed declared")

    # ── Requirement 3: loop-level contrasts must not confound with chambers ──
    # The criterion is not "a treatment's replicates must not share a chamber" --
    # that would mark a *randomised complete block* design confounded. In a block
    # design chambers are the blocks, loop-level factors vary *within* a chamber,
    # and the chamber effect cancels exactly in the contrast. That is textbook
    # practice, not a defect.
    # Real confounding is: the factor is constant inside every chamber and varies
    # only between chambers.
    loop_factors = [f.name for f in task.factors if f.control_level == "loop"]
    bad = []
    if loop_factors and facility.n_chambers > 1:
        by_ch = defaultdict(lambda: defaultdict(set))
        for tid, uids in design.allocation.items():
            for u in uids:
                if u not in facility._by_id:
                    continue
                c = facility.get(u).chamber
                for fn in loop_factors:
                    by_ch[fn][c].add(round(design.treatments[tid][fn], 9))
        for fn in loop_factors:
            levels = {v for s_ in by_ch[fn].values() for v in s_}
            if len(levels) < 2:
                continue                      # factor not under study, no contrast to confound
            varying = sum(1 for s_ in by_ch[fn].values() if len(s_) >= 2)
            need = min(2, facility.n_chambers)
            if varying < need:
                bad.append(f"{fn} (varies inside only {varying} chambers, needs {need})")
    r.checks["no_chamber_confounding"] = not bad
    r.notes["no_chamber_confounding"] = "none" if not bad else "confounded with chamber: " + "; ".join(bad)

    # ── Requirement 4: allocation must not correlate with facility structure ──
    # Checking only that a seed was declared is a formality: an agent can declare
    # a seed and still lay treatments into chambers in order of factor magnitude.
    # This is the substantive test -- whether the association between factor
    # values and physical unit position exceeds what random permutation produces.
    rows_f, rows_n = [], []
    for tid, uids in design.allocation.items():
        fv = [design.treatments[tid][f.name] for f in task.factors]
        for u in uids:
            if u not in facility._by_id:
                continue
            unit = facility.get(u)
            rows_f.append(fv)
            rows_n.append([unit.chamber, unit.loop])
    if len(rows_f) >= 4:
        Fall = np.asarray(rows_f, float)
        N = np.asarray(rows_n, float)
        shape0 = (len(rows_f), tuple(sorted(reps.values())),
                  facility.n_chambers, facility.loops_per_chamber)
        cols = {lv: [i for i, f in enumerate(task.factors) if f.control_level == lv]
                for lv in ("loop", "chamber")}
        ok, detail, worst = True, [], 0.0
        for lv, ix in cols.items():
            if not ix:
                continue
            F = Fall[:, ix]
            obs = _assoc_stat(F, N)
            thr = _null_threshold(shape0 + (len(ix), lv), F, N, lv)
            detail.append(f"{lv} {obs:.2f}/{thr:.2f}")
            worst = max(worst, obs)
            if obs > thr:
                ok = False
        r.checks["allocation_balanced"] = ok
        r.notes["allocation_balanced"] = "factor x position correlation/threshold: " + ", ".join(detail)
        r.advisory["assoc_stat"] = round(worst, 3)
        r.advisory["assoc_threshold"] = round(
            _null_threshold(shape0 + (len(cols["loop"]), "loop"),
                            Fall[:, cols["loop"]], N, "loop"), 3) if cols["loop"] else 1.0
    else:
        r.checks["allocation_balanced"] = True
        r.notes["allocation_balanced"] = "fewer than 4 units, test not run"

    # ── Requirement 6: every treatment must pre-register an interval forecast ──
    # This is about *completeness of the specification*: it does not judge whether
    # the forecast is good (that is the interval score's job), only that the agent
    # said everything it was asked to, since otherwise it could dodge the process
    # measure by staying silent.
    missing = [t for t in design.treatments if t not in design.predictions]
    r.checks["predictions_declared"] = not missing
    r.notes["predictions_declared"] = (
        "all treatments pre-registered an interval" if not missing else f"treatments missing a forecast: {sorted(missing)}")

    # ── Advisory: power. It needs statistical assumptions, so it is reported
    # rather than enforced. How much replication would detect target_mde, in units
    # of the response sd.
    if task.target_mde and task.noise_sd > 0:
        need = 2 * ((1.96 + 0.84) * task.noise_sd / task.target_mde) ** 2
        r.advisory["replicates_needed_per_arm"] = round(need, 1)
        r.advisory["min_replicates_submitted"] = float(min_rep)
        r.checks["adequate_power"] = min_rep >= need
        r.notes["adequate_power"] = f"needs ~{need:.1f} per group, actual {min_rep}"

    return r
