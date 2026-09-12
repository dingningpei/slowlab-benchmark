"""The four tasks -- Sanity, Screen, Optimise, Transfer -- defined over the mechanistic
TOMGRO world.

Three differences from the earlier GP version:
  * factors are *real management decisions* in real units (C / ppm / mol /
    plants.m^-2 / days)
  * the control hierarchy comes from physical fact (lamps and heating serve a
    chamber, cultural operations a loop), not from a label we chose
  * effect sparsity arises from the mechanism and the economics together; it is
    no longer injected by hand
"""
from __future__ import annotations
from dataclasses import dataclass, field

from .world import MANAGEMENT_FACTORS


def _pick(*names):
    m = {f.name: f for f in MANAGEMENT_FACTORS}
    return [m[n] for n in names]


@dataclass
class Task:
    name: str
    factor_names: tuple[str, ...]
    n_chambers: int
    loops_per_chamber: int
    n_rounds: int              # P1: how many times the agent may adapt
    units_per_round: int       # P3: hard parallelism cap per round
    # ── The three noise layers. The values below are reasoned, not arbitrary,
    # but they are also *not literature values*: the literature reports a
    # coefficient of variation on yield, and we chose parameters that make the
    # model produce variation of the same magnitude.
    #
    # plant_cv=0.12 gives a per-plant margin-rate CV of 17.9% at the optimum.
    #   Published CVs for greenhouse tomato yield are 18.2% (first season) and
    #   11.8% (second), and trial CVs below 27% are considered good precision. We
    #   sit at the top of that range.
    # tau_chamber=0.10 makes the chamber effect 3.4% of margin at the optimum,
    #   with tau_loop and tau_batch 1.7% each. Horizontal temperature spreads of
    #   7-10 C occur in low-automation Mediterranean houses, with 14% GDD
    #   variation over a 25-point grid, so if 3.4% is wrong it is *too small*.
    #   Too small makes the tasks look easier, which works against our own claim
    #   that two of them cannot resolve design quality -- so the direction is
    #   conservative.
    plant_cv: float            # plant-to-plant parameter variation (layer 1: the source of observation noise)
    tau_chamber: float         # layer 2: chamber effect
    tau_loop: float
    tau_batch: float
    target_mde: float = 0.5    # minimum effect of interest, in units of the response sd
    noise_sd: float = 0.0      # filled in by env from the world's response scale
    transfer_energy_shock: float | None = None  # T4: the new energy price multiplier announced at the end

    # ── Irreversibility: the recovery period after heat damage ──────────
    # A unit whose crop was killed by heat must be cleared, disinfected and
    # replanted before it can be used again. Recovery is counted in *days* and
    # decoupled from rounds: occupancy is set by duration_days, so a
    # recovery_days shorter than one cycle only blocks the start of the next
    # round, and one longer than a cycle blocks two.
    # Severity is graded: the further above TCRIT, the longer the recovery, up to
    # recovery_days.
    # Energy price multiplier during training. None means the site's own prices
    # (a normal year).
    # This used to be train_cost_weight (lambda), which scaled *all* cost and so
    # amounted to swapping the objective function; at lambda=0.25 four of six
    # factors went rail-bound. Removed; see the economics module docstring.
    train_energy_shock: float | None = None

    recovery_days: float = 0.0        # 0 disables it (the default, which keeps existing results unchanged)
    damage_margin: float = 2.0        # degrees above TCRIT before damage counts
    damage_scale: float = 6.0         # further degrees to reach the full recovery period

    @property
    def factors(self):
        return _pick(*self.factor_names)

    @property
    def d(self) -> int:
        return len(self.factor_names)

    @property
    def budget_units(self) -> int:
        return self.n_rounds * self.units_per_round

    # Cycle length. **Single source of truth**: the forward model, the budget
    # accounting and the cost rates all read this one value. It used to be a
    # module constant in world and a separate literal in tasks, and the two
    # drifting apart is exactly what caused the old accounting bug (the budget was
    # charged for 160 days while the model ran a different number).
    cycle_days: float = 210.0

    @property
    def duration_days(self) -> int:
        """Days a round occupies the facility -- the cycle length."""
        return int(self.cycle_days)


TASKS = {
    # T1, close the loop at all. One factor, low plant-to-plant variation, ample
    # budget -- everything should pass.
    # Sanity's variance components are *identical* to the other three
    # configurations. It is easy only because one factor gets 16 unit-slots, not
    # because we turned the noise down: the four conditions hold at the same
    # strength in every configuration, otherwise the claim that we do not relax
    # them would be false.
    "T1": Task("T1", ("day_temp",),
               n_chambers=4, loops_per_chamber=4,
               n_rounds=2, units_per_round=8,
               plant_cv=0.12, tau_chamber=0.10, tau_loop=0.05, tau_batch=0.05,
               target_mde=0.5),

    # T2, eight-factor screening on a budget of 24 units.
    # Effect sparsity holds naturally: the first three factors carry about 80% of
    # the variation.
    # Note the facility size: five of the eight factors are chamber-level (two
    # temperatures, CO2, supplemental light, cycle length), and four chambers
    # cannot hold a screening design at all -- the number of chamber-level
    # combinations is the number of testable treatments. A real screening trial
    # does need a larger facility.
    "T2": Task("T2", tuple(f.name for f in MANAGEMENT_FACTORS),
               n_chambers=8, loops_per_chamber=3,
               n_rounds=2, units_per_round=12,
               plant_cv=0.12, tau_chamber=0.10, tau_loop=0.05, tau_batch=0.05,
               target_mde=0.6),

    # T3, sequential optimisation over three rounds. The two factors are
    # deliberately one chamber-level and one loop-level: if both were
    # chamber-level, a treatment would *be* a chamber setting, the
    # chamber-confounding check would be vacuous, and the control hierarchy would
    # constrain nothing.
    "T3": Task("T3", ("day_temp", "density"),
               n_chambers=4, loops_per_chamber=3,
               n_rounds=3, units_per_round=12,
               plant_cv=0.12, tau_chamber=0.10, tau_loop=0.05, tau_batch=0.05,
               target_mde=0.5),
}

# T4, transfer of economic conditions. After the campaign, *energy gets more
# expensive*, no new budget is granted, and the only question is: what do you
# recommend now?
#
# Source for the magnitude: the 2021-2022 European energy crisis. Eurostat
# non-household gas rose from 0.029-0.042 EUR/kWh in the late 2010s to 0.0867
# EUR/kWh in 2022H2, the highest on record; non-household electricity excluding
# taxes peaked at 0.1987 EUR/kWh in 2022H2. Both are 2-3x pre-crisis. We take
# 2.2, inside that range and on the conservative side.
#
# Training uses the site's own prices (train_energy_shock=None) rather than
# artificially cheap energy -- the old lambda=0.25 put four of six factors on
# their rails, leaving the agent a world with almost no interior.
#
# Why it is decomposable: the shock acts on electricity and gas only, and each
# observation returns energy_cost_rate and other_cost_rate separately. An agent
# that modelled the two apart can re-solve without running anything; one that
# learned only profit, or only a merged cost_rate, cannot.
TASKS["T4"] = Task("T4", ("day_temp", "night_temp", "co2", "par"),
                   n_chambers=8, loops_per_chamber=3,
                   n_rounds=3, units_per_round=24,
                   plant_cv=0.12, tau_chamber=0.10, tau_loop=0.05, tau_batch=0.05,
                   target_mde=0.5,
                   train_energy_shock=None, transfer_energy_shock=2.2)


# ── Readable aliases. The paper uses only these names; T1-T4 survive as code
# keys for backwards compatibility. The names are the four standard stages of an
# industrial experimental campaign, so a reader need not learn a private code. ──
LABEL = {"T1": "Sanity", "T2": "Screen", "T3": "Optimise", "T4": "Transfer"}
_OLD = {"sanity-1f": "T1", "screen-8f": "T2", "core-2f": "T3", "shift-4f": "T4"}
for _k, _v in LABEL.items():
    TASKS[_v] = TASKS[_k]
    TASKS[_v.lower()] = TASKS[_k]
for _old, _k in _OLD.items():          # backwards compatibility for old scripts and result files
    TASKS[_old] = TASKS[_k]
