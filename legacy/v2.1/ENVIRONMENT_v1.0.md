# SlowLab environment v1.0 — freeze declaration

**Frozen: 2026-09-10**

From this point the environment is a **fixed artefact**. The defect register below is
public, complete, and not intended to be worked through. A benchmark's value is
comparability across agents, not correctness of the simulator.

---

## Why freeze

The working pattern before this was: find a defect → fix it → the ceiling moves → the
tasks have to be redefined → find another defect. That does not converge. The cause was
not the data but a commitment:

> for design efficiency `c(D) = [R*(∅) − R*(D)] / R*(∅)` to mean anything,
> `R*(∅)` has to be resolvable.

But `R*(∅)` is a **property of the simulator's ground truth**, so it inherits every one
of the simulator's defects. As long as an absolutely calibrated ceiling is required, the
loop has no exit — the simulator can always be made a little more correct.

Not one of the paper's actual findings needs `R*(∅)` to be right as an absolute number:

| Finding | What it rests on |
|---|---|
| 99% / 92% / 54% of designs rank-deficient on Screen / Transfer / Optimise | the submissions; independent of the simulator |
| the median design uses two distinct treatments, whatever the task needs | the submissions; independent of the simulator |
| in 55% of turns the stated recommendation is the tool's point to six digits | the submissions; independent of the simulator |
| the first round of data makes the recommendation worse in 44% of episodes | the simulator, but only through **ordering** |
| the tools reorder the leaderboard | the simulator, but only through **ordering** |
| design quality does not predict the tool's effect | two separable measurements, **ordering** only |

Appendix F had already measured it: going from M = 80 to M = 320 atoms preserves only
0.81–0.92 of the **level** of `R*`, while the rank correlation is 0.96–0.97. The level
does not converge; the order is stable.

**We had been paying for a property the paper does not use.**

---

## Three rules that take effect with the freeze

### 1. Stopping rule

> **A defect is fixed if it changes a claim the paper makes, and registered if it does not.**

The four fixes below are all of the first kind. The open question about how block effects
are anchored only affected the admission threshold, and the threshold is gone, so it is
**registered, not fixed**.

### 2. The admission threshold is withdrawn

`R*(∅) / (σ/√B) ≥ 2` is no longer an entry condition for a task. It becomes a **declared
property of each task**: the ratio is reported and the reader discounts the task
accordingly. The task set is no longer a function of the simulator's fine detail.

### 3. Design measurements are reported as ranks

`R*(D)` is still computed, but the **headline is the order**: where an agent's design
places on a task, and its paired win rate against the reference policies. `c(D)` and `η`
are reported with the levels marked **uncalibrated, for ordering only**.

---

## Fixes included in v1.0

These four make the paper wrong if left in, independently of how the tasks are defined:

| Fix | The problem | Effect |
|---|---|---|
| **λ withdrawn** (`cost_weight` pinned to 1) | costs were multiplied by 0.5 / 0.25, which directly sets where the optimum is; four of the six factors were rail-bound because of it | the density optimum moves from 6.36 back to 3.39, inside the commercial range 2.5–4 |
| **`_Pg` reads the site parameters** | the photosynthesis function ignored the parameter dictionary passed to it, so every site had an identical light response, and the appendix claim that sampling light and CO₂ parameters doubles the ceiling was a no-op | `R*(∅)` +32% |
| **solar gain added to the heating model** | `HL = A·U·ΔT` is a *sizing* formula (boiler capacity on the coldest night, explicitly assuming no solar gain) and was being used as 210 days of running energy; heating is 48% of cost, and transmitted solar is 6.2 times daytime heat loss | gross margin −1 → +11 €/(m²·yr); `R*`/revenue ×4.4 |
| **definition-level corrections to the economic model** | `net_profit` was not net profit; harvest labour had been removed by us and never added back; revenue carried no grading loss | renamed `gross_margin`, with the omitted items listed; 0.19 €/kg harvest labour and 5% grading loss added back |

Two pieces of physics completed at the same time, from the same cause:

- **Ventilation-driven CO₂ loss.** Venting excess solar heat means opening windows, and
  open windows carry away enriched CO₂. The constant `co2_leak` was physically wrong.
- **Flue-gas CO₂.** Greenhouse CO₂ is mostly a by-product of heating (about 0.20 kg
  CO₂/kWh) and does not have to be bought. Counting only the ventilation loss degenerates
  the CO₂ factor entirely (optimum 393 ppm, zero enrichment).

Together they produce the real pattern: a cold site heats more → more free CO₂, and a
larger ΔT → less ventilation → cheap enrichment; a warm site the reverse. Measured,
`corr(T_out, co2*) = −0.98`.

Plus three earlier methodological fixes, recorded in Appendix I: multi-start oracle
search, under-converged `best_fixed`, and the EIG reference and the model using two
different yardsticks.

### The state of v1.0 in numbers

```
Full six-factor space:  gross margin +11 €/(m²·yr)   R*/revenue 2.63%
  day_temp    23.0 ± 0.7    corr(TCRIT, day_temp*) = +1.00
  night_temp  17.2 ± 0.8
  co2        488.4 ± 57.2   corr(T_out, co2*)      = −0.98
  par         18.0 ± 0.0    rail-bound 100% (electricity prices; real)
  density      2.9 ± 0.6    rail-bound 21%
  lai_max      4.2 ± 0.3

R*(∅) against noise, per task (a declared property, not an entry condition):
  task       d   B     R*(∅)    σ_tot/√B   ratio
  Sanity     1  16   0.00575   0.00254    2.26
  Screen     6  24   0.00424   0.00305    1.39
  Optimise   2  36   0.00680   0.00175    3.88
  Transfer   4  72   0.00366   0.00178    2.06
```

---

## Register of known defects (**not fixed, published**)

### A. Noise model

**A1. Block effects are anchored to a quantity with no physical meaning.**
`τ_chamber/loop/batch` are all multiplied by `response_sd`, which is defined as the
standard deviation over points drawn uniformly from the **whole design space** — and is
therefore dominated by catastrophic corners no one would ever visit. Measured: after the
ventilation physics was corrected the optimum barely moved (gross margin 0.0672 → 0.0669),
but `response_sd` went from 0.0669 to 0.1142 (**+71%**), so chamber noise rose
mechanically from 4.4% of revenue to 7.9%. **How uniform the climate is between chambers
has nothing to do with how bad the worst corner of the design space is.** The right
treatment is to make block effects a **climate offset** (this chamber runs some degrees
warmer than set point; measured greenhouse-scale temperature uniformity is ±0.5–1.5 °C)
pushed through the forward model. Not implemented.

**A2. `task.noise_sd` is a declared value, not one the simulation produces.**
`plant_cv × response_sd` is 1.3–1.9 times the measured unit-level sd. The environment
itself is right (a unit is one hydroponic loop of 36 plants and the observation is their
mean), but the calibration table uses the declared value.

### B. Crop and economic model

**B1. Profit-optimal yield is 30–31 kg/(m²·yr); commercial practice is 50–70.**
The model's **maximum attainable** yield is 63–65 kg/(m²·yr), inside the commercial range,
so this is not a capacity limit: the cost structure penalises the high-yield direction too
heavily. Cause not established.

**B2. `par` is rail-bound at 100% of sites**, so the supplemental-light sub-model
contributes nothing to the objective (the profit elasticity of both `elec_price` and
`par_ambient` is 0.0%). `par_ambient` is fixed at 18 and does not vary with climate — yet
the difference between a Dutch winter and an Almería winter in natural DLI is exactly what
decides whether supplemental light pays.

**B3. A daily-step model cannot express a time-varying CO₂ enrichment strategy.**
Real growers enrich when the vents are shut and stop when they open; we can only set a
seasonal mean.

**B4. `gross_margin` is not net profit.** Not included: depreciation and facility capital,
land or rent, water, fertiliser and substrate, crop protection, packaging, transport and
sales commission, and management overhead. These are approximately constant in the
management factors, so they do not move the optimum and they cancel from both `R*` and the
noise — but the name has to be honest.

**B5. `vent_capacity` only truncates**; overheating and heat damage beyond the ventilation
capacity are not modelled.

### C. Provenance of parameters

| Parameter | Value | Status |
|---|---|---|
| `price_per_kg_fw` | 1.50 €/kg | ❌ the bib entry's URL is a site root, with no article and no date. **Highest profit elasticity (2.7×)** |
| `dm_fraction` | 0.055 | ❌ **no citation at all**, no comment and no bib entry. Elasticity 2.45× |
| `transplant_cost` | 1.00 | ❌ the source is in dollars (0.59 / 1.25), taken as euros at 1:1, no bib entry |
| `labour_per_plant_day` | 0.0087 | ❌ the derivation contains numbers we invented (wage level, harvest share) |
| `harvest_labour_per_kg` | 0.19 | ❌ as above |
| `co2_leak` | 1.3e-4 | ⚠️ originally back-solved (calibrated into the AHDB range); now used for the leakage term only |
| PAR→shortwave conversion | 2.1 mol/MJ | ⚠️ a standard agrometeorological conversion, **primary citation still to be added** |
| `gas_co2_factor` | 0.20 kg/kWh | ⚠️ a standard factor, **DEFRA/BEIS/IPCC citation still to be added** |
| `u_value` / `led_efficacy` | 4.0 / 2.5 | ⚠️ the ranges are cited; the point values are ones we picked inside them |
| `elec_price` / `heat_price` | 0.189 / 0.0605 | ✅ Eurostat, verified |
| the six site-distribution ranges | — | ✅ Eurostat / DLC / Kittas & Baille |

**The key property of group C: `price_per_kg_fw` and `dm_fraction` both scale revenue
linearly, and `R*` scales with them, so they cancel from every ratio the paper reports.**
Measured: after harvest labour and grading loss were added back, `R*`/revenue is 0.55% in
both cases, to the digit. They decide whether the euro figures look plausible; they decide
none of the conclusions.

### D. Estimator

**D1. `R*` does not converge in the number of atoms M.** Going from 80 to 320 preserves
only 0.81–0.92 of the level. **This is precisely why we report order only** — the rank
correlation over the same interval is 0.96–0.97.

**D2. Sequential conditioning collapses the effective atom count below 5** by the third
round.

### E. Action space

**E1. Replication does not pay at these budgets**, so an agent is never rewarded for
buying precision. This is an arithmetic consequence of limited parallelism rather than a
bug, but it does narrow what the benchmark can distinguish.

**E2. The action space stops at designs and recommendations**: an agent cannot choose what
to measure, when to measure it, or how many plants to grow per unit.

---

## What happens next

The environment does not change again. The next version is v2.0, and it is not part of
this paper.
