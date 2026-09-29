# SlowLab

**A benchmark for experimental-design agents when experiments are slow, noisy, expensive
and irreversible.**

SlowLab is an executable benchmark for agents that allocate treatments, inspect evidence
and make operating recommendations in a resource-limited experimental campaign. Its current
controlled-environment horticulture domain combines four conditions:

| Condition | How it appears here |
|---|---|
| **Slow** | a trial is a growing cycle. A task gives an agent two or three of them, for 420–630 simulated days in total. |
| **Noisy** | plant-to-plant, loop-to-loop, chamber-to-chamber and batch-to-batch variance, none of it removable by asking again. |
| **Costly** | a trial is priced in the objective's own units — running a bad treatment *is* the loss. |
| **Irreversible** | a submitted treatment cannot be changed or cancelled during its 210-day crop cycle. |

An agent designs a feasible split-plot experiment, commits named units, may inspect
time-appropriate measurements while the crop develops, and recommends operating
conditions. Ground truth uses the reduced state-variable TOMGRO crop model (Jones, Kenig
and Vallejos, 1999) with an explicit greenhouse cost model. The published model provides
traceability; task budgets, ranges, facility shape and observation channels remain benchmark
choices. The frozen environment and package remain at **v2.0.0**; the Version 2.1 reporting scope
uses the same environment and does not change ground truth. Version 1 remains an audit
artifact described in [`ENVIRONMENT_v1.0.md`](ENVIRONMENT_v1.0.md).

### Version 2.2 reality boundary

The Version 2.2 redesign is a **reality-constrained simulation benchmark**, not a validated
digital twin. A preregistered GreenLight v8 observed-action holdout failed the trajectory
gate: pooled temperature RMSE was 2.579 °C against a 2.04 °C limit and relative-humidity
RMSE was 9.983 percentage points against an 8.5-point limit; CO2 passed its separate
threshold. Public AGC data therefore constrain observed ranges, dynamics, missingness,
facility capacities and executor sensitivities, while the failed audit remains an explicit
external-validity result.

The evidence classes, source roles, supported ranges and unsupported actuator fluxes are
frozen in [`configs/v22/reality_constraints_v2_2.json`](configs/v22/reality_constraints_v2_2.json)
and explained in
[`docs/v22/reality_constraints_v2_2.md`](docs/v22/reality_constraints_v2_2.md). Formal Version 2.2
records must pass `scripts/v22/audit_reality_support.py`; passing that range audit does not
establish causal or counterfactual validity.

Formal model calls are additionally governed by
[`configs/v22/simulation_blinding_v2_2.json`](configs/v22/simulation_blinding_v2_2.json). Public
datasets define aggregate constraints but are not replayed as confirmatory outcomes. Site
seeds, simulator identity, latent parameters, oracle values and future records are blocked
from provider messages by a fail-closed prompt firewall; see
[`docs/v22/simulation_blinding_v2_2.md`](docs/v22/simulation_blinding_v2_2.md).

---

## Install and run

```bash
pip install -r requirements.txt
python3 -m pytest -q
python3 scripts/run_baselines.py           # the three scripted reference strategies
```

An episode, in full:

```python
from slowlab import SlowLabEnv, TASKS, Design

env = SlowLabEnv(TASKS["Optimise"], seed=0)   # a seed draws a *site*, not a noise draw

while env.rounds_left:
    u = env.available_units()                 # excludes units still in heat recovery
    d = Design(
        treatments={"A": {"day_temp": 0.3, "density": 0.5},   # factors are coded to [0, 1]
                    "B": {"day_temp": 0.7, "density": 0.5}},
        allocation={"A": u[:6], "B": u[6:12]},
        randomization_seed=1,
    )
    r = env.submit_design(d)                  # a falsy Rejection, or the round index
    env.advance_to(105)                       # partial time passes
    obs = env.observe(modality="canopy_lai") # currently available records only
    env.advance_to(210)                       # terminal outcomes become available
    env.interim_recommendation(x_now)         # scored every round, costs nothing

res = env.submit_recommendation(x_final)
print(res.simple_regret, res.cumulative_regret)
```

Factor values are coded to $[0, 1]$ over the task's range — `day_temp` spans 18–32 °C on
**Optimise**, `density` 2.2–4.5 plants m⁻². `env.validate_design(d)` answers the same
question as `submit_design` without committing anything.

The interface guarantees timing and feasibility and nothing else. It never reveals the
site parameters $\theta$, the instance distribution $P_\Theta$, or the forward model.
Feasibility checking is free and unlimited, and a rejected design costs no budget, so an
agent can never lose a cycle to a formatting mistake.

To run language models, see [`RUNNING.md`](RUNNING.md).

---

## Core tasks

The formal Version 2.1 suite contains three tasks. They share the same variance components,
cycle length, per-unit cost and commitment structure; each tests a distinct decision.

| Task | What the agent must do | Factors | Facility | rounds × units | days |
|---|---|---|---|---|---|
| **Sanity** | close the loop and calibrate the evaluator | 1 (day temperature) | 4 × 4 | 2 × 8 | 420 |
| **Optimise** | find the optimum, respecting the facility | 2 (mixed level) | 4 × 3 | 3 × 12 | 630 |
| **Transfer** | survive an energy price shock | 4 (resource) | 8 × 3 | 3 × 24 | 630 |

*Facility* is chambers × loops. *Mixed level* means one chamber-level and one loop-level
factor, so a valid design has to be a split-plot. In **Transfer**, electricity and gas
both rise by a factor of 2.2 after the campaign ends and the agent must recommend again
**with no further experiments** — an agent that modelled revenue and the two cost streams
separately can re-solve; one that modelled only the margin cannot.

**Screen** remains available in `TASKS` as an experimental configuration for archival and
future development, but it is excluded from the core suite and Version 2.1 model claims.
Its current endpoint is final operating regret, which does not directly measure screening.
A future task must elicit factor ranks, effect directions and confidence, and score rank,
sign and top-k selection quality directly. The status and redesign contract are recorded in
[`docs/task_scope_v2_1.md`](docs/task_scope_v2_1.md).

---

## What is measured

Four quantities per episode, never combined into one score.

**Realised regret** $\bar{\mathcal{R}}$ — the loss of the recommendation against the
site's true optimum, elicited after *every* round rather than only at the end.

**Cumulative regret** $\mathcal{R}_{\mathrm{cum}}$ — what the experiments themselves
forwent, in the same currency, summed over occupied slot-days. An agent that finds the
optimum by first running every bad treatment has answered the question and exhausted the
facility that paid for it.

Version 2 conditions both diagnostic quantities on the same timestamped visible history
$H_t$. The remaining Bayes risk is
$b(H_t)=\min_x\mathbb{E}[R(\theta,x)\mid H_t]$; the agent's posterior excess risk is
$g(H_t,\hat x_t)=\mathbb{E}[R(\theta,\hat x_t)\mid H_t]-b(H_t)\geq0$.
Crop and economic parameters use separately calibrated continuous posteriors, terminal
revenue couples them, and independent posterior samples select and evaluate the action.

The Version 1 ratios $c$ and $\eta$ are withdrawn. In particular, $\eta$ cannot be read as
the fraction of acquired information used, and their correlation, opposite movement, or
product does not establish independent design and reader failures.

---

## Headline results

The frozen Version 2 study contains 928 episodes; a separately frozen adaptive-noise
extension contains 384 more. A prospective post-primary Version 2.1 extension adds 288
same-site Optimise episodes across Luna, DeepSeek V4.1 Flash, and Qwen 3.8 27B. Sites are
the statistical units and provider generation seeds are within-site replicates.

- On 56 paired Optimise sites, Luna's design, inference and combined aids do not
  significantly change final simple regret after Holm correction.
- On 24 independent sites, Sol has lower descriptive regret than Luna, but its inference aid
  does not improve its paired bare arm.
- A fresh 16-site extension reproduces the tool null result at twice the observation noise.
  A prespecified secondary analysis finds that the design aid is harmful at half noise.
- Within-cycle observation changes behavior but does not significantly improve final regret
  in the current synchronous action space.
- Fixed-history replay shows that design source and history reader are separable effects,
  with their relative sizes changing by task.
- In the 288-episode same-site model extension, the inference aid leaves Luna unchanged,
  is directionally favorable but uncertain for DeepSeek, and significantly lowers Qwen
  final regret by 0.0146 after BH correction. Qwen's bare-arm gap is absent under inference.

The compact numerical sources for the paper are
[`results/phase5_paper_summary_env2.0.0.json`](results/phase5_paper_summary_env2.0.0.json)
and [`results/phase5_model_extension_summary_env2.1.0.json`](results/phase5_model_extension_summary_env2.1.0.json).

---

## Repository layout

Three generations of work live here. The frozen Version 2.1 stack stays flat, because its
scripts, results and paper are a fixed artifact. The Version 2.2 redesign and the retired
AGC calibration line each have their own sub-tree.

```
slowlab/            frozen Version 2.1 environment and evaluators
  env.py              episode loop, commitment, timing, feasibility
  tasks.py            three core tasks plus the experimental Screen configuration
  world.py            P_Theta: a seed draws a site (climate, prices, crop parameters)
  tomgro.py           reduced state-variable TOMGRO
  economics.py        greenhouse gross-margin model
  facility.py         chambers and loops; the control hierarchy
  design.py           the submitted design and rejection semantics
  history_risk.py     common-history Bayes-risk evaluator
  eventlog.py         timestamped execution records
  agents.py           scripted reference strategies
  llm.py              the language-model harness (prompt, retry loop, tools)
  providers.py        chat APIs over urllib; no SDK required
slowlab/v22/        the Version 2.2 event-driven greenhouse campaign
  task_contract.py    policy schema and contract validation
  campaign_executor.py four-compartment executor with a JSON-only dispatch surface
  controller.py, resources.py, feedback_view.py, online_observations.py, sensor_bridge.py
  greenlight_reuse.py, cached_solver.py, native_rhs.py, greenlight_adapter.py   pinned GreenLight
  cabauw_weather.py   historical weather forcing (Cabauw lc1)
  prompt_firewall.py, reality_constraints.py, frozen_paths.py
slowlab/archive/    retired directions kept for the appendix: AGC 2019 calibration
                    (agc_*, greenhouse_data, greenlight_sequence), root-zone modules and the
                    failed private weather generators (private_weather*, weather_similarity)
scripts/            Version 2.1 reproduction path (table below)
scripts/v22/        Version 2.2 integration checks, audits, pilots and verifiers
scripts/agc/        AGC 2019/2023/2024 audit, calibration and holdout scripts
configs/            frozen Version 2.1 protocols (phase4_*, phase5_*, block_aware_gp_phase3)
configs/v22/        Version 2.2 inputs: task contracts v0-v3, campaign example, model family,
                    reality constraints, blinding policy, weather plans and protocols
configs/agc/        AGC contracts, calibration protocols and their recorded results
results/            frozen Version 2.1 summaries (inputs to the paper build)
results/v22/        Version 2.2 pilot results, structural audits and daily progress traces
docs/               Version 2 protocol documents (observation, history risk, task scope, phases)
docs/v22/           Version 2.2 design decisions and the Phase 0-1 lab notebook
docs/agc/           AGC feasibility and audit write-ups
tests/              Version 2.1 regression, protocol and claim tests
tests/v22/, tests/archive/   tests for the two sub-trees above
paper/              the LaTeX source
```

Frozen JSON artifacts under `configs/v22/`, `configs/agc/` and `results/v22/` are byte-identical
to their registration and may still name files by the earlier flat paths (`configs/v22_*.json`).
`slowlab.v22.frozen_paths` resolves those names; verifiers use it rather than editing hashed files.

**`scripts/` is the reproduction path.** Nothing in the paper is a one-off analysis:

| Script | What it produces |
|---|---|
| `run_llm.py`, `run_all_llm.sh` | the language-model episodes |
| `run_baselines.py` | the three scripted reference strategies |
| `run_achievable.py`, `rescore_cv.py` | $\mathcal{R}^\star(D)$ under the cross-validated estimator |
| `build_phase5_paper_artifacts.py` | compact Version 2 summary and the three main result tables |
| `build_phase5_model_extension_artifacts.py` | aggregate Version 2.1 model-extension summary and table |
| `reproduce_phase5_paper.py` | raw frozen events → analyses → tables → figures → checked PDF |
| `verify_results_claims.py` | verifies the Phase 5 paper summary and manuscript claims |
| `tool_regret.py`, `tool_design_width.py`, `tool_hyper_control.py` | the three legs of the tool analysis |
| `factor_value.py` | the cost of setting a factor wrongly, against the value of locating it |
| `rstar_convergence.py`, `eig_rank_stability.py` | why the estimator is read as ranks |
| `check_crossrefs.py` | orphan labels, dangling refs, duplicates |

The public v1.0.0 release contains the archived 800 Version 1 transcripts. Version 2 raw
episodes are kept outside Git and are packaged separately for anonymous review; private
notes under `output/reviews/` are never part of the repository artifact.

With that frozen Version 2 bundle unpacked locally, the complete paper build is one command:

```bash
python3 scripts/reproduce_phase5_paper.py --analysis-dir /path/to/slowlab-v2-frozen
```

The frozen bundle must include `phase5_model_extension_combined_a3/` and
`phase5_model_extension_a3/` alongside the Phase 3/4 inputs. The first contains the
combined 288 formal episodes; the second supplies the A3 DeepSeek/Qwen cost audit.

---

## Adding a domain

Several domains satisfy the four conditions: microbial evolution, bioprocess scale-up,
materials ageing, dose-finding trials. Adding one requires no environment code — a domain
supplies a forward model $f_\theta$, an instance distribution $P_\Theta$, a facility with
its control hierarchy, and a price list. The appendix of the paper sets out what a new
domain has to bring.

---

## Citation

See [`CITATION.cff`](CITATION.cff). Licensed under the terms in [`LICENSE`](LICENSE).
