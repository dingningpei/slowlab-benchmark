# Phase 3 baseline protocol

Phase 3 uses `phase3-baseline-0.1`. Every deployable scripted baseline is run
through `SlowLabEnv`, submits ordinary `Design` objects, receives only public
`history()` events, and is charged by the same round and unit limits. Mechanical
validation remains the environment's single feasibility gate.

The initial suite is:

- `split_plot_doe`: the existing screening then response-surface strategy, with
  chamber-safe whole-plot allocation and centre-point replication.
- `constraint_aware_batch_bo`: block-aware GP-UCB whose candidate allocation
  respects chamber controls. The registered comparison keeps the pre-existing
  one-unit coverage policy for continuity with the original GP-UCB reference.
- `adaptive_replication_bo`: the experimental noise/MDE rule tested in the first
  replication grid. It remains registered for audit but is labelled experimental.
- `component_reconstruction`: the component-wise reader paired with the same
  GP-UCB design loop; its distinct role is the Transfer price-shock question.
- `prior_optimal_fixed`: a no-experiment prior-optimal constant.
- `site_oracle`: the latent site's true optimum.

The last two entries are privileged diagnostics. `registry.describe()` labels
them and records why; tables must not present them as deployable agents.

`BlockAwareGP` represents persistent chamber intercepts and per-cycle batch
intercepts in the training covariance. Its prediction targets the population
response of a new campaign, so old nuisance intercepts are not propagated as
treatment effects. Plant-level uncertainty is reduced by the physical plant
count after denormalising density, while loop uncertainty remains unit-specific.
The final recommendation and probe intervals use this same reader.

The current replication rule has a regression test showing that low-noise
regimes select one unit per treatment and high-noise regimes select four. This
establishes that the benchmark pipeline can represent a coverage/precision
switch. It is not an empirical performance claim. The next calibration must run
a predeclared noise × slot-budget grid and evaluate the policy on held-out site
seeds. Length scale and covariance-scale sensitivity remain part of Phase 3.5.

A two-seed T1 smoke run verifies the runner and output schema only. It is not
included in paper results. Full Phase 3 experiments must write to a new versioned
output directory and must not overwrite Version 1 artifacts.

## First held-out calibration result

Protocol `phase3-block-reader-calibration-0.2` selected mean and uncertainty
parameters separately on seeds 200--207 for T1 and T3, then evaluated them once
on seeds 1200--1211. The block-aware reader did not beat the ordinary GP: mean
normalised-regret improvement was -0.0112 (SE 0.0095), and it won 37.5% of 24
paired episodes. Held-out interval coverage was 85.4% on T1 and 89.6% on T3
after validation-selected interval inflation. It remains an experimental reader,
not evidence that block modelling improves recommendations.

Protocol `phase3-replication-grid-0.1` then compared the same BO implementation
with adaptive, fixed-1, fixed-2 and fixed-4 replication on seeds 2200--2207. The
predeclared adaptive rule matched the best fixed policy in only 6 of 16 noise ×
slot-budget cells, so its gate failed. Fixed-2 was best in 7 cells, showing that
replication can help and that the benchmark does not uniformly reward maximum
coverage. These seeds are now frozen test data and cannot be used to tune a
replacement rule.
