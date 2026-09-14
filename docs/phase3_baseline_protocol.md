# Phase 3 baseline protocol

Phase 3 uses `phase3-baseline-0.1`. Every deployable scripted baseline is run
through `SlowLabEnv`, submits ordinary `Design` objects, receives only public
`history()` events, and is charged by the same round and unit limits. Mechanical
validation remains the environment's single feasibility gate.

The initial suite is:

- `split_plot_doe`: the existing screening then response-surface strategy, with
  chamber-safe whole-plot allocation and centre-point replication.
- `constraint_aware_batch_bo`: block-aware GP-UCB whose candidate allocation
  respects chamber controls. It chooses replication from the declared noise and
  minimum-effect regime rather than fixing one replication policy for every task.
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
