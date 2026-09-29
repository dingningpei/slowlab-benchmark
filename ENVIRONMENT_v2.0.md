# SLOWLAB environment v2.0

Version 2.0 introduces within-cycle observation. A round now contains an
integer-day simulation clock, a persistent crop state, timestamped measurement
records and optional within-round recommendation updates. The terminal-only
`advance()` method is retained as the v1 control condition.

The terminal crop equations, task geometry, factor ranges and economic
coefficients are unchanged from v1.0. Given the same seed and submitted static
design, advancing directly to the end or inspecting the crop during segmented
advances yields the same terminal values.

The public terminal component fields were made algebraically consistent.
Chamber, loop and batch nuisance effects are assigned to observed revenue, so
`value = rev_rate - energy_cost_rate - other_cost_rate` and component
reconstruction cannot remove those effects. Revenue, energy and other-cost rates
are accounting records reported to four decimal places in EUR/(m2 day); total
cost and margin are recomputed from those displayed components so both public
identities remain exact. The resolution is about EUR 0.02/m2 over a 210-day crop
and is included in the evaluator likelihood.

Version 1 transcripts remain the terminal-only comparison arm. They cannot be
treated as v2 closed-loop runs because their agents never received measurements
or chose within-cycle decision times. All v2 agent experiments therefore require
new execution traces under environment version 2.0.0.

The exact visible modalities, units and measurement errors are specified in
`docs/observation_protocol_v2.md`.

## Version 2.1 task status

The formal benchmark suite contains **Sanity**, **Optimise**, and **Transfer**. The
**Screen** configuration remains executable for archival analyses and future task
development, but it is experimental and excluded from Version 2.1 model comparisons. Its
current final-regret endpoint scores high-dimensional operating optimisation rather than
factor screening. A future Screen task must elicit factor rankings, effect directions and
confidence, then score rank, sign and top-k selection quality directly. This reporting
change does not alter environment ground truth, so the environment version remains 2.0.0.
The complete scope and future screening-output contract are in
`docs/task_scope_v2_1.md`.
