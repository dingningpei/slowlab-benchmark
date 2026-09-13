# Version 2 within-cycle observation protocol

The simulator advances in integer crop days. A submitted design fixes the
treatments and allocation for the whole crop cycle. `advance_to(day)` moves the
current round monotonically to a round-relative day; `advance()` remains the
terminal-only compatibility interface. Final margin observations appear only
when the crop reaches `Task.duration_days`.

The environment instantiates each plant and draws its parameters once at the
start of a round. The same biological state is advanced across every call.
Reading a measurement uses a separate deterministic measurement-noise stream,
so inspecting the facility cannot change crop growth or final outcomes.

`observe(units, modality)` currently exposes read-only records:

| Modality | Unit | Method | Persistent unit bias SD | Per-day record SD | Resolution | Availability |
|---|---|---|---:|---:|---:|---|
| `setpoint:<factor>` | physical factor unit | environment sensor | 0 | 0 | 0.01 | day 0 onward |
| `canopy_lai` | m2 leaf / m2 ground | non-destructive canopy estimate | 3% | 2% | 0.001 | day 0 onward |
| `harvested_fresh_mass` | kg / m2 | harvest ledger | 1.5% | 1% | 0.001 | day 0 onward |
| `energy_cost_to_date` | EUR / m2 | utility meter | 0.3% | 0.2% | 0.0001 | day 0 onward |
| `other_cost_to_date` | EUR / m2 | cost ledger | 0 | 0 | 0.0001 | day 0 onward |

Percentage errors are multiplicative, records are clipped at zero, and the
result is rounded to the listed sensor resolution. Quantisation variance is
included in the evaluator likelihood. A bias is
shared by all dates for one unit and modality. The smaller record error varies
by day. Re-reading the same `(design, day, unit, modality)` returns the cached
record and creates no independent sample. Every new record has `kind=measured`,
round-relative and absolute timestamps, method, unit, value, cost and current
availability.

These are passive sensors or existing ledgers, so their marginal read cost is
zero. Active assays, destructive sampling, early termination, replanting and
mid-cycle treatment changes are outside this first protocol. Adding any of them
requires an explicit resource and state-transition model.

Latent node number, biomass pools, physiological parameters, future trajectory,
final margin and the oracle are private simulator state. Mature fruit is exposed
only after it appears in the harvest ledger. Accrued costs are observable; future
cost and terminal revenue are not.

At completion, chamber, loop and batch nuisance effects enter reported revenue.
Consequently every public terminal record obeys
`cost_rate = energy_cost_rate + other_cost_rate` and
`value = rev_rate - energy_cost_rate - other_cost_rate`. This prevents the
component fields from reconstructing a nuisance-free margin unavailable through
the primary response.
