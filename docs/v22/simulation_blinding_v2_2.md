# Version 2.2 simulator blinding and public-data contamination control

## What must be hidden

The model needs factor semantics, feasible bounds, facility constraints, the current time,
its own actions and observations already produced. It does not need the benchmark name,
public-data source, simulator family, code path, site seed, latent parameters, generator,
oracle, future observations or counterfactual outcomes. The formal API payload must contain
only the first set.

The executable policy is
[`configs/v22/simulation_blinding_v2_2.json`](../configs/v22/simulation_blinding_v2_2.json).
`slowlab.v22.prompt_firewall.BlindedCompleter` audits the complete outbound message list
immediately before a provider call and fails closed when a forbidden identity or
hidden-state marker is present. OpenRouter application-attribution headers are disabled by
default, so provider metadata does not identify SlowLab or its repository.

## Public-data familiarity

A model may know public greenhouse papers and datasets. General agronomic knowledge is not
leakage: a real scientist also starts with prior knowledge, and the zero-data arm measures
its value. The prohibited advantage is knowledge of an exact hidden site or future result.

To prevent public-data recall from solving the task, confirmatory episodes will not replay
public outcome rows or exact weather windows. Public sources may determine aggregate
ranges, correlations, cadence, missingness and equipment limits. A frozen generator then
creates new private sites and trajectories. Before any API call, every generated trajectory
must pass a preregistered exact/near-duplicate comparison against every source trajectory.

## Commit and reveal

The generator, exclusions, statistics and similarity threshold are frozen first. A private
confirmatory seed list is then generated and kept outside Git and provider payloads. Only a
salted cryptographic commitment is recorded before execution. The seed list and salt are
revealed after the formal matrix is complete and locked. This allows later reproduction
without making hidden sites reconstructible during evaluation.

## Limits of the claim

The firewall, private sites, similarity audit and commit/reveal procedure establish
mechanical information isolation. They cannot reveal the contents of a proprietary model's
training corpus. The paper must therefore report the controls and residual contamination
risk, rather than claim proof that a model never encountered a public source.
