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
reconstruction cannot remove those effects. This preserves the v1 primary
margin value while changing the previously cleaner `rev_rate` field.

Version 1 transcripts remain the terminal-only comparison arm. They cannot be
treated as v2 closed-loop runs because their agents never received measurements
or chose within-cycle decision times. All v2 agent experiments therefore require
new execution traces under environment version 2.0.0.

The exact visible modalities, units and measurement errors are specified in
`docs/observation_protocol_v2.md`.
