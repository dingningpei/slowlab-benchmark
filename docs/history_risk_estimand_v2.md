# Version 2 history-conditioned risk estimand

At simulation time `t`, `H_t` is the ordered result of `env.history(t)`. It
contains only submitted designs, measurements already made, terminal records
already completed, forfeitures, and recommendations already stated. Future
terminal outcomes and latent simulator state are absent.

For posterior belief `p(theta | H_t)`, define

`b(H_t) = min_x E[R(theta, x) | H_t]`.

For the agent recommendation `x_hat_t`, define

`a(H_t, x_hat_t) = E[R(theta, x_hat_t) | H_t]`

and posterior excess risk

`g(H_t, x_hat_t) = a(H_t, x_hat_t) - b(H_t) >= 0`.

The implementation inserts `x_hat_t` into the numerical action set before
minimising, so nonnegativity follows from the definition rather than clipping.
The evaluator has privileged access to the site prior, forward model and oracle;
the agent does not.

The terms diagnose different limits. `b(H_t)` is the residual decision risk
after the executed design and visible observations, even for a Bayes-optimal
reader; it measures information left unresolved by the history. `g(H_t,
x_hat_t)` is additional loss from the agent's recommendation given that same
history. Error caused by an agent using the wrong scientific model belongs to
the reader/model-knowledge mechanism and can contribute to `g`; it is not
identified as a separate numeric addend without an explicit model-family
intervention. Evaluator discretisation and likelihood misspecification are
estimation error and must be reported through calibration, rather than labelled
as agent failure.

For a proposed design `D` after history `H`, conditional design value is

`v(D | H) = b(H) - E[b(H, D, Y_D) | H, D]`.

This conditions on all earlier evidence. It must not restart from the prior each
round. A realised decrease `b(H_before) - b(H_after)` is a useful trajectory
diagnostic but is not the conditional expected value above.

`conditional_design_value_full_history` evaluates this expectation by sampling
the new record vector from each atom's Gaussian conditional distribution given
the existing history, including covariance shared across the boundary.
`evaluate_history_trajectory` applies the same action set to every visible
recommendation and reports `b(H_t)`, agent posterior risk, excess risk, and the
conditional value of records acquired since the preceding recommendation. If a
measurement schedule was itself chosen after earlier within-cycle values, the
reported value is for that realised acquisition set; evaluating the adaptive
selection policy would require integrating over its branching decisions.

`history-risk-0.1-terminal` handles completed margin records jointly. Its
covariance contains persistent chamber and loop effects, round-specific batch
effects and unit observation variance. It deliberately raises
`IncompleteLikelihoodError` if it is accidentally called on within-cycle data.

`history-risk-0.2-full-fields` handles the independent basis of all visible
numeric fields: terminal revenue, energy cost and other cost, plus every
within-cycle modality. Margin itself is omitted from the likelihood because it
is the exact linear combination of the three terminal components. Duplicate
exact ledger values are also removed. The likelihood estimates plant-level
cross-time and cross-modality covariance by parameter-perturbation Monte Carlo,
divides it by the unit's plant count, and adds persistent measurement bias,
per-day record error, chamber, loop and batch covariance according to their
actual scopes. It reports its Monte Carlo size and seed. Until Phase 2.5
calibration estimates numerical error across independent seeds, the result
carries the failure flag `mc_error_not_estimated` and is exploratory.

The old `R*(D)` remains a prior design-geometry diagnostic. It must not be called
`b(H)` because it does not condition on the realised full history, and the old
efficiency ratio must not be described as the fraction of acquired information
the agent used.
