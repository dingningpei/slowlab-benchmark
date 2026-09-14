# Phase 3 mechanism results

This document records the confirmatory protocol and the conclusions that the
implemented evidence currently supports. Raw rows and private review notes live
under the ignored `output/reviews/` directory.

## Fixed-history design by reader replay

Each design policy generated an accepted, completed history once. Ordinary GP,
block-aware GP, component GP, and the privileged site oracle then read copies of
that same history. Readers could recommend a point but could not alter designs or
observations. On held-out seeds 4000--4007, the deployable-reader decomposition
gave an overall design-effect range of 0.01718 regret, a reader-effect range of
0.00619, and interaction RMS 0.00209. The design effect was largest on T4
(0.04727), while T1 was the one task where the reader range exceeded the design
range. These results support separating design quality from history reading; they
do not support a universal claim that either component dominates on every task.

The calibrated block-aware reader did not beat ordinary GP on its earlier
held-out test: mean paired normalised-regret improvement was -0.0112 (SE 0.0095;
37.5% wins over 24 pairs). It remains an explicit structured reader rather than a
claimed strong winner.

## Feedback and constraint counterfactuals

The scripted mechanism grid used held-out seeds 5200--5211. With total capacity
fixed at 24 slots, mean regret was 0.01883 for 2x12, 0.00562 for 3x8, 0.00783 for
4x6, and 0.00417 for 6x4 rounds by slots. Feedback frequency matters in this
setting, but the non-monotone middle cells and twelve-site uncertainty rule out a
simple linear claim.

Removing chamber-shared control while keeping the 3x12 budget reduced mean
regret from 0.00919 to 0.00484. Scaling the declared noise components from 0.5x
to 1.0x to 1.5x produced mean regrets 0.00855, 0.00919, and 0.01235. Both results
match the proposed mechanisms, subject to replication on the larger Phase 4
sample.

Fixed and actively timed within-cycle inspection used exactly two canopy visits
per round. The active rule moved the second visit from day 140 to day 105 in all
36 rounds, and both inspection arms detected at least 10% treatment separation
in all twelve sites. Final recommendations and regrets were exactly identical to
terminal-only BO. This is the expected causal result under the current
synchronous protocol: inspection can update an interim recommendation, while
the submitted treatment cannot change and the next design receives complete
terminal outcomes. A claim of final-regret benefit therefore requires an action
space with early stopping, treatment modification, or staggered capacity.

Early termination and asynchronous batches are not represented as completed
ablations. The current environment lacks cleanup time, residual value, replant
dynamics, staggered unit release, and a matched oracle for those actions. Smaller
synchronous batches are only a feedback-frequency counterfactual.

## Tool quality

GP settings were selected on seeds 5400--5407 and evaluated without reselection
on seeds 5500--5507, using fixed accepted histories from constraint-aware batch
BO. The selected test mean regret was 0.00635 on T1, 0.01673 on T3, and 0.05114
on T4. T1 and T4 selected block-aware configurations; T3 selected an ordinary GP
and deteriorated sharply from validation (0.00328) to test (0.01673). Thus the
experiment rules out one fixed default as the sole explanation, but exposes
substantial hyperparameter-selection instability on small validation samples.

All block-aware fits retain chamber, batch, heteroscedastic observation-noise,
and cross-round design identifiers from the actual history. Phase 4 should expand
selection and evaluation sites before treating any chosen configuration as a
benchmark winner.
