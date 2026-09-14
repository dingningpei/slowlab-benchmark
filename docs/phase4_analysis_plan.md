# Phase 4 preregistered analysis plan

Protocol `slowlab-v2-phase4-confirmatory-2026-09-13` is frozen before any site
in the declared ranges is executed. The machine-readable source of truth is
`configs/phase4_preregistered.json`. Changes after the first confirmatory API
call require a new protocol id and are reported as amendments; existing results
remain under the original id.

## Questions and estimands

The primary question is whether design-only, inference-only, or combined passive
tools change final simple regret relative to bare prompting on T3 for
`openai/gpt-5.6-luna`. Each contrast is paired by site and provider generation
seed. Generation-seed contrasts are averaged within site before uncertainty is
computed, so the independent statistical unit is the site.

The primary family contains three contrasts and uses Holm family-wise error
control. The reported effect is tool minus bare, so negative values favor the
tool. We report the paired mean, median, standardized effect, a 10,000-resample
site-cluster bootstrap 95% interval, raw p value, and Holm-adjusted p value.
Final simple regret is the sole primary metric.

Secondary analyses test the inference contrast with the stronger
`openai/gpt-6-astra` model, test T1/T4 heterogeneity with the standard model, and
compare terminal-only with within-cycle feedback. Benjamini-Hochberg correction
is applied within each named secondary family. A small constraint-checklist
prompt study is exploratory and cannot support a primary claim.

## Precision and sample structure

The planning script reconstructs paired standard deviations from the twenty
corrected Phase 1 tool intervals. Their median, 75th percentile, 90th percentile,
and maximum are retained in the private precision report. The primary count of
56 sites targets an approximate 95% half-width of 0.018 at the 75th-percentile
historical paired SD, after finite-sample and unusable-episode inflation. Two
generation seeds per site quantify provider stochasticity without treating them
as independent sites.

All Phase 4 site ranges start at 6000 and were unused during development,
calibration, model selection, and Phase 3 mechanism pilots. No result from these
ranges may be used to change prompts, tasks, hyperparameters, hypotheses, or the
analysis code under this protocol.

## Execution and missing data

The runner records both requested and returned model ids, provider metadata,
generation seed, temperature, maximum output tokens, reasoning effort, token
usage, retry attempt, finish reason, prompt-protocol hash, and realized episode
prompt hash. A transport or rate-limit failure is retried using exactly the same
site and generation seed. Model-generated invalid JSON, infeasible submissions,
and voluntary stopping remain outcomes. Unresolved cells are reported and never
imputed.

There is no efficacy stopping. Runs may pause for cost limits or service
availability without opening condition summaries. Evaluator-based conclusions
use three independent evaluator seeds and include Monte Carlo variation.

## Claim rule

Only primary or prespecified secondary effects whose direction survives the
independent sites, multiplicity control, task/model heterogeneity reporting, and
relevant sensitivity analysis can enter the title, abstract, or contribution
list. Exploratory prompt results and two-site Phase 3 pilots remain labeled as
hypothesis-generating evidence.
