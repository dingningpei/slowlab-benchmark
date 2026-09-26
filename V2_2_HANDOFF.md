# v2.2 handoff — 2026-09-25

## Branch and latest state

Worktree: `/Users/dingningpei/Desktop/slowlab-v2-observation`  
Branch: `codex/v2.1`

Phase -1 reached the pre-registered temporal holdout. The frozen v8 hybrid model failed the real-trajectory gate. Do not refit v8 or reinterpret this holdout as development data while retaining a validation claim.

## Frozen holdout result

- 8 sequences, 137 compartment-days, 3,288 hourly samples.
- Solver completion: 100%; sampled physical violations: 0.
- Pooled corrected temperature RMSE: 2.579 °C (limit 2.04; fail).
- Pooled raw RH RMSE: 9.983 percentage points (limit 8.5; fail).
- Pooled raw CO2 RMSE: 211.8 ppm (limit 361; pass).
- Every-compartment requirement failed; exact values and audit trail are in `configs/agc2019_holdout_result_v8.json`.

## Scientific interpretation

The calibration-only residual layer transferred poorly to the late temporal holdout. Failures are heterogeneous: Automatoes and Digilog have large cold biases; AICU and TheAutomators have high humidity errors; Automatoes also fails CO2. Public AGC files do not identify enough physical boundary conditions to justify post-hoc compartment corrections. v8 is not evidence of an AGC digital twin and does not validate counterfactual policies.

A valid next attempt must be called a new prospective version and reserve a new untouched validation source. Preferred data are independent greenhouse trajectories containing aligned weather, realized actuator states, indoor climate, facility geometry, and actuator capacities. Otherwise narrow the paper to a synthetic benchmark and report AGC as a failed external-validity audit.

## Reproducibility assets

- Gate: `configs/agc2019_climate_validation_gate_v0.json`
- Holdout contract: `configs/agc2019_holdout_execution_v0.json`
- Residual protocol/model: `configs/agc2019_calibration_protocol_v8.json`, `configs/agc2019_temperature_residual_model_v8.json`
- Calibration outputs: `configs/agc2019_calibration_result_v7.json`, `configs/agc2019_calibration_result_v8.json`
- Holdout result: `configs/agc2019_holdout_result_v8.json`
- Runner: `scripts/run_agc2019_holdout_validation.py`
- Interpretation: `docs/agc2019_greenlight_feasibility.md`

Raw public data and generated replay trajectories are intentionally outside Git under `/private/tmp`. The private `output/reviews` directory must never be pushed.

## Working tree caution

The following pre-existing local changes are unrelated to this Phase -1 handoff and were deliberately not staged: `scripts/run_baselines.py`, `scripts/run_llm.py`, `slowlab/llm.py`, `slowlab/providers.py`, `tests/test_llm.py`, `tests/test_providers.py`, and untracked `scripts/run_v22_callable_matrix.py`. Review them separately before any later commit.

## Execution notes

The first holdout attempt exposed a runner-only candidate JSON wrapper bug; the fix is commit `afc286b`. Some workers had read initial sequence states, but no trajectory metrics were produced. A later command typo also stopped before observation access. Neither event changed the frozen model, identities, features, lambda, physical parameters, or thresholds. The completed retry followed the same frozen contract.
