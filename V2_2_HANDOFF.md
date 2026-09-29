# v2.2 handoff — 2026-09-25

## Current redesign — 2026-09-27

## Latest Phase 1 update — 2026-09-28

Latest numerical follow-up: the day-240.7569 overflow was an intermediate `exp` overflow inside a condensation sigmoid. Python reference and old native final derivatives were finite and agreed within 7.63e-17. Stable algebraic emission passed the exact-state replay and a 600-step native/reference trajectory regression. The bounded **four-unit fixed-policy annual run passed all 365 days** on Tokyo: 105,120 synchronized steps, 8,343.18 s, peak RSS 252,141,568 bytes under the 350 MB cap. An independent structural audit passed clocks, lifecycle events, resource accounting, final states and limits. This is a fixed-policy stability/resource gate, not a complete event-driven LLM campaign or physical greenhouse validation. See `configs/v22_remote_four_unit_annual_stable_result.json` and `configs/v22_remote_four_unit_annual_stable_audit.json`.

Latest weather follow-up: the v2 2020 monthly diagnostic had a leap-month grouping bug; corrected maximum is 1.672 training SD, and the independent wind daily-variance failure remains. V3a multiday blocks had 48/48 near-replay rejections. V3b cross-year intraday blending passed 8/8/6/9 of 12 candidates across four development folds but 2020 distribution coverage and energy score remain weak. Do not freeze v3b. After explicit user authorization, 2012/2013/2014/2016 development weather was downloaded within the exact 13,027,903-byte net-new cap. Full audit found 89 consecutive missing 2012-12 LWD records; 2013/2014/2016 are complete development years, with substantial meteo/radiation SWD product disagreement disclosed and radiation SWD selected a priori. No 2015/2025 holdout files were downloaded. See `docs/v22_weather_v3_development_acquisition.md` and `configs/v22_weather_v3_development_full_audit.json`.

After the annual pilot passed, v3c ran an outcome-blind seven-fold weather-only development pilot on 2013/2014/2016/2017–2020 (12 fixed attempts per fold). It passed physical support 84/84 and the current aligned-window non-replay check 75/84, but its sample-size-corrected energy score was worse than simply resampling a real training year in **all seven folds**. Do not freeze v3c or download 2015/2025 holdouts for it. See `docs/v22_weather_v3c_development_result.md` and `configs/v22_weather_v3c_sevenfold_development_pilot.json`. A real historical-weather boundary with blinded identity/future is the leading alternative, subject to a separately designed site/parameter distribution and weather-year blocking.

Newest blocker: candidate-v2 private weather generator FAILED its locked 2020 cross-year distribution diagnostic (monthly mean 1.631 training SD > 1.25; minimum daily variance ratio 0.451 < 0.50, wind). Physical support and non-replay passed. Preserve `configs/v22_weather_generator_2020_diagnostic_lock.json`, `configs/v22_weather_candidate_v2_2020_diagnostic.json`, and `configs/v22_weather_candidate_v2_2020_gate.json`; do not retune v2 on 2020 and continue calling 2020 untouched. Candidate-v3 needs a new independent diagnostic source and versioned rules. No formal private sites/seeds are frozen.

A real GreenLight/Cabauw four-unit 577-tick dynamic campaign prefix passed staggered starts, early stop, two-day cleanup, replant, causal public observation and private tick-trace digest checks (`configs/v22_campaign_dynamic_prefix_result.json`). Full 365-day stability and process-level API isolation remain open. The four-unit event-driven campaign executor has a JSON-only public dispatch surface, Full/Endpoint observation filtering, staggered planting, stop/cleanup/replant, safety stop, resource accounting and optional streamed private trace. Lightweight injected-backend tests pass; no complete GreenLight campaign has been run through this executor. A 14-component evidence register classifies observed, literature-constrained, assumed and unsupported mechanisms. The annual 8°C/20°C deep-soil sensitivity remains planned, not passed. This is not a frozen site distribution; quality thresholds, formal site partition and seed commitment remain. See `docs/v22_phase1_parallel_work.md` and the current weather acquisition update above. No LLM API call or Git push in this update.


LATEST: Approved Cabauw lb1 98-file/21,770,259-byte and lc1 gapfilled
98-file/18,413,090-byte acquisitions completed into separate Git-external
/private/tmp caches. Both sets have SHA-256/size audited. Full reports:
configs/v22_weather_quality_audit.json and
configs/v22_weather_gapfilled_quality_audit.json; interpretation:
docs/v22_weather_feasibility.md and docs/v22_weather_lc1_audit.md.
Original lb1 joint required-channel missingness is 2.31–6.33% per year with
multi-day gaps; lc1 2017–2020 has zero joint missing across 210,384 10-min
intervals. This is gapfilled/merged external weather, NOT original continuous
measurement nor greenhouse validation. lc1 TA002/TD002 are labeled degC but
actual values are K-scale and match lb1 K at valid anchors. Source indices vary,
and nighttime SWD includes small negatives. Dataset-specific fail-closed conversion now exists in
slowlab/v22_cabauw_weather.py; optional weather channels are registered before
GreenLight compilation and supply only one current input row per 300s step.
configs/v22_weather_reader_smoke.json passed 49-month boundary reads and six
short dynamic steps; 10 focused existing tests passed. This is NOT annual or
agent-observation causality validation. Weather protocol, CO2/soil assumptions,
independent partition remain unfrozen. Prior sampled-screen v2 601-step test passed (89 light tests);
A six-step weather→controller→GreenLight→OnlineObservations→ledger bridge now
passes with fixed CET midnight origin; 10-min exterior means release only at
right endpoints, while 300s interior endpoints release after solving. Public
indoor and private controller weather channels are separate. See
scripts/check_v22_online_weather.py, configs/v22_online_weather_bridge_smoke.json
and docs/v22_online_weather_bridge.md. This remains one active compartment;
shared capacities, complete public channels/permissions, lifecycle campaign,
noise and annual validation remain open. Phase 1 incomplete. No LLM API
experiment or Git push this turn.

Deep-soil boundary audit: native `tSoOut` is fixed 20°C and represents ~2 m external soil. Cabauw field soil product reaches only 50 cm and has NOT been downloaded. Explicit constant scenario input and a six-step 8°C/20°C sensitivity check are implemented; no detectable 30-minute room-climate difference is not evidence about an annual campaign. See `docs/v22_deep_soil_boundary_audit.md` and `configs/v22_deep_soil_boundary_short_sensitivity.json`. Annual deep-soil boundary and sensitivity remain open.

Four-compartment capacity gap: v2 defines only per-compartment heat/CO2/lamp maxima, no central supply caps or allocation rule. Do not claim shared contention or invent central limits. Recommended main environment: independently capped compartments with adequately sized central supply; version contract before full scheduler. See `docs/v22_shared_capacity_gap.md`.

Latest Phase 1: User approved adequately sized central supply; versioned contract v3 retains independent per-compartment limits. Four active units on one clock passed a six-step Cabauw weather gate, all eight public sensors and Full/Endpoint running permissions, with measured heat/CO2/lamp fluxes at but not over per-unit maxima. Fixed non-midnight origin clock bug: controller and model now use the same fixed-CET offset. One unit passed two-day weather-driven stop/cleanup/replant with run-specific Endpoint accounting. Compiled RHS made the short lifecycle 11.67s, with 8.30s model loading and 3.36s two-day empty stepping; linear four-unit annual projection is ~0.68h, ~698 worker-hours for 1024 planned campaigns before overhead. Compute feasibility remains open. See `docs/v22_phase1_four_unit_weather_and_runtime.md`; 80 focused tests pass. No new data download or LLM API call.

2026-09-28 runtime follow-up: Causal observation index and optional all-auxiliary array output reduced a 576-step weather-driven empty cleanup segment from 3.36s to ~2.47s. Paired frame/array short runs have identical 28 final states and resource ledger. The optimistic empty-crop extrapolation is still ~0.50h per four-unit year and ~513 single-worker hours for the old 1024-campaign draft; mature-crop annual pilot and compute budget are open. See `docs/v22_phase1_runtime_optimization.md`. Do not launch formal matrix on MacBook.

A seven-day early active-crop window (2016 steps) with the array-output option took 8.24s stepping; frame mode 9.61s. Final 28 states and ledger exactly matched. The early-crop linear estimate is ~0.48h per four-unit year / ~489 worker-hours for the 1024-campaign draft, still not a mature/full-year benchmark. See `docs/v22_phase1_runtime_optimization.md`.

Remote preflight 2026-09-28: a private Tokyo Linux host has 2 vCPU (one physical core), ~896 MB RAM, Python 3.11 and GCC. A checksum-verified code bundle, isolated numerical venv, and, with user authorization, the audited 98-file/18,413,090-byte Cabauw lc1 product were transferred; no .env, keys, or reviews. Same-server six-step GCC-versus-reference check agreed to <7.23e-10 absolute state difference; a seven-day active-crop run took 37.27s stepping / 49.80s total, ~4.5× the Mac stepping time. Same-server DataFrame/array seven-day outputs matched exactly, but Mac-versus-Linux closed-loop seven-day final state max relative difference was ~1.59e-4 and heat-ledger relative difference ~0.175%. In 77/2,016 steps at least one controller command differed by >0.01; replaying the Mac command trace on Tokyo made the heat ledger identical and all final states agree within 2.33e-11 relative. Thus feedback amplifies tiny platform numerical differences. Pin one runtime for formal comparisons; the initial difference source remains open. The server is suitable for single-worker pilot, not a fast 1024-campaign matrix. See `docs/v22_remote_compute_preflight.md`.

Annual follow-up: user approved a single-compartment 365-day pilot with 3-hour/600 MB maximum. First run with a stricter *virtual address-space* cap segfaulted after day 59; second used a true 600 MB cgroup but host-wide OOM killed it after day 290 (896 MB host, no swap). `OnlineObservations` had a redundant global identity dictionary: removing it and using slotted immutable records reduced a 200k-record Mac sample from ~109 MB to ~66 MB; focused tests pass and old/new one-day state and ledger match exactly. Third run used an even safer 350 MB cgroup and **passed all 365 days / 105,120 steps**, including 180+2+180+2+1-day lifecycle. Wall time 2,119.19s (35.32min), peak RSS 271.22 MB; old/new first 290 daily harvest totals match exactly. Result and all three daily progress traces are in `configs/v22_remote_annual_pilot_*`; interpretation in `docs/v22_remote_compute_preflight.md`. A naive four-unit serial projection is ~2.35 worker-hours/campaign, or ~2,411 worker-hours for the draft 1,024 campaigns before LLM/evaluation/retries. Simultaneous four-unit memory/runtime remains untested, and no LLM or formal matrix has run.

Four-compartment runtime follow-up (2026-09-28): same-clock 1-day and 7-day gates passed; 7-day object and packed observation stores have identical four-unit final states and ledgers. A staggered 18-day lifecycle prefix passed under a 350 MB cgroup: 5,184 shared steps, 480.63s, peak RSS 153.17 MB; unit 0 stopped day 14, stayed in cleanup for two days, and replanted day 16 with no cleanup harvest. Full 365-day four-unit pilot FAILED at day 240.7569 / 69,338 steps with native RHS floating-point overflow flag 8; wall 5,488.11s, peak RSS 220.43MB, so neither cap caused it. Original failure artifacts are preserved; a bounded 241-day diagnostic replay with exact failing RHS state/input capture and reference RHS comparison has started; see `docs/v22_remote_compute_preflight.md`. No formal LLM matrix or API calls.

2026-09-28 independent Phase 1 work while the detached annual pilot runs: source-weather partition v0 is now machine-verified (2017–2018 development/fit, 2019 selection, 2020 out-of-year diagnostic; **not** formal private confirmatory weather). The 18-day four-unit artifact has a reproducible structural audit. A private optional per-tick gzip trace and fault-injection-tested audit were added to the bounded 1–7-day runner, but its dynamic trace gate has not yet run; the ongoing annual pilot contains only daily summaries and must not be called a complete executor activity log. See `docs/v22_weather_partition_protocol.md` and `docs/v22_phase1_event_trace.md`.

Phase 1 dynamic-input registration is FIXED. ReusableGreenLight loads a single current
bootstrap CSV before compilation, registering all five command channels. Six 300s steps
match all 28 states of the reload reference exactly, including forced command switches.
See `configs/v22_phase1_reuse_comparison_v1.json` and `docs/v22_phase1_reuse_audit.md`;
the earlier failure artifact remains unchanged. One parse: 3.75s; reused steps: 0.047–0.103s,
~51.8x stepping speedup excluding initialization. Independent replicas with the same
executed prefix and different unexecuted future plans also matched exactly.
This is a bounded fixed-weather adapter test, NOT completion of Phase 1. Next implement
lifecycle/resource accounting and full event/permission integration, then bounded dynamic
checks. No long-season runs, API calls or data downloads were performed.

New Phase 0 is COMPLETE as a development task contract, not a dynamic validation.
See `docs/v22_task_contract_v0.md`, `configs/v22_task_contract_v0.json` and the
logical example `configs/v22_campaign_example_v0.json`.
Four 96 m² compartments, 180-day crops, 365-day campaigns, 2-day cleanup;
six management variables; non-limiting root zone; fixed synthetic cost scenario.
All unsupported geometry/boundary/economic values are explicitly model assumptions.
39 focused tests passed. Run `scripts/verify_v22_task_contract.py` for the logical
schedule/unit check; it does NOT simulate harvest or energy.
Phase 0 was completed and handed over; the user subsequently authorized Phase 1 (current status above).
Its gates include physical response, scaling/boundary sensitivity, controller causality,
idle/cleanup/replant continuity, Full/Endpoint permissions and runtime. No API calls,
new data downloads or numerical greenhouse runs in this phase-completion step.
Version any necessary Phase 1 revisions before pilot/test; no silent parameter changes.
The main runtime remains TOMGRO until the new executor is implemented.

The authoritative plan is now `V2_2_EXPERIMENT_REDESIGN_TODO.md`, rewritten around
three studies: end-to-end experimental performance, Full versus Endpoint feedback,
and fixed-history reader replacement. Its new Phase 0–7 numbering replaces the previous
planning sequence. The old four-tool-arm matrix runner must not launch the new study.
Historical Phase 0 completion below refers only to the earlier evidence-ledger work;
it does not certify the new environment or experiments. Historical validation artifacts
and their failed gate remain unchanged.

The existing keyword firewall is a prototype, not proof of information isolation.
Public-state-only tool interfaces, independent tool randomness, numerical noninterference
tests, hard-stop handling and persisted outbound payload audits remain required by the
new plan. A model may be told it is operating in simulation; the generating mechanism,
private site parameters and future outcomes must remain inaccessible.

## Branch and latest state

Worktree: `/Users/dingningpei/Desktop/slowlab-v2-observation`  
Branch: `codex/v2.1`

Phase 0 is complete. It reached the pre-registered temporal holdout and the frozen v8 hybrid model failed the real-trajectory gate. Do not refit v8 or reinterpret this holdout as development data while retaining a validation claim. Version 2.2 proceeds as a reality-constrained simulation benchmark under `configs/reality_constraints_v2_2.json`.

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
- Version 2.2 evidence and support contract: `configs/reality_constraints_v2_2.json`
- Range-support auditor: `scripts/audit_reality_support.py`
- Frozen conclusion verifier: `scripts/verify_phase0_reality_contract.py`
- Simulation-blinding policy: `configs/simulation_blinding_v2_2.json`
- Outbound prompt firewall: `slowlab/prompt_firewall.py`

Formal Version 2.2 model calls must use `--blinding-policy configs/simulation_blinding_v2_2.json`.
The firewall blocks benchmark/source identity, simulator names, site/environment seeds,
latent state, oracle values and future outcomes before a provider call. OpenRouter app
attribution headers are disabled by default. Confirmatory sites must be newly generated
from aggregate public-data constraints and pass a source-trajectory similarity audit;
the private seed commitment/reveal step is still pending.

Raw public data and generated replay trajectories are intentionally outside Git under `/private/tmp`. The private `output/reviews` directory must never be pushed.

## Working tree caution

This worktree already contained uncommitted changes in `scripts/run_baselines.py`,
`scripts/run_llm.py`, `slowlab/llm.py`, `slowlab/providers.py`, `tests/test_llm.py`,
`tests/test_providers.py`, and untracked `scripts/run_v22_callable_matrix.py`. The v2.2
blinding implementation now intentionally adds changes to `scripts/run_llm.py`,
`slowlab/providers.py`, `tests/test_providers.py`, and the draft matrix runner. Review the
complete diffs and stage by hunk before committing so earlier unrelated work is preserved.

## Execution notes

The first holdout attempt exposed a runner-only candidate JSON wrapper bug; the fix is commit `afc286b`. Some workers had read initial sequence states, but no trajectory metrics were produced. A later command typo also stopped before observation access. Neither event changed the frozen model, identities, features, lambda, physical parameters, or thresholds. The completed retry followed the same frozen contract.
