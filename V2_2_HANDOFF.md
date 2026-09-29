# SlowLab Version 2.2 — status

Last restructured 2026-09-29. Branch `codex/v2.1`, worktree `worktrees/slowlab-v2-observation`.
The plan of record is `V2_2_EXPERIMENT_REDESIGN_TODO.md` (three experiments, Phase 0–7);
edits to it beyond checkbox flips need the user's sign-off (`CLAUDE.md`). Repository layout
is described in `README.md`. This file records **what is frozen, what has passed, what is
open and what was abandoned**, so that a new agent does not re-derive or re-litigate it.
Earlier chronological handoff notes are in the Git history of this file before `4fb4c93`.

Nothing here is an LLM result or a physical-greenhouse validation. No formal Version 2.2
model call has been made; no formal sites or seeds exist; nothing has been pushed.

---

## 1. Frozen

Frozen means byte-identical and hash-locked. Do not edit; add a new version instead.
Records under `configs/v22/`, `configs/agc/` and `results/v22/` may name files by the
pre-carve-up flat paths; `slowlab.v22.frozen_paths.frozen_path` resolves them.

| Item | File | Status |
| --- | --- | --- |
| Research object and three experiments | `V2_2_EXPERIMENT_REDESIGN_TODO.md` §1–9 | plan of record |
| Task contract v3 (four 96 m² compartments, 180-day crops, 365-day campaign, 2-day cleanup, six policy fields, adequately sized central supply, independent per-compartment caps) | `configs/v22/task_contract_v3.json`; v0–v2 kept with `supersedes` chain | development contract, not a preregistration |
| Campaign example (logical schedule) | `configs/v22/campaign_example_v0.json` | verified by `scripts/v22/verify_task_contract.py` |
| Model family: GreenLight `fa502eddae5f9eff7b3380c88037d9b5f3f14bf5`, `main_katzin_2021.json`, four source-file hashes | `configs/v22/greenlight_model_family.json` | static selection; native coefficients are literature assumptions, not AGC-calibrated |
| Reality-constraint ledger (evidence classes, supported ranges, unsupported actuator fluxes) | `configs/v22/reality_constraints_v2_2.json` + `.sha256`; `docs/v22/reality_constraints_v2_2.md` | verified by `scripts/v22/verify_phase0_reality_contract.py` |
| Simulation-blinding policy | `configs/v22/simulation_blinding_v2_2.json` + `.sha256`; `docs/v22/simulation_blinding_v2_2.md`; `slowlab/v22/prompt_firewall.py` | keyword firewall is a prototype, not proof of isolation |
| 14-component evidence register (observed / literature-constrained / assumed / unsupported) | `configs/v22/component_evidence_v0.json` | verified by `scripts/v22/verify_component_evidence.py` |
| Source-weather partition v0 (2017–2018 fit, 2019 selection, 2020 out-of-year diagnostic) | `configs/v22/weather_partition_v0.json` | development record only; 2020 is no longer untouched |
| Historical-weather route: real Cabauw years as external forcing, shared weather year across arms, CET fixed clock, leap-year rule, new holdout years unsealed only after code/site/metric freeze | `docs/v22/historical_weather_boundary_protocol.md` | route decision, not a formal weather set |
| Standby policy (10 °C protective heating, 30 °C ventilation, lamps and CO₂ off) and sampled thermal screen | contract v1/v2 derivations; `docs/v22/phase1_standby.md`, `docs/v22/phase1_sampled_screen.md` | |
| Deep-soil boundary: explicit `tSoOut` scenario input; default remains native 20 °C | `docs/v22/deep_soil_boundary_audit.md` | choice of default still open (§3) |
| Error-attribution rules for four failure classes | `docs/v22/phase1_error_attribution.md` | protocol only |
| AGC GreenLight v8 temporal holdout: FAILED (see §4) | `configs/agc/agc2019_holdout_result_v8.json` | final; do not refit |

Data outside Git (never commit): Cabauw lb1 (98 files, 21,770,259 B) and lc1 gap-filled
(98 files, 18,413,090 B) 2016-12–2020 under `/private/tmp`; 2013/2014/2016 development
years (13,027,903 B net new); AGC raw data and replay trajectories. Every acquisition was
individually approved by the user and is hash-audited (`results/v22/weather_quality_audit.json`,
`results/v22/weather_gapfilled_quality_audit.json`, `results/v22/weather_v3_development_full_audit.json`).
2015 and 2025 have not been downloaded. Any further download needs prior approval of the
exact size and scope.

---

## 2. Passed (Phase 1 engineering gates)

All of these are scripted integration or structural checks on a pinned runtime. None is an
LLM experiment, a formal site, or evidence that GreenLight reproduces a real greenhouse.

**Executor and model reuse**

- `ReusableGreenLight` parses once and registers all command channels before compilation;
  six 300 s steps match the reload reference on all 28 states, including forced command
  switches; independent replicas with the same executed prefix and different unexecuted
  futures match exactly. Parse 3.75 s; reused step 0.047–0.103 s (~51.8× speedup).
  `results/v22/phase1_reuse_comparison_v1.json`, `docs/v22/phase1_reuse_audit.md`.
- Optional compiled RHS (`slowlab/v22/native_rhs.py`) and cached solver: derivative and
  600-step trajectory regressions pass. A day-240.76 overflow (intermediate `exp` in a
  condensation sigmoid) was diagnosed and replaced by a stable algebraic form; exact-state
  replay and native/reference regression pass. `docs/v22/native_overflow_fault.md`,
  `results/v22/native_fault_fix_replay.json`.
- Explicit empty-compartment mode (crop processes removed, zero-LAI radiation retained);
  cleanup/idle/replant continuity on fixed weather. `docs/v22/phase1_empty_crop_gate.md`.
- Causal observation stores: 10-minute exterior means released only at their right
  endpoint; 300 s interior endpoints released after solving; public and controller
  channels separate. Slotted immutable records cut a 200k-record sample from ~109 MB to
  ~66 MB. `docs/v22/online_weather_bridge.md`, `docs/v22/observation_causality_audit.md`.
- Four units on one clock: controller reads only arrived measurements; requested = realised
  under the adequately-sized-supply contract; per-unit heat/CO₂/lamp fluxes at but not over
  their maxima; Full/Endpoint permissions; fixed-CET origin bug fixed.
  `docs/v22/phase1_four_unit_weather_and_runtime.md`.

**Annual runs (Tokyo host, 2 vCPU / 1 physical core / ~896 MB, cgroup-capped)**

| Run | Result | Pointer |
| --- | --- | --- |
| Single compartment, 365 days, 180+2+180+2+1 lifecycle | passed; 2,119 s, peak RSS 271 MB (350 MB cap). Two earlier attempts failed on a virtual-address cap (day 59) and host OOM (day 290). | `results/v22/remote_annual_pilot_v3_result.json`, `docs/v22/remote_compute_preflight.md` |
| Four units, fixed policy, 365 days | passed; 105,120 steps, 8,343 s, peak RSS 252 MB; structural audit passed | `results/v22/remote_four_unit_annual_stable_result.json`, `_audit.json` |
| Four units, **event-driven** `CampaignExecutor`, 365 days, 5 plantings | passed; 7,925 s, peak RSS 261 MB. Day-14 read of 289 arrived temperature records (mean 21.705 °C) chose the next-round policy by a pre-scripted 22 °C rule; unit 0 stopped, cleaned two days, replanted day 16; units 2/3 started days 14/30; unit 3 stopped early on day 60 and settled only realised resources; remaining crops completed days 180/194/196; day-365 recommendation without fallback. Independent streaming audit of the 63 MB private tick trace passed (time continuity, requested = realised, causal releases, ledger consistency, hashes). | `results/v22/dynamic_annual_result.json`, `dynamic_annual_audit.json`, `docs/v22/phase1_event_trace.md` |
| Deep-soil 8 °C vs 20 °C, single compartment, 2017 weather, paired | passed structural audit; 8 °C uses 7.49 % more heat and yields 1.25 % more simulated harvest | `results/v22/soil_annual_sensitivity_audit.json`, `docs/v22/deep_soil_boundary_audit.md` |
| 2014 expanded weather reader (independent product selection, 105,121 boundary queries, zero difference against a hash-locked five-channel matrix) | passed; no greenhouse dynamics run on 2014 | `results/v22/expanded_weather_2014_boundary_scan.json` |

**Cross-platform note.** Mac versus Linux closed-loop seven-day runs differ by up to
1.59e-4 relative in final state and 0.175 % in heat ledger because feedback amplifies
platform rounding (77/2,016 steps had a command differing by >0.01); replaying the Mac
command trace on Linux agrees to 2.33e-11. Pin one runtime for every formal comparison.

**Tests.** `python3 -m pytest -q`: 378 passed, 4 skipped (2026-09-29). GreenLight is not
importable in the development environment; scripts that need it run only on the pinned host
with `--source <GreenLight checkout>`.

---

## 3. Open (ordered by how hard each blocks a submission)

1. **No site distribution.** All runs use identical native GreenLight parameters, one
   weather year and one price scenario. Experiment 1 needs N private sites drawn from a
   frozen generator with development/pilot/test partition and seed commitment (Phase 3).
2. **No agent interface, BO baseline, pre-experiment recommendation, fixed reference or
   evaluator** (Phase 2–3). `CampaignExecutor.dispatch` is a JSON action surface only;
   process isolation, the public tool set with independent randomness, the
   "same public history + different private seed → identical prompt" test, hard-stop on
   firewall block and outbound payload persistence are all unimplemented. The old TOMGRO
   harness `slowlab/llm.py` and `scripts/v22/run_callable_matrix.py` must not launch the
   new study.
3. **Compute budget.** ~2.2 worker-hours per four-unit annual campaign on the Tokyo host;
   the 1,024-campaign draft is ~2,400 worker-hours before LLM calls, evaluation and
   retries. Formal N, tick size, parallel workers and host are undecided. Do not run the
   matrix on the MacBook.
4. **Sensor noise is zero.** `slowlab/v22/sensor_bridge.py` records deterministic virtual
   sensors. Freeze a measurement-noise and missingness model independent of site truth,
   then implement sample-once caching per (compartment, channel, time).
5. **Executor generality.** `CampaignExecutor` hard-codes four units and dict policies;
   simultaneous four-unit memory on other hosts is untested.
6. **Deep-soil default.** Annual 8/20 °C sensitivity passed; which boundary the main
   scenario uses, and how it is disclosed, is not frozen.
7. **Range audit refinement.** `scripts/v22/audit_reality_support.py` must distinguish
   physical error, source extrapolation and declared assumption instead of dropping sites.
8. **Blinding and compute feasibility are unverified for varying sites and weather years.**
9. Phase 4–7: fake-model pilot, exact model IDs and routing, cost accounting, preregistration
   lock, formal runs, Experiment 3 packets, paper rewrite.

---

## 4. Abandoned (report only in the appendix)

| Direction | Outcome | Evidence kept |
| --- | --- | --- |
| AGC 2019 GreenLight calibration v0–v8 with observed-action temporal holdout | **Failed the trajectory gate**: 8 sequences, 137 compartment-days, 3,288 hourly samples; solver completion 100 %, 0 sampled physical violations; pooled temperature RMSE 2.579 °C (limit 2.04), RH RMSE 9.983 pp (limit 8.5), CO₂ 211.8 ppm (limit 361, pass). Heterogeneous failures (cold bias in Automatoes/Digilog, humidity in AICU/TheAutomators). Public AGC files do not identify enough boundary conditions for post-hoc corrections. Do not refit v8 or reinterpret the holdout as development data. | `configs/agc/`, `scripts/agc/`, `docs/agc/agc2019_greenlight_feasibility.md`, `slowlab/archive/` |
| Private weather generator candidate v2 | failed locked 2020 cross-year diagnostic (monthly mean 1.631 → 1.672 training SD after the leap-month erratum, limit 1.25; wind daily-variance ratio 0.451, limit 0.50) | `configs/v22/weather_generator_2020_diagnostic_lock.json`, `results/v22/weather_candidate_v2_2020_*.json`, `scripts/v22/verify_weather_2020_diagnostic.py` |
| Block generators v3a / v3b / v3c | v3a 48/48 near-replay rejections; v3b partial non-replay pass but weak 2020 coverage; v3c seven-fold pilot 84/84 physical support, 75/84 non-replay, but worse energy score than resampling a real year in all seven folds | `results/v22/weather_v3a_development_pilot.json`, `weather_v3b_development_pilot.json`, `weather_v3c_sevenfold_development_pilot.json`, `docs/v22/weather_v3c_development_result.md` |
| Generator code | deleted 2026-09-29 (`49011fe`); the whole-year source reader survives as `slowlab.v22.cabauw_weather.read_source_year` | Git history |
| Cabauw 0–50 cm field soil as greenhouse deep-soil boundary | not downloaded; not a valid stand-in for ~2 m soil under a greenhouse | `docs/v22/deep_soil_boundary_audit.md` |
| Cross-compartment central-supply contention | not modelled; contract v3 declares adequately sized central supply | `docs/v22/shared_capacity_gap.md` |
| Old four-tool-arm matrix (bare/design/reader/both) as the main experiment | superseded by the three-experiment design | `scripts/v22/run_callable_matrix.py` is a historical draft |

---

## 5. Operating constraints

- Frozen JSON is never edited. Private `output/reviews/` is never pushed. Keys live only in `.env`.
- Every data download and every LLM API spend needs prior, specific user approval.
- Single worker, single-threaded numerics by default; time any long computation first and
  report a budget; never leave the MacBook under sustained load.
- The pinned runtime for formal comparisons is one Linux host and one GreenLight commit;
  record RSS, wall time and hashes for every run, and keep failed runs with their identity.
- A firewall block during a formal call is an infrastructure error, not a model format
  failure; it must abort and be logged, never retried into a result.
