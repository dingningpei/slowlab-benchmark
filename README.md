# SlowLab

**A benchmark for experimental-design agents when experiments are slow, noisy, costly and
irreversible.**

An agent runs a 365-day greenhouse campaign over four independently controlled 96 m²
compartments: it starts crops under a management policy, inspects measurements as they
arrive, stops or replants, and finally recommends a policy that is scored in an isolated
evaluation the agent never sees. Ground truth is the pinned GreenLight climate–crop model
forced by archived real weather. It is a reality-constrained simulation, not a validated
digital twin; the failed external-validity audit is reported, not hidden.

This repository accompanies the paper *SlowLab: Benchmarking Autonomous AI Agents on a Year of
Experiments They Cannot Undo* (Ningpei Ding, 2026; source in [`paper/`](paper/)).
In a preregistered run on 48 unseen sites, three low-cost language models ended clearly below
a prior-informed local Bayesian-optimisation baseline, and in-season process feedback did not
improve the final policy of any method.

The research plan is [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md); the task list is
[`TODO.md`](TODO.md). These two files are the only planning documents.

## Paper and results

| What | Where |
|---|---|
| Paper source (NeurIPS 2026 style, `preprint` option) | [`paper/main.tex`](paper/main.tex), sections in `paper/sections/` |
| E1/E2 primary and secondary tests | [`results/repaired_formal_analysis_e1e2_20261006.json`](results/repaired_formal_analysis_e1e2_20261006.json) |
| E3 Readers, boundary and dry-matter sensitivity, headroom | [`results/repaired_phase6_analysis_20261006.json`](results/repaired_phase6_analysis_20261006.json) |
| How the run was completed in two parts (field-order error and correction) | [`results/repair_run_record_20261006.json`](results/repair_run_record_20261006.json), [`results/repaired_audit_20261006.json`](results/repaired_audit_20261006.json), paper Appendix D |
| Results of the original first part, for comparison | [`results/formal_analysis_e1e2_20261005.json`](results/formal_analysis_e1e2_20261005.json), [`results/phase6_analysis_original_server2_20261006.json`](results/phase6_analysis_original_server2_20261006.json) |
| Locks (code, configuration, prompts, models, seed commitments, environment) | `configs/formal_lock_v1.json` … `configs/formal_lock_v8.json` (v5 superseded before use) |
| Decisions and deviations, dated | the decision table in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) |

Every number in the paper is generated from these files and checked:

```bash
python3 scripts/make_paper_numbers.py        # writes paper/generated/numbers.tex
python3 scripts/verify_results_claims.py     # no stale or hand-written result numbers
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Figure 4 is drawn from the results file alone (`scripts/make_figure4.py`); Figures 1–3 also
need the campaign and evaluation records of the data package below.

## Data

The public campaign records, evaluation records and E3 history packets will be released as a
data package under CC BY 4.0. The test-site and E3 sample master seeds stay sealed, so that the
48 test sites remain unseen by future agents; their salted commitments are public
(`configs/seed_commitment_test_v1.json`, `configs/seed_commitment_e3_sample_v1.json`).
The Cabauw weather data come from the KNMI Data Platform (datasets
`cesar_surface_meteo_lc1_t10` and `cesar_surface_radiation_lc1_t10`, version 1.0, CC BY 4.0)
and are not redistributed here; `scripts/download_cabauw_weather.py` and
`scripts/download_weather_formal.py` fetch and audit them.

## Layout

```
RESEARCH_PLAN.md   the single research plan (research question, three experiments, protocol, paper map, status)
TODO.md            the single task list, by phase
CLAUDE.md          rules for agents editing this repo
slowlab/           the live package: task contract, campaign executor, controller, resources,
                   feedback views, causal observation stores, sensor bridge, GreenLight adapters,
                   Cabauw weather reader, prompt firewall, reality constraints, frozen-path resolver
slowlab/archive/   retired AGC 2019 calibration modules and root-zone modules (appendix only)
scripts/           checks, audits, annual pilots and verifiers for the live environment
scripts/agc/       retired AGC 2019/2023/2024 scripts
configs/           frozen inputs: task contracts v0–v8 (v8 is the formal one), site distribution,
                   models, baseline configs, seed commitments, locks, blinding policy, weather plans
configs/agc/       frozen AGC contracts, protocols and recorded results
results/           frozen results: audits, pilots, the formal run, its correction and the analyses
docs/              protocol, decision and data-provenance documents (12 files)
tests/, tests/archive/
legacy/v2.1/       the frozen Version 2.1 study: TOMGRO environment, scripts, results, tests, paper
paper/             the v2.3 paper; numbers in paper/generated/ come from results/
```

Frozen JSON under `configs/`, `configs/agc/` and `results/` is hash-locked and never edited.
Some records name files by older paths; `slowlab.frozen_paths.frozen_path` resolves them.
Lab-notebook documents from Phase 0–1 were deleted on 2026-09-29; `git show 21623bf:docs/v22/<name>`
recovers any of them.

## Install and check

```bash
pip install -r requirements.txt
python3 -m pytest -q                       # live environment tests
python3 scripts/verify_task_contract.py    # logical schedule / unit check of the contract
python3 scripts/verify_component_evidence.py
python3 scripts/verify_phase0_reality_contract.py
```

Runs that step the physical model need GreenLight at commit
`fa502eddae5f9eff7b3380c88037d9b5f3f14bf5` and the audited Cabauw weather cache (kept outside
Git). Either install the pinned engine and numerical stack:

```bash
pip install -e '.[greenlight]'
```

or pass a checkout at that commit with `--source`. Both routes verify the four model
definition hashes before loading. See `configs/greenlight_model_family.json` and
`docs/historical_weather_boundary_protocol.md`.

## The frozen Version 2.1 study

`legacy/v2.1/` contains the earlier synchronous-round benchmark (reduced TOMGRO, three
tasks, 928 + 384 + 288 LLM episodes) and its paper. It is self-contained and reproducible from
inside that directory:

```bash
cd legacy/v2.1 && python3 -m pytest -q
```

Its README, `RUNNING.md` and `ENVIRONMENT_v2.0.md` describe it. It is not imported by the
live package and is cited in the new paper only as a pilot study.

## Citation and licence

Code: MIT ([`LICENSE`](LICENSE)). Data package: CC BY 4.0. Weather data: KNMI, CC BY 4.0.
See [`CITATION.cff`](CITATION.cff).
