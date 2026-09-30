# SlowLab

**A benchmark for experimental-design agents when experiments are slow, noisy, costly and
irreversible.**

An agent runs a 365-day greenhouse campaign over four independently controlled 96 m²
compartments: it starts crops under a management policy, inspects measurements as they
arrive, stops or replants, and finally recommends a policy that is scored in an isolated
evaluation the agent never sees. Ground truth is the pinned GreenLight climate–crop model
forced by archived real weather. It is a reality-constrained simulation, not a validated
digital twin; the failed external-validity audit is reported, not hidden.

The research plan is [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md); the task list is
[`TODO.md`](TODO.md). These two files are the only planning documents.

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
configs/           frozen inputs: task contracts v0–v3, campaign example, model family,
                   reality constraints, blinding policy, weather plans and protocols
configs/agc/       frozen AGC contracts, protocols and recorded results
results/           frozen pilot results, structural audits and daily progress traces
docs/              living protocol and decision documents (14 files)
tests/, tests/archive/
legacy/v2.1/       the frozen Version 2.1 study: TOMGRO environment, scripts, results, tests, paper
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

See [`CITATION.cff`](CITATION.cff) and [`LICENSE`](LICENSE).
