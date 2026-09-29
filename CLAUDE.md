# Instructions for coding agents working in this repo

## Planning documents

`RESEARCH_PLAN.md` is the single research plan; `TODO.md` is the single task list. There
are no other planning or handoff documents, and none should be created. Progress notes go
into `TODO.md` under the item they concern, not into new files.

- **Needs the user's sign-off before writing:** anything in `RESEARCH_PLAN.md`, and anything
  in `TODO.md` beyond flipping a checkbox — rewording an item, rewriting a status note,
  adding a sub-item, recording a decision, restructuring a section. Propose the change in
  chat and wait for a go-ahead before writing it, not just before committing.
- **Does not need to ask first:** checking a `- [ ]` off to `- [x]` for an item the agent
  itself just finished in this session, with a one-line note of the evidence file.

When unsure which bucket an edit falls into, treat it as needing sign-off.

## Frozen artifacts

JSON files under `configs/`, `configs/agc/` and `results/` are frozen, hash-locked records.
Never edit their contents; add a new versioned file instead. Several name other files by
older paths; resolve those with `slowlab.frozen_paths.frozen_path` rather than rewriting
the record.

## Legacy

`legacy/v2.1/` is the frozen Version 2.1 study (environment, scripts, results, paper). It
is not imported by the live package and must not change. Run its tests from inside that
directory: `cd legacy/v2.1 && python3 -m pytest -q`.

## Operating rules

- Every data download and every LLM API spend needs prior, specific user approval (size,
  scope, model, budget).
- Do not run long simulations on the MacBook; time a short run first and report a budget.
- Pin one Linux host and one GreenLight commit for every formal comparison; record wall
  time, peak RSS and hashes; keep failed runs with their identity.
- Never push `output/reviews/`, `.env`, raw datasets or private traces.
