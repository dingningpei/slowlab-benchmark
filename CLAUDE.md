# Instructions for coding agents working in this repo

## Editing `V2_2_EXPERIMENT_REDESIGN_TODO.md`

This is the shared task list for the Version 2.2 push (Phase 0–7). It is the working
source of truth for what is left to do.

- **Needs the user's sign-off before writing:** anything beyond flipping a checkbox —
  rewording an item, rewriting a status note, adding a new sub-item, recording a
  decision, restructuring a section. Propose the change in chat and wait for a
  go-ahead before writing it, not just before committing.
- **Does not need to ask first:** checking a `- [ ]` off to `- [x]` for an item the
  agent itself just finished in this session.

When unsure which bucket an edit falls into, treat it as needing sign-off.

## Frozen artifacts

JSON files under `configs/v22/`, `configs/agc/` and `results/v22/` are frozen, hash-locked
records. Never edit their contents; add a new versioned file instead. Several of them
name other files by the pre-2026-09-29 flat paths; resolve those with
`slowlab.v22.frozen_paths.frozen_path` rather than rewriting the record.

## Layout

The flat `slowlab/`, `scripts/`, `configs/`, `results/`, `docs/`, `tests/` trees are the
frozen Version 2.1 stack and must not change. Version 2.2 work goes under the `v22/`
sub-trees; retired AGC material lives under `archive/` and `agc/`. The current state
summary is `V2_2_HANDOFF.md`.
