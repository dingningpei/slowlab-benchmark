# Running the LLM evaluation

## One-time setup

```bash
pip install -r requirements.txt
python3 -m pytest -q
```

`.env` (next to this file; already excluded by .gitignore):

```
OPENROUTER_API_KEY=sk-or-...
DEEPSEEK_API_KEY=sk-...
```

A model name containing `/` is routed through OpenRouter; one without it goes to the
vendor directly (`DEEPSEEK_API_KEY` and friends).

## Smoke test

```bash
python3 scripts/run_llm.py --model z-ai/glm-5.3-flash --tasks Optimise --seeds 1 --fresh
```

One line per episode reading `regret ... (zero-shot ...) ... round 3/3 format-failures 0
infeasible 0` means the loop is working.

## Core task sweep

The Version 2.1 core suite contains Sanity, Optimise, and Transfer. Screen remains an
experimental configuration and must be requested explicitly.

```bash
for M in z-ai/glm-5.3-flash deepseek/deepseek-v4-flash xiaomi/mimo-v2.5 \
         openai/gpt-5.6-luna qwen/qwen3.8-27b; do
  python3 scripts/run_llm.py --model "$M" \
      --tasks Sanity Optimise Transfer --seeds 20 --fresh
done
```

`scripts/run_all_llm.sh` is the archived Version 1 four-task sweep and is not the Version
2.1 protocol. Frozen Version 2 and 2.1 experiments use their registered runners and configs.
To inspect the experimental Screen configuration without treating it as a benchmark result,
pass `--tasks Screen` and write to a separate output directory.

Results are written per episode to `results/llm_env<version>/`. Re-running after an
interruption skips the episodes that finished; `--fresh` forces them to be redone.

## Tool ablation

The same command with `--tools` adds two things to the prompt: a ready-made
Plackett–Burman allocation table, and the posterior maximum of a GP fitted to whatever
observations the agent has collected. Results go to `summary_<model>+tools.json` and do
not overwrite the bare arm.

```bash
python3 scripts/run_llm.py --model z-ai/glm-5.3-flash \
    --tasks Sanity Optimise Transfer --seeds 20 --tools
```

## Frozen Version 2.1 model extension

The registered A3 protocol is in `configs/phase5_model_task_extension_a3.json`. It runs
bare and inference-aid conditions on the same Optimise sites. Luna and Qwen use OpenRouter;
DeepSeek V4.1 Flash uses the direct endpoint and native JSON mode:

```bash
python3 scripts/run_llm.py --model deepseek-flash --json-mode \
    --tasks Optimise --seeds 24 --fresh
```

Provider-hidden reasoning is disabled for every formal A3 condition. Do not combine an
ad-hoc sweep with the registered matrix or use intermediate effects as a stopping rule.

## Producing the tables

```bash
python3 scripts/make_tables.py       # scans results/llm_env*/summary_*.json
python3 scripts/make_llm_tables.py   # Tables 3-5 into paper/sections/
```

## Known traps

* **`#` is not a comment in zsh** (`interactive_comments` is off by default). Do not put
  a comment on the same line as a command.
* **Reasoning tokens are billed as output** and are off by default
  (`reasoning.enabled=false`). Turning reasoning on multiplies the cost several times.
* **TLS certificates on macOS**: python.org builds of Python do not use the system
  certificate store. `pip install certifi` is enough; the code picks it up automatically.
* **The environment is frozen at v2.0.0** (see `ENVIRONMENT_v2.0.md`). Do not change it
  part-way through a run, or the models stop being comparable — `tests/test_frozen.py`
  will fail if ground truth moves.
