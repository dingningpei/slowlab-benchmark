# Running the LLM evaluation

## One-time setup

```bash
pip install -r requirements.txt
python3 -m pytest -q          # expect 92 passed
```

`.env` (next to this file; already excluded by .gitignore):

```
OPENROUTER_API_KEY=sk-or-...
```

A model name containing `/` is routed through OpenRouter; one without it goes to the
vendor directly (`DEEPSEEK_API_KEY` and friends).

## Smoke test

```bash
python3 scripts/run_llm.py --model z-ai/glm-5.3-flash --tasks Optimise --seeds 1 --fresh
```

One line per episode reading `regret ... (zero-shot ...) ... round 3/3 format-failures 0
infeasible 0` means the loop is working.

## Full run (four tasks x 20 sites per model, roughly 400 calls)

```bash
for M in z-ai/glm-5.3-flash deepseek/deepseek-v4-flash xiaomi/mimo-v2.5 \
         openai/gpt-5.6-luna qwen/qwen3.8-27b; do
  python3 scripts/run_llm.py --model "$M" \
      --tasks Sanity Screen Optimise Transfer --seeds 20 --fresh
done
```

Or `bash scripts/run_all_llm.sh`, which does the same for all five models with and
without tools.

Results are written per episode to `results/llm_env<version>/`. Re-running after an
interruption skips the episodes that finished; `--fresh` forces them to be redone.

## Tool ablation

The same command with `--tools` adds two things to the prompt: a ready-made
Plackett–Burman allocation table, and the posterior maximum of a GP fitted to whatever
observations the agent has collected. Results go to `summary_<model>+tools.json` and do
not overwrite the bare arm.

```bash
python3 scripts/run_llm.py --model z-ai/glm-5.3-flash \
    --tasks Sanity Screen Optimise Transfer --seeds 20 --tools
```

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
* **The environment is frozen at v1.0.0** (see `ENVIRONMENT_v1.0.md`). Do not change it
  part-way through a run, or the models stop being comparable — `tests/test_frozen.py`
  will fail if ground truth moves.
