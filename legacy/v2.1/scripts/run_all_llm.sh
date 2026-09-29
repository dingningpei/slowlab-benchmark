#!/usr/bin/env bash
# The full LLM evaluation on environment v1.0: five models x four tasks x 20 instances
# x with/without tool.
#
#   bash scripts/run_all_llm.sh              # the real run
#   bash scripts/run_all_llm.sh --smoke      # one episode per model per task, as a smoke test
#
# Resumable: run_llm.py skips episodes that are already finished, so after Ctrl-C, a
# dropped connection, or a rate limit, simply re-run the same command to continue.
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS=20
if [[ "${1:-}" == "--smoke" ]]; then SEEDS=1; fi

# OpenRouter: a model name containing "/" is routed through OpenRouter automatically,
# so one key covers every vendor.
MODELS=(
  "z-ai/glm-5.3-flash"
  "deepseek/deepseek-v4-flash"
  "xiaomi/mimo-v2.5"
  "openai/gpt-5.6-luna"
  "qwen/qwen3.8-27b"
)
TASKS=(Sanity Optimise Screen Transfer)
OUT="results/llm_env1.0.0"          # carries the environment version, so it cannot mix with older results

# Rate limiting: the OpenRouter quota is per account. When running in parallel, set
# --rpm to (total quota / number of parallel runs).
RPM=0

echo "environment version: $(python3 -c 'import slowlab;print(slowlab.ENV_VERSION)')"
echo "output directory: $OUT   $SEEDS episodes per task"
echo

for M in "${MODELS[@]}"; do
  for TOOLS in "" "--tools"; do
    echo "=== $M ${TOOLS:-(no tool)} ==="
    python3 scripts/run_llm.py \
      --model "$M" --tasks "${TASKS[@]}" --seeds "$SEEDS" \
      --out "$OUT" --rpm "$RPM" --redo-incomplete $TOOLS
    echo
  done
done

echo "all done. Next:"
echo "  python3 scripts/run_achievable.py --M 80 --budget 600   # compute R*(D) for the LLM designs"
echo "  python3 scripts/make_tables.py                          # generate the paper's tables"
