"""API spend accounting and the worst-case cost bound used by the pilot's hard cap.

The billed cost of a call is OpenRouter's reported ``usage.cost`` when present,
otherwise tokens times the frozen price table. The worst case of one branched
campaign follows from the harness limits alone: at most twice the per-branch
prompt-character budget (the shared prefix counts once in each branch), at
most twice ``max_llm_calls`` model calls (plus the initial recommendation's
attempts), each producing at most ``max_tokens`` visible tokens plus a capped
reasoning allowance, and characters converted to tokens at a conservative
three characters per token.
"""
from __future__ import annotations

CHARS_PER_TOKEN_CONSERVATIVE = 3.0
REASONING_ALLOWANCE_TOKENS = 1024


def call_cost(record: dict, price: dict) -> float:
    usage = record.get('usage') or {}
    if isinstance(usage.get('cost'), (int, float)):
        return float(usage['cost'])
    prompt = usage.get('prompt_tokens') or 0
    completion = usage.get('completion_tokens') or 0
    return (prompt * price['input'] + completion * price['output']) / 1e6


def records_cost(records: list, price: dict) -> float:
    return sum(call_cost(r, price) for r in records)


def worst_case_campaign_cost(price: dict, *, max_input_chars_per_branch: int, max_llm_calls_per_branch: int,
                             max_tokens: int, initial_attempts: int = 3, initial_prompt_chars: int = 20_000) -> float:
    input_chars = 2 * max_input_chars_per_branch + initial_attempts * initial_prompt_chars
    calls = 2 * max_llm_calls_per_branch + initial_attempts
    input_tokens = input_chars / CHARS_PER_TOKEN_CONSERVATIVE
    output_tokens = calls * (max_tokens + REASONING_ALLOWANCE_TOKENS)
    return (input_tokens * price['input'] + output_tokens * price['output']) / 1e6
