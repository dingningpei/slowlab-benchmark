#!/usr/bin/env python3
"""Check that every result number in the paper traces to the analysis outputs.

    verify_results_claims.py

1. The generated numbers (paper/generated/numbers.json) equal a fresh regeneration from the
   result files, so no number is stale.
2. Every \\res{key} used in paper/sections/*.tex exists.
3. The abstract and results sections print no hard-coded decimal number: results come only
   through \\res{...}.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from make_paper_numbers import numbers  # noqa: E402

RESULT_FILES = ('00_abstract.tex', '05_results.tex', '06_e3.tex')


def main():
    problems = []
    stored = json.loads((ROOT / 'paper/generated/numbers.json').read_text())['numbers']
    fresh = numbers()
    if stored != fresh:
        changed = sorted(k for k in set(stored) | set(fresh) if stored.get(k) != fresh.get(k))
        problems.append(f'generated numbers are stale: {changed[:10]}')
    used = set()
    for tex in sorted((ROOT / 'paper/sections').glob('*.tex')):
        text = tex.read_text()
        for key in re.findall(r'\\res\{([^}]+)\}', text):
            used.add(key)
            if key not in fresh:
                problems.append(f'{tex.name}: unknown result key {key}')
        if tex.name in RESULT_FILES:
            body = re.sub(r'%.*', '', text)
            body = re.sub(r'\\res\{[^}]+\}', '', body)
            body = re.sub(r'\\(?:todo|label|ref|cite[pt]?|includegraphics(?:\[[^]]*\])?)\{[^}]*\}', '', body)
            for m in re.finditer(r'(?<![\w.-])\d+\.\d+(?![\w.])', body):
                problems.append(f'{tex.name}: hard-coded number {m.group(0)}')
    out = {'numbers_available': len(fresh), 'numbers_used': len(used), 'problems': problems, 'passed': not problems}
    print(json.dumps(out, indent=1))
    raise SystemExit(0 if not problems else 1)


if __name__ == '__main__':
    main()
