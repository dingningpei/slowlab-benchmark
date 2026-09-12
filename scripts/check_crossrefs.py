#!/usr/bin/env python3
"""Does every label get cited in the text, and does every ref have a label?

Figure 3 was inserted and never cited -- an orphan float, with no point in the text
at which the reader is told to look at it, which a reviewer would flag immediately.
Defects of this kind are invisible to the eye, so make them a check.

    python scripts/check_crossrefs.py
"""
from __future__ import annotations
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEC = ROOT / "paper" / "sections"
MAIN = ROOT / "paper" / "main.tex"
# These prefixes mark numbered entities the text must lead the reader to
MUST_CITE = ("fig:", "tab:", "eq:")


def main():
    files = sorted(SEC.glob("*.tex")) + [MAIN]
    labels, refs = {}, {}
    for f in files:
        txt = f.read_text()
        for m in re.finditer(r"\\label\{([^}]+)\}", txt):
            labels.setdefault(m.group(1), []).append(f.name)
        for m in re.finditer(r"\\(?:eq)?ref\{([^}]+)\}", txt):
            refs.setdefault(m.group(1), []).append(f.name)

    bad = 0
    orphans = [(l, w) for l, w in sorted(labels.items())
               if l.startswith(MUST_CITE) and l not in refs]
    if orphans:
        bad += len(orphans)
        print("Orphans: labelled but never cited")
        for l, w in orphans:
            print(f"  {l:28s} defined in {', '.join(sorted(set(w)))}")

    dangling = [(r, w) for r, w in sorted(refs.items()) if r not in labels]
    if dangling:
        bad += len(dangling)
        print("\nDangling: cited but no such label")
        for r, w in dangling:
            print(f"  {r:28s} cited in {', '.join(sorted(set(w)))}")

    dupes = [(l, w) for l, w in sorted(labels.items()) if len(w) > 1]
    if dupes:
        bad += len(dupes)
        print("\nDuplicate labels")
        for l, w in dupes:
            print(f"  {l:28s} {', '.join(w)}")

    if not bad:
        n = sum(1 for l in labels if l.startswith(MUST_CITE))
        print(f"OK -- all {n} numbered entities are cited, all {len(refs)} citations "
              f"resolve, no duplicates.")
    # Also list the least-cited entities, to catch "cited, but once and far away"
    thin = sorted(((len(refs.get(l, [])), l) for l in labels
                   if l.startswith(MUST_CITE)))[:5]
    print("\nLeast-cited numbered entities:",
          ", ".join(f"{l}({n})" for n, l in thin))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
