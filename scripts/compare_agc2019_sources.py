#!/usr/bin/env python3
"""Compare extracted official AGC 2019 data against an engineering mirror.

No raw data are copied into Git.  The report records byte-level differences and
whether a CSV differs only by CRLF versus LF line endings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inventory(root: Path) -> dict[str, Path]:
    return {str(path.relative_to(root)): path for path in root.rglob("*") if path.is_file()}


def compare_sources(official: Path, mirror: Path) -> dict:
    official_files = inventory(official)
    mirror_files = inventory(mirror)
    report: dict = {
        "official_root": str(official.resolve()),
        "mirror_root": str(mirror.resolve()),
        "official_file_count": len(official_files),
        "mirror_file_count": len(mirror_files),
        "official_only": sorted(official_files.keys() - mirror_files.keys()),
        "mirror_only": sorted(mirror_files.keys() - official_files.keys()),
        "files": {},
    }
    for name in sorted(official_files.keys() & mirror_files.keys()):
        original = official_files[name].read_bytes()
        mirrored = mirror_files[name].read_bytes()
        if original == mirrored:
            verdict = "byte_identical"
        elif name.lower().endswith(".csv") and original.replace(b"\r\n", b"\n") == mirrored:
            verdict = "line_endings_only"
        else:
            verdict = "content_mismatch"
        report["files"][name] = {
            "verdict": verdict,
            "official_sha256": sha256(original),
            "mirror_sha256": sha256(mirrored),
        }
    report["verdict_counts"] = {
        verdict: sum(row["verdict"] == verdict for row in report["files"].values())
        for verdict in ("byte_identical", "line_endings_only", "content_mismatch")
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True, type=Path)
    parser.add_argument("--mirror", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = compare_sources(args.official, args.mirror)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "official_file_count": report["official_file_count"],
        "mirror_file_count": report["mirror_file_count"],
        "official_only": report["official_only"],
        "mirror_only": report["mirror_only"],
        "verdict_counts": report["verdict_counts"],
    }, indent=2))
    if report["verdict_counts"]["content_mismatch"]:
        raise SystemExit("content differences found")


if __name__ == "__main__":
    main()
