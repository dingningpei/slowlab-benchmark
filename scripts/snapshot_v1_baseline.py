#!/usr/bin/env python3
"""Create a private, machine-readable snapshot of the frozen v1 baseline."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pathlib
import subprocess
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import slowlab


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def tree_digest(paths: list[pathlib.Path]) -> str:
    value = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        value.update(path.name.encode())
        value.update(bytes.fromhex(digest(path)))
    return value.hexdigest()


def git(*args: str, cwd: pathlib.Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True).strip()


def committed_digest(commit: str, path: pathlib.Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    content = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    return hashlib.sha256(content).hexdigest()


def claims(results: pathlib.Path) -> dict:
    achievable_path = results / "achievable_cv_env1.0.0_M320.json"
    achievable = json.loads(achievable_path.read_text())
    episodes_dir = results / "llm_env1.0.0"
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(episodes_dir.glob("episodes_*.json")):
        arm = path.stem.removeprefix("episodes_")
        for episode in json.loads(path.read_text()):
            grouped[(arm, episode["task"])].append(episode)
    episode_metrics = []
    for (arm, task), items in sorted(grouped.items()):
        episode_metrics.append({
            "arm": arm,
            "task": task,
            "n": len(items),
            "mean_regret": sum(float(x["regret"]) for x in items) / len(items),
            "mean_zero_shot_regret": sum(
                float(x["regret_zero_shot"]) for x in items) / len(items),
            "mean_cumulative_regret": sum(
                float(x["cumulative_regret"]) for x in items) / len(items),
        })
    design_metrics = []
    for task, slot in achievable.items():
        for arm, item in sorted(slot["agents"].items()):
            if arm.startswith("ref:"):
                continue
            design_metrics.append({
                "arm": arm,
                "task": task,
                "n_designs": int(item["n"]),
                "R_star": float(item["R_star"]),
                "c": 1 - float(item["R_star"]) / float(slot["prior"]),
            })
    return {
        "environment_version": slowlab.ENV_VERSION,
        "sources": {str(achievable_path): digest(achievable_path)},
        "episode_metrics": episode_metrics,
        "design_metrics_transcript_only": design_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--source-worktree", required=True, type=pathlib.Path)
    args = parser.parse_args()
    llm = args.results / "llm_env1.0.0"
    key_files = [
        ROOT / "slowlab/env.py", ROOT / "slowlab/tasks.py",
        ROOT / "slowlab/llm.py", ROOT / "slowlab/eig.py",
        ROOT / "slowlab/achievable.py", ROOT / "scripts/rescore_cv.py",
        ROOT / "scripts/make_llm_tables.py", ROOT / "paper/main.tex",
    ]
    versions = {}
    for package in ("numpy", "scipy", "pytest", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    commit = git("rev-parse", "HEAD")
    source_diff = git("diff", "--binary", cwd=args.source_worktree)
    snapshot = {
        "git_commit": commit,
        "source_worktree": str(args.source_worktree.resolve()),
        "source_worktree_status": git(
            "status", "--short", cwd=args.source_worktree).splitlines(),
        "source_worktree_diff_sha256": hashlib.sha256(
            source_diff.encode()).hexdigest(),
        "implementation_worktree_status": git(
            "status", "--short").splitlines(),
        "environment_version": slowlab.ENV_VERSION,
        "python": sys.version,
        "dependencies": versions,
        "files_at_commit": {
            str(path.relative_to(ROOT)): committed_digest(commit, path)
            for path in key_files
        },
        "raw_transcripts": {
            "count": len(list(llm.glob("transcript_*.json"))),
            "tree_sha256": tree_digest(list(llm.glob("transcript_*.json"))),
        },
        "episode_summaries": {
            "count": len(list(llm.glob("episodes_*.json"))),
            "tree_sha256": tree_digest(list(llm.glob("episodes_*.json"))),
        },
        "immutability_rule": "Do not overwrite artifacts bearing env1.0.0.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "v1.0_baseline.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False))
    (args.out / "v1.0_claim_snapshot.json").write_text(
        json.dumps(claims(args.results), indent=2, ensure_ascii=False,
                   allow_nan=False))
    print(json.dumps({
        "commit": snapshot["git_commit"],
        "transcripts": snapshot["raw_transcripts"]["count"],
        "episode_files": snapshot["episode_summaries"]["count"],
    }, indent=2))


if __name__ == "__main__":
    main()
