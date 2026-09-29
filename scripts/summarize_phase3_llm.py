#!/usr/bin/env python3
"""Summarize the private Phase 3 LLM mechanism pilot without copying transcripts."""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def mean_se(values):
    values = np.asarray(values, float)
    return {"mean": float(values.mean()),
            "se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0}


def tool_summary(folder):
    report = {}
    for path in sorted(folder.glob("episodes_*.json")):
        rows = json.loads(path.read_text())
        mode = rows[0].get("tool_mode", "bare")
        records = [record for row in rows for record in row.get("tool_use_records", [])]
        item = {
            "episodes": len(rows), "completed": sum(row["rounds_submitted"] == 3 for row in rows),
            "regret": mean_se([row["regret"] for row in rows]),
            "mean_infeasible": float(np.mean([row["infeasible"] for row in rows])),
            "delivery": "none" if mode == "bare" else "passive_prompt_output",
            "model_invocations": sum(record.get("invoked_by_model") is True for record in records),
        }
        if mode in {"design", "both"}:
            valid = [record for record in records if record.get("design_distance") is not None]
            item["design_adoption"] = {
                "adopted": sum(record["design_adopted"] is True for record in valid),
                "eligible": len(valid),
                "mean_geometric_distance": float(np.mean([record["design_distance"] for record in valid]))
                if valid else None,
            }
        if mode in {"inference", "both"}:
            valid = [record for record in records if record.get("recommendation_distance") is not None]
            item["inference_adoption"] = {
                "adopted": sum(record["inference_adopted"] is True for record in valid),
                "eligible": len(valid),
                "mean_normalized_distance": float(np.mean([
                    record["recommendation_distance"] for record in valid])) if valid else None,
            }
        report[mode] = item
    return report


def within_cycle_summary(folder):
    episode_file = next(folder.glob("episodes_*.json"))
    rows = json.loads(episode_file.read_text())
    actions, modalities = [], {}
    for path in sorted(folder.glob("transcript_*.json")):
        for turn in json.loads(path.read_text()):
            try:
                blob = json.loads(turn["assistant"])
            except Exception:
                continue
            if "action" not in blob:
                continue
            actions.append({"action": blob.get("action"), "day": blob.get("day")})
            for modality in blob.get("modalities") or []:
                modalities[modality] = modalities.get(modality, 0) + 1
    days = [action["day"] for action in actions
            if action["action"] == "observe" and action["day"] is not None]
    return {"episodes": len(rows), "completed": sum(row["rounds_submitted"] == 3 for row in rows),
            "regret": mean_se([row["regret"] for row in rows]),
            "observation_actions": len(days), "observation_days": days,
            "modality_counts": modalities,
            "finish_actions": sum(action["action"] == "finish" for action in actions)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-dir", type=pathlib.Path, required=True)
    parser.add_argument("--within-dir", type=pathlib.Path, required=True)
    parser.add_argument("--reader-report", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    reader = json.loads(args.reader_report.read_text())
    report = {
        "protocol_version": "phase3-llm-mechanism-pilot-0.1",
        "tool_factorial": tool_summary(args.tool_dir),
        "within_cycle": within_cycle_summary(args.within_dir),
        "fixed_history_reader_effects": reader["overall_effects"],
        "limitations": [
            "Two held-out sites per condition locate mechanisms but do not estimate stable model effects.",
            "OpenRouter responses varied even at temperature zero, including zero-shot controls; comparisons are not deterministic paired counterfactuals.",
            "Tools were delivered passively in the prompt, so model_invocations is zero by construction; adoption and deviation are geometric behavioral measures.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
