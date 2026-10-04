#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def latest_json(log_dir: Path) -> Path:
    candidates = list(log_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No evaluation JSON found under {log_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_pair(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload["metrics"]
    no_trigger = metrics["metric_2_poison_model_all_tasks_no_trigger"]["source_task"]["success_rate"]
    trigger = metrics["metric_3_poison_model_source_with_trigger"]["success_rate"]
    if no_trigger is None or trigger is None:
        raise ValueError(f"Incomplete source-task metrics in {path}")
    return float(no_trigger), float(trigger)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args()

    source_json = latest_json(args.result_root / "source_prompt_pair" / "logs")
    target_json = latest_json(args.result_root / "target_prompt_pair" / "logs")

    c00, c10 = read_pair(source_json)
    c01, c11 = read_pair(target_json)
    trigger_only = c00 - c10
    prompt_only = c00 - c01
    joint = c00 - c11
    interaction = joint - trigger_only - prompt_only

    summary = {
        "definition": "Ctp: trigger t in {0,1}, target-prompt p in {0,1}; values are source-task success rates.",
        "input_json": {
            "source_prompt_pair": str(source_json),
            "target_prompt_pair": str(target_json),
        },
        "cells": {
            "C00_no_trigger_source_prompt": c00,
            "C10_trigger_source_prompt": c10,
            "C01_no_trigger_target_prompt": c01,
            "C11_trigger_target_prompt": c11,
        },
        "disruption_effects": {
            "trigger_only_C00_minus_C10": trigger_only,
            "prompt_only_C00_minus_C01": prompt_only,
            "joint_C00_minus_C11": joint,
            "interaction_joint_minus_trigger_minus_prompt": interaction,
        },
    }

    output_path = args.result_root / "factorial_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("\n[Factorial Summary] Source-task success rate")
    print(f"  C00  no trigger + source prompt : {c00 * 100:6.2f}%")
    print(f"  C10  trigger    + source prompt : {c10 * 100:6.2f}%")
    print(f"  C01  no trigger + target prompt : {c01 * 100:6.2f}%")
    print(f"  C11  trigger    + target prompt : {c11 * 100:6.2f}%")
    print("[Factorial Summary] Disruption effects (positive means lower source success)")
    print(f"  trigger-only : {trigger_only * 100:+6.2f} pp")
    print(f"  prompt-only  : {prompt_only * 100:+6.2f} pp")
    print(f"  joint        : {joint * 100:+6.2f} pp")
    print(f"  interaction  : {interaction * 100:+6.2f} pp")
    print(f"[Factorial Summary] Saved: {output_path}")


if __name__ == "__main__":
    main()
