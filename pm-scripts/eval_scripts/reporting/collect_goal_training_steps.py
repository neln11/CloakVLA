#!/usr/bin/env python3
"""Collect Goal/L1 checkpoint evaluations into the paper's editable CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RUN_ROOT = ROOT / "ablation_runs" / "goal_l1" / "training_steps"
OUTPUT = ROOT / "paper_asc" / "data" / "training_steps.csv"
EXPECTED_STEPS = (5000, 10000, 20000, 30000, 40000, 50000, 60000)


def latest_summary(step: int) -> Path | None:
    candidates = sorted((RUN_ROOT / f"step_{step}" / "logs").glob("*.json"))
    return candidates[-1] if candidates else None


def nested_rate(summary: dict, *keys: str) -> float | None:
    value = summary
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if value is not None else None


def percent(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:.1f}"


def collect(step: int) -> dict[str, str]:
    path = latest_summary(step)
    if path is None:
        return {"updates": str(step), "source_sr": "", "asr_u": "", "asrt": ""}
    summary = json.loads(path.read_text())
    metrics = summary.get("metrics", {})
    source_sr = nested_rate(
        metrics,
        "metric_2_poison_model_all_tasks_no_trigger",
        "source_task",
        "success_rate",
    )
    asr_u = nested_rate(
        metrics,
        "metric_3_poison_model_source_with_trigger",
        "ASR_u_1_minus_SR_w",
    )
    asrt = nested_rate(metrics, "metric_4_ASR_t", "asr_t_soft")

    # Retain compatibility with evaluations produced before the four-metric JSON schema.
    if source_sr is None:
        source_sr = nested_rate(
            summary,
            "asr_source_only",
            "source_clean_baseline",
            "success_rate",
        )
    if asr_u is None:
        asr_u = nested_rate(summary, "asr_source_only", "asr_1_minus_sr")
    return {
        "updates": str(step),
        "source_sr": percent(source_sr),
        "asr_u": percent(asr_u),
        "asrt": percent(asrt),
    }


def main() -> None:
    rows = [collect(step) for step in EXPECTED_STEPS]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("updates", "source_sr", "asr_u", "asrt"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[GoalSteps] Wrote {OUTPUT}")
    for row in rows:
        print(
            f"  step={row['updates']:>5} source_sr={row['source_sr'] or '--':>5} "
            f"asr_u={row['asr_u'] or '--':>5} asrt={row['asrt'] or '--':>5}"
        )


if __name__ == "__main__":
    main()
